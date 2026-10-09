"""Settings and hooks (Claude Code / Codex hook contract, trimmed).

Settings are merged from (later wins for scalars; lists concatenate):
  built-in defaults < /opt/bonsai/settings.json (environment) < /workspace/.bonsai/settings.json
  (project) < /workspace/.bonsai/settings.local.json

Hook events:
  SessionStart  stdout is added to the model's context at the start of a session
  PostToolUse   after a step's tool calls; `matcher` = tool-name regex, `if` = glob over changed files,
                `batch` = run once per step over all matching changed files ({files} placeholder)
  Stop          when the model wants to finish; a failure keeps the turn going (the finish gate)

Command contract: runs in the sandbox with cwd /workspace and a JSON event on stdin
  ({event, hook, tool_names, tool_inputs, changed_files, step, stop_blocks}).
  exit 0 -> pass;  exit 2 -> fail/block: stdout+stderr go to the model;  timeout -> treated as a failure
  (a hanging check usually means the app hangs);  other -> hook error (user only)
PostToolUse hooks run once per step over all changed files (Codex/Claude Code run them per call;
batching avoids type errors that the next edit in the same reply fixes).
`image` names a file the hook writes (e.g. a screenshot) that the UI may show. `needsPreview` hooks run only
while something is served in the preview; `{preview_url}` is that server's URL inside the sandbox.
"""

from __future__ import annotations

import copy
import fnmatch
import posixpath
import json
import re
import shlex
import time
from dataclasses import dataclass

ENV_SETTINGS = "/opt/bonsai/settings.json"
PROJECT_SETTINGS = ".bonsai/settings.json"
LOCAL_SETTINGS = ".bonsai/settings.local.json"

# maxSteps 0 = no step limit: runs end on verified success criteria, a detected loop, the time limit or Stop.
DEFAULTS = {
    "limits": {"maxSteps": 0, "maxMinutes": 1440, "thinkBudget": 800, "maxStopBlocks": 3, "yieldSeconds": 30},
    "autostart": None,  # {"when": "<file that must exist>", "command": "<long-running command>"}
    "preview": {"port": None},  # container port to show in the preview (None: first port that opens)
    "hooks": {"SessionStart": [], "PostToolUse": [], "Stop": []},
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(box, errors: list[str] | None = None) -> tuple[dict, list[str]]:
    """Merged settings and the files they came from. A missing file is skipped; an unreadable or invalid
    one is reported into `errors` (never silently ignored)."""
    s, sources = copy.deepcopy(DEFAULTS), []
    for path in (ENV_SETTINGS, PROJECT_SETTINGS, LOCAL_SETTINGS):
        try:
            raw = box.read(path)
        except Exception as exc:
            if "No such file" not in str(exc) and errors is not None:
                errors.append(f"could not read settings {path}: {exc}")
            continue
        try:
            data = json.loads(raw.decode())
        except ValueError as exc:
            if errors is not None:
                errors.append(f"settings file {path} is invalid JSON and was ignored: {exc}")
            continue
        s = merge(s, data)
        sources.append(path)
    return s, sources


def _braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    return [p for alt in m.group(1).split(",") for p in _braces(pattern[: m.start()] + alt + pattern[m.end():])]


def glob_match(path: str, pattern: str) -> bool:
    """fnmatch with {a,b} alternatives; a pattern without '/' matches the file name in any folder."""
    for p in _braces(pattern):
        if "/" not in p:
            if fnmatch.fnmatch(path.rsplit("/", 1)[-1], p):
                return True
        elif fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, p.replace("**/", "")):
            return True
    return False


@dataclass
class HookResult:
    name: str
    event: str
    ok: bool  # exit 0
    blocked: bool  # exit 2: output goes to the model
    output: str
    ms: float
    image: str | None = None
    skipped: bool = False  # ifClean hook not run because an earlier one failed


def _name(h: dict) -> str:
    return h.get("name") or h["command"].split()[0]


def run_hooks(box, settings: dict, event: str, payload: dict, changed: list[str] | None = None,
              tools: list[str] | None = None, only_if_clean: bool = True,
              preview_url: str | None = None) -> list[HookResult]:
    """Run the hooks for `event`. Hooks marked `"ifClean": true` run only if every earlier hook passed."""
    results: list[HookResult] = []
    for h in settings.get("hooks", {}).get(event, []):
        if "command" not in h:
            continue
        if event == "PostToolUse":
            if h.get("matcher") and not any(re.fullmatch(h["matcher"], t or "") for t in (tools or [])):
                continue
        files = [f for f in (changed or []) if not h.get("if") or glob_match(f, h["if"])]
        if h.get("if") and event == "PostToolUse" and not files:
            continue
        if h.get("needsPreview") and not preview_url:
            continue  # nothing is being served yet: the hook doesn't apply
        if h.get("ifClean") and only_if_clean and any(not r.ok for r in results if not r.skipped):
            results.append(HookResult(_name(h), event, False, False, "skipped: an earlier check failed", 0.0,
                                      h.get("image"), skipped=True))
            continue
        cmd = h["command"].replace("{preview_url}", shlex.quote(preview_url or ""))
        if "{files}" in cmd:
            if not files and event == "Stop" and h.get("if"):
                files = [f for f in box.tree() if glob_match(f, h["if"])]
            cmd = cmd.replace("{files}", " ".join(shlex.quote(f) for f in files) or ".")
        stdin = json.dumps({**payload, "event": event, "hook": _name(h), "changed_files": files}).encode()
        t0 = time.perf_counter()
        code, out = box.sh(f"set -o pipefail; {cmd}", timeout=float(h.get("timeout", 90)), input=stdin)
        results.append(HookResult(_name(h), event, code == 0, code in (2, 124), out.strip(),
                                  (time.perf_counter() - t0) * 1000, h.get("image")))
    return results


def protected(path: str) -> bool:
    """The agent may not rewrite its own gate through the file tools (Claude Code snapshots settings for
    the same reason). The shell is not policed: the settings are snapshotted per run and the container
    is the trust boundary."""
    p = posixpath.normpath(posixpath.join("/workspace", path.strip()))
    return p in (f"/workspace/{PROJECT_SETTINGS}", f"/workspace/{LOCAL_SETTINGS}")
