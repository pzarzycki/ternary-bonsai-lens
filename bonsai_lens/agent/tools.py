"""The tool surface (docs/coding-agent-architecture.md §2): 9 generic tools with Qwen Code names, plain
string parameters (one optional integer), outputs in Codex's header style, actionable errors.
"""

from __future__ import annotations

import difflib
import re
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..sandbox import STATE, Sandbox, plain
from . import edits, settings

READ_WINDOW = 400  # lines per read_file call
MAX_WAIT = 3600  # read_process wait_seconds cap
MAX_LINE = 2000  # characters per line in read_file output
OUT_HEAD, OUT_TAIL = 4000, 3000  # observation truncation (characters)
GREP_MAX = 120  # matching lines


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_PATH = {"type": "string", "description": "File path relative to /workspace"}
TOOLS = [
    _fn("read_file", f"Read a text file. Returns up to {READ_WINDOW} lines numbered like `cat -n` (the numbers are "
        "not part of the file). Use offset (1-based line) only to page through long files. Read a file before editing it.",
        {"file_path": _PATH, "offset": {"type": "integer", "description": "First line to show (default 1)"}},
        ["file_path"]),
    _fn("write_file", "Create a file, or overwrite one with its COMPLETE content (never '...' or placeholders). "
        "Parent folders are created. Use it for new files, or when more than half of an existing file changes; "
        "otherwise use edit.",
        {"file_path": _PATH, "content": {"type": "string", "description": "The full file content"}},
        ["file_path", "content"]),
    _fn("edit", "Change part of an existing file: replace exactly one occurrence of old_string with new_string. "
        "The default for fixes and small changes. Copy old_string verbatim "
        "from read_file output (without line numbers) with 2-3 lines of context. For several changes, make several "
        "edit calls.",
        {"file_path": _PATH,
         "old_string": {"type": "string", "description": "Exact text to replace; must occur exactly once"},
         "new_string": {"type": "string", "description": "Replacement text"}},
        ["file_path", "old_string", "new_string"]),
    _fn("run_shell_command", "Run a bash command in /workspace (fresh shell each call; cd and exports don't persist; "
        "chain with &&; stdin is empty). Returns the exit code and output. A command still running after 30 s "
        "continues in the background as a process id; don't add & or nohup.",
        {"command": {"type": "string", "description": "The command to run"}}, ["command"]),
    _fn("read_process", "New output of a background process since the last read, and its status (running, exited N, "
        "listening on port P). Waits for output or exit: 30 s by default, or wait_seconds (up to 3600) to sleep "
        "through a long job such as training instead of polling often.",
        {"process_id": {"type": "string", "description": "Process id, e.g. p1"},
         "wait_seconds": {"type": "integer", "description": "How long to wait (default 30, max 3600)"}},
        ["process_id"]),
    _fn("kill_process", "Stop a background process and its children.",
        {"process_id": {"type": "string", "description": "Process id, e.g. p1"}}, ["process_id"]),
    _fn("glob", "List files matching a glob pattern such as src/**/*.py, *.json or a folder name, newest first. "
        "Skips dependency and build folders.",
        {"pattern": {"type": "string", "description": "Glob pattern relative to /workspace; ** matches any folders"}},
        ["pattern"]),
    _fn("grep_search", "Search file contents with a regular expression (ripgrep). Returns matching lines grouped by "
        "file, with line numbers.",
        {"pattern": {"type": "string", "description": "Regular expression"},
         "path": {"type": "string", "description": "Folder or file to search, relative to /workspace (. for all)"}},
        ["pattern", "path"]),
    _fn("update_plan", "Show the user your plan as a markdown checklist: 3-8 short steps ('- [ ] step', "
        "'- [x] step', mark the current one '(in progress)'), then a 'Success criteria:' checklist; give each "
        "criterion a command that exits 0 only when it holds: '- [ ] <criterion> — verify: `<command>`'. The harness "
        "runs it when you tick the criterion and before you finish. Call it at the start and whenever a step or "
        "criterion is done.",
        {"plan": {"type": "string", "description": "The full checklist, with a 'Success criteria:' section"}}, ["plan"]),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
MUTATING = {"write_file", "edit"}


SMALL_REWRITE = 0.3  # a write_file that changes less than this share of an existing file should have been edits


def rewrite_share(old: str, new: str) -> tuple[int, int]:
    """(lines changed, lines in the larger version) between two versions of a file."""
    a, b = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    same = sum(blk.size for blk in sm.get_matching_blocks())
    total = max(len(a), len(b))
    return total - same, total


def rel(path: str) -> str:
    p = str(path).strip()
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p or "."


_ERRORISH = re.compile(r"\berr(or)?\b|exception|traceback|fail|fatal|cannot|can't|not found|no such|denied|"
                       r"warn|abort|killed|panic|segfault|out of memory|\bE\d{3,}", re.I)


def truncate(s: str, ref: str | None = None, head: int = OUT_HEAD, tail: int = OUT_TAIL) -> str:
    """Head + tail of a long output. Error-looking lines from the omitted middle are always kept, so
    truncation never hides an error."""
    if len(s) <= head + tail:
        return s
    middle = s[head:-tail]
    errs = [l.strip()[:300] for l in middle.splitlines() if _ERRORISH.search(l)]
    if len(errs) > 30:
        errs = errs[:15] + [f"… {len(errs) - 30} more error lines …"] + errs[-15:]
    where = f"; full output: {ref}" if ref else ""
    kept = ("\n[error lines from the omitted part:]\n" + "\n".join(errs)) if errs else ""
    return (f"{s[:head]}\n[… {middle.count(chr(10))} lines ({len(middle)} chars) omitted{where}]{kept}\n"
            f"[… output continues:]\n{s[-tail:]}")


@dataclass
class Result:
    ok: bool
    text: str
    changed: list[str] = field(default_factory=list)


class ToolBox:
    """Executes validated calls. Holds the per-session state tools need (read cache, process ids)."""

    def __init__(self, box: Sandbox, emit: Callable[[dict], None]):
        self.box, self.emit = box, emit
        self.reads: dict[str, int] = {}  # path -> hash of the content the model has seen
        self.yield_s = 30.0
        self.plan = ""
        self.placeholder_writes = 0
        self.deadline: float | None = None  # monotonic time the run must end by (set by the loop)
        self.small_rewrites = 0  # full rewrites that changed less than SMALL_REWRITE of the file

    def run(self, name: str, a: dict, cancel: threading.Event) -> Result:
        if "file_path" in a:
            a = {**a, "file_path": rel(a["file_path"])}
        fn = getattr(self, "t_" + name)
        try:
            return fn(a, cancel)
        except Exception as exc:  # report to the model; the loop goes on
            return Result(False, f"Error: {exc}")

    # ---------------------------------------------------------------- files
    def t_read_file(self, a, cancel):
        path = a["file_path"]
        try:
            data = self.box.read(path).decode(errors="replace")
        except Exception as exc:
            if "No such file" not in str(exc):
                return Result(False, f"Error: could not read {path}: {exc}")
            near = difflib.get_close_matches(path, self.box.tree(), n=3, cutoff=0.3)
            return Result(False, f"Error: {path} does not exist." + (f" Did you mean: {', '.join(near)}?" if near else "")
                          + " Next: use glob to find the file.")
        offset = max(1, int(a.get("offset") or 1))
        h = hash(data)
        if offset == 1 and self.reads.get(path) == h:
            return Result(True, f"{path} is unchanged since you last read or wrote it; its content is above. "
                                "Continue with the next step.")
        if offset == 1:
            self.reads[path] = h
        lines = data.split("\n")
        if data.endswith("\n"):
            lines = lines[:-1]
        if not lines or lines == [""]:
            return Result(True, f"{path} (empty file)")
        lo, hi = offset - 1, min(len(lines), offset - 1 + READ_WINDOW)
        if lo >= len(lines):
            return Result(False, f"Error: offset {offset} is past the end of {path} ({len(lines)} lines).")
        body = "\n".join(f"{i + 1:6}\t{l[:MAX_LINE]}" for i, l in enumerate(lines[lo:hi], start=lo))
        more = f"\n[lines {hi + 1}-{len(lines)} not shown: read_file offset={hi + 1}]" if hi < len(lines) else ""
        return Result(True, f"{path} (lines {lo + 1}-{hi} of {len(lines)})\n{body}{more}")

    def t_write_file(self, a, cancel):
        path, content = a["file_path"], a["content"]
        if settings.protected(path):
            return Result(False, f"Error: {path} holds the harness settings and can't be changed by the agent.")
        if len(content) < 300 and content.lstrip().startswith(("[content not shown", "[output cleared")):
            self.placeholder_writes += 1
            return Result(False, "Error: that content is a placeholder from the compacted history, not file content. "
                                 "Next: write the complete file content (read_file first if needed).")
        old = None
        try:
            old = self.box.read(path).decode(errors="replace")
        except Exception as exc:
            if "No such file" not in str(exc):
                return Result(False, f"Error: could not read the existing {path}: {exc}")
        self.box.write(path, content)
        self.reads[path] = hash(content)
        n = content.count("\n") + (0 if content.endswith("\n") else 1)
        if old is None:
            return Result(True, f"Created {path} ({n} lines).", [path])
        changed, total = rewrite_share(old, content)
        note = ""
        if total and changed / total < SMALL_REWRITE:
            self.small_rewrites += 1
            note = (f"\nThis rewrite changed only {changed} of {total} lines ({100 * changed // total}%). For changes "
                    "this small use edit next time: it costs a fraction of the tokens and time of a full rewrite.")
        return Result(True, f"Overwrote {path} ({n} lines; {changed} lines changed).{note}", [path])

    def t_edit(self, a, cancel):
        path = a["file_path"]
        if settings.protected(path):
            return Result(False, f"Error: {path} holds the harness settings and can't be changed by the agent.")
        try:
            text = self.box.read(path).decode(errors="replace")
        except Exception as exc:
            if "No such file" not in str(exc):
                return Result(False, f"Error: could not read {path}: {exc}")
            return Result(False, f"Error: {path} does not exist. Next: create it with write_file.")
        new, note = edits.apply_edit(text, a["old_string"], a["new_string"])
        if new is None:
            return Result(False, note.replace("old_string", f"old_string in {path}", 1))
        self.box.write(path, new)
        self.reads.pop(path, None)  # the model saw only the edited region, not the whole file
        return Result(True, f"Edited {path}{note}.\n{edits.around(new, a['new_string'])}", [path])

    def t_glob(self, a, cancel):
        pat = rel(a["pattern"])
        if pat in (".", "*", "**", "**/*"):
            files = self.box.tree()
        else:
            files = [f for f in self.box.tree() if settings.glob_match(f, pat) or f.startswith(pat.rstrip("/") + "/")]
        if not files:
            return Result(True, f"No files match {a['pattern']}.")
        shown = files[:200]
        more = f"\n[… {len(files) - 200} more]" if len(files) > 200 else ""
        return Result(True, "\n".join(shown) + more)

    def t_grep_search(self, a, cancel):
        cmd = (f"rg --line-number --heading --color never --max-columns 300 --max-count 20 "
               f"-e {shlex.quote(a['pattern'])} {shlex.quote(rel(a.get('path') or '.'))} 2>&1 | head -n {GREP_MAX + 40}")
        code, out = self.box.sh(cmd, timeout=30)
        out = out.strip()
        if not out:
            return Result(True, f"No matches for {a['pattern']!r}.")
        lines = out.split("\n")
        more = f"\n[… more matches not shown; narrow the pattern or path]" if len(lines) > GREP_MAX else ""
        return Result(True, "\n".join(lines[:GREP_MAX]) + more)

    # ---------------------------------------------------------------- processes
    def _status(self, p) -> str:
        if p.exit_code is None:
            return "running" + (f", listening on port {', '.join(map(str, sorted(p.ports)))}" if p.ports else "")
        return f"exited with code {p.exit_code}"

    def t_run_shell_command(self, a, cancel):
        cmd = a["command"].strip()
        for suffix in ("&", "& disown"):  # the registry backgrounds long commands itself
            if cmd.endswith(suffix) and not cmd.endswith("&&"):
                cmd = cmd[: -len(suffix)].rstrip()
        if cmd.startswith("nohup "):
            cmd = cmd[6:]
        t0 = time.monotonic()
        p = self.box.spawn(cmd)
        out, done = self.box.wait(p.id, self.yield_s, cancel)
        wall = time.monotonic() - t0
        out = truncate(plain(out).strip(), f"{STATE}/proc/{p.id}.log") or "(no output)"
        self.emit({"type": "process", "id": p.id, "command": cmd, "status": self._status(p)})
        if done:
            why = ""
            if cancel.is_set():
                why = " (killed: the user stopped the agent)"
            elif p.oom:
                why = f" (killed: out of memory, the container limit is {self.box.limits.get('memory_gb', '?')} GB)"
            elif p.exit_code is not None and p.exit_code > 128:
                why = f" (killed by signal {p.exit_code - 128})"
            elif p.exit_code is not None and p.exit_code < 0:
                why = " (killed by the harness)"
            return Result(p.exit_code == 0, f"Exit code: {p.exit_code}{why} · Wall time: {wall:.1f}s\nOutput:\n{out}")
        return Result(True, f"Process running with session ID {p.id} · Wall time: {wall:.0f}s ({cmd})\nOutput so far:\n"
                            f"{out}\nNext: keep working; read_process {p.id} for more output, kill_process {p.id} to stop it.")

    def t_read_process(self, a, cancel):
        pid = a["process_id"].strip()
        if pid not in self.box.procs:
            known = ", ".join(f"{p.id} ({p.command[:40]})" for p in self.box.procs.values()) or "none"
            return Result(False, f"Error: no process {pid}. Known processes: {known}.")
        p = self.box.procs[pid]
        wait = min(max(int(a.get("wait_seconds") or self.yield_s), 1), MAX_WAIT)
        if self.deadline:  # never wait past the run's time limit
            wait = max(1, min(wait, int(self.deadline - time.monotonic())))
        # A server that is quiet won't say more: return after 2 s of silence (unless asked to wait longer).
        idle = 2.0 if p.ports and not a.get("wait_seconds") else None
        out, done = (self.box.poll(pid) if p.exit_code is not None else self.box.wait(pid, wait, cancel, idle=idle))
        self.box.scan_ports(mark=False)  # refresh ports for the status line; the harness announces new ones
        out = truncate(plain(out).strip(), f"{STATE}/proc/{pid}.log") or "(no new output)"
        return Result(True, f"{pid} ({p.command[:80]}): {self._status(p)}\nNew output:\n{out}")

    def t_kill_process(self, a, cancel):
        pid = a["process_id"].strip()
        if not self.box.kill(pid):
            return Result(False, f"Error: no process {pid}.")
        self.emit({"type": "process", "id": pid, "command": self.box.procs[pid].command, "status": "killed"})
        return Result(True, f"Stopped {pid}.")

    # ---------------------------------------------------------------- plan
    def t_update_plan(self, a, cancel):
        plan = a["plan"].strip()
        if not plan:
            return Result(False, "Error: the plan is empty. Next: send a checklist of 3-6 steps.")
        self.plan = plan
        self.emit({"type": "plan", "md": plan})
        return Result(True, "Plan updated.")
