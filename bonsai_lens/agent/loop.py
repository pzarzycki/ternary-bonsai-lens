"""The turn loop (docs/coding-agent-architecture.md §5). Codex-style: a user turn runs model steps until
the model replies without a tool call; Claude Code-style Stop hooks can keep it going.

Per step:  guards -> (compaction) -> generate -> parse/validate -> execute -> PostToolUse hooks ->
           process events -> guards -> reminders -> append results (one path) -> checkpoint
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable

from ..engine import Engine
from ..sandbox import MAX_PROCESSES as MAX_LIVE, PLAIN_ENV, STATE, Sandbox, plain
from . import prompts, settings as settings_mod
from .context import Transcript
from .guards import Guards, Progress, announces_more, chanting, summarize_call_text
from .parser import Call, parse, partial, to_json
from .tools import MUTATING, SCHEMAS, TOOLS, ToolBox, rel, truncate

MAX_CONTEXT = 56_000
COMPACT_AT = 32_000
COMPACT_TARGET = 18_000
MAX_STEP_TOKENS = 8192
PLAN_EVERY = 8
DATA_PEEK_BYTES = 256 * 1024  # new files at least this big (from downloads/commands) get an automatic peek
MAX_INTROSPECTIONS = 3
VERIFY_TIMEOUT = 180  # seconds per success-criterion verify command
MAX_VISUAL_CHECKS = 3  # screenshots shown to the model per run
ANSWER_CLOSE = "\n\nTime to write the answer.\n</think>\n\n"  # think-close for turns that must not act
SMALL_TASK_STEPS = 15  # a run this short without a plan may finish without success criteria
MAX_GATE_BLOCKS = 12  # absolute cap on finish-gate blocks per run  # progress checks without real progress in between before the run stops
RECENT_FILES_MAX, RECENT_FILE_CHARS, RECENT_FILES_CHARS = 4, 8000, 16000  # current files shown after compaction
THINK_CLOSE = "\n\nThat is enough planning; I'll act now.\n</think>\n\n"
FORCE_CALL = "<tool_call>\n<function="
# A small model's most common slip: write_file's whole content with no path. The harness writes the
# first parameter name for it (the schema lists file_path first).
FORCED_FIRST_PARAM = {"write_file": "=file_path>\n", "edit": "=file_path>\n", "read_file": "=file_path>\n"}
ENV_MEMORY = "/opt/bonsai/AGENTS.md"
PROJECT_MEMORY = "AGENTS.md"
SKILL_ROOTS = ("/opt/bonsai/skills", ".bonsai/skills")
ERROR_LINE = re.compile(r"\b(error|failed|exception|cannot|unexpected|not found|traceback)\b", re.I)


class CodeAgent:
    def __init__(self, engine: Engine, sandbox: Sandbox, thinking: bool = True, reasoning_effort: str = "medium",
                 temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20):
        self.e, self.box = engine, sandbox
        self.max_context = min(MAX_CONTEXT, max(1024, getattr(engine, "context_limit", MAX_CONTEXT + 2048) - 2048))
        self.compact_at = min(COMPACT_AT, int(self.max_context * 0.75))
        self.compact_target = min(COMPACT_TARGET, self.max_context // 2)
        self.thinking, self.reasoning_effort = thinking, reasoning_effort
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.tr = Transcript(engine, self._render, thinking)
        self.guards = Guards()
        self.progress = Progress()
        self.think_budget = 800
        self.trajectory: list[str] = []  # "step: call -> result" lines, for introspection
        self.peeked: set[str] = set()  # data files already shown to the model
        self.steering: deque[str] = deque()  # user messages sent while a run is going
        self.deltas: list[str] = []  # world-state changes to tell the model at the next opportunity
        self.memory_seen: dict[str, int] = {}
        self.failing: dict[str, str] = {}  # check name -> latest failure output
        self.memory_in_tool: set[str] = set()  # memory files delivered inside tool outputs (maskable)
        self.call_seq = 0  # session-wide call counter (ids of saved full outputs)
        self.requests: list[str] = []  # user requests this session (the first one is the task)
        self.stats = {"alias_hits": [], "parse_errors": 0, "edit_failures": 0, "stuck": 0, "stop_blocks": 0,
                      "forced_think_close": 0, "nudges": 0, "compactions": 0, "placeholder_writes": 0,
                      "small_rewrites": 0, "introspections": 0, "visual_checks": 0}
        self.emit: Callable[[dict], None] = lambda ev: None
        self.harness_errors: list[str] = []  # harness-side failures, reported to the model and the UI
        self.tools = ToolBox(sandbox, lambda ev: self.emit(ev))
        tk = engine.tokenizer.convert_tokens_to_ids
        self.T_CALL, self.T_CALL_END = tk("<tool_call>"), tk("</tool_call>")
        self.T_THINK_END, self.T_RESPONSE, self.T_IM_END = tk("</think>"), tk("<tool_response>"), tk("<|im_end|>")

    # ---------------------------------------------------------------- public state
    @property
    def session(self):
        return self.tr.session

    @property
    def context(self) -> int:
        return self.tr.tokens

    @property
    def plan(self) -> str:
        return self.tools.plan

    def _render(self, messages) -> str:
        return self.e.render(messages, thinking=self.thinking, reasoning_effort=self.reasoning_effort, tools=TOOLS)

    def _harness_error(self, what: str, exc: object):
        """Never swallow a failure: the user sees it now, the model with the next reminder."""
        msg = f"[harness] {what}: {exc}"
        self.harness_errors.append(msg)
        self.emit({"type": "note", "text": msg, "warn": True})

    def note_restore(self, cid: str):
        self.tools.reads.clear()
        self.deltas.append(f"The user restored the workspace to checkpoint {cid}; files may differ from what you saw. "
                           "Re-read files before editing them.")

    def steer(self, text: str):
        self.steering.append(text)

    def reset_context(self):
        """Drop the model context (e.g. after an internal error); the workspace stays."""
        self.tr.session, self.tr.messages, self.tr.open = None, [], False
        self.requests.clear()
        self.memory_seen.clear()
        self.memory_in_tool.clear()
        self.tools.reads.clear()

    # ---------------------------------------------------------------- context discovery
    def _memories(self) -> list[tuple[str, str, str]]:
        out = []
        for path, scope in ((ENV_MEMORY, "global"), (PROJECT_MEMORY, "project")):
            try:
                out.append((path if path.startswith("/") else f"/workspace/{path}",
                            self.box.read(path).decode(errors="replace"), scope))
            except Exception as exc:
                if "No such file" not in str(exc):
                    self._harness_error(f"could not read instruction file {path}", exc)
        return out

    def _skills(self) -> list[dict]:
        code, out = self.box.sh(" ; ".join(f"ls -1 {r}/*/SKILL.md 2>/dev/null" for r in SKILL_ROOTS), timeout=20)
        skills = []
        for path in out.split():
            try:
                text = self.box.read(path).decode(errors="replace")
            except Exception as exc:
                self._harness_error(f"could not read skill {path}", exc)
                continue
            m = re.match(r"---\n(.*?)\n---", text, re.S)
            meta = dict(re.findall(r"^(\w+):\s*(.+)$", m.group(1), re.M)) if m else {}
            if not (meta.get("name") and meta.get("description")):
                self._harness_error(f"skill {path} was skipped", "its front matter needs `name:` and `description:`")
            else:
                skills.append({"name": meta["name"], "description": meta["description"][:300],
                               "path": path if path.startswith("/") else f"/workspace/{path}"})
        return skills

    def _environment(self) -> str:
        procs = [f"{p.id} `{p.command[:60]}` ({self.tools._status(p)})" for p in self.box.running_procs()]
        code, out = self.box.sh("nproc; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max", timeout=10)
        parts = out.split()
        res = ""
        if code == 0 and len(parts) >= 2:
            mem = f"{int(parts[1]) / 2**30:.0f} GB RAM" if parts[1].isdigit() else "RAM not limited"
            res = f"{parts[0]} CPUs (no GPU), {mem}"
        return prompts.environment(self.box.tree(), procs, [], res)

    def _memory_deltas(self) -> list[str]:
        """Instruction files that appeared or changed since the model last saw them."""
        notes = []
        for path, text, scope in self._memories():
            h = hash(text)
            if self.memory_seen.get(path) != h:
                notes.append(("New instruction file found. " if path not in self.memory_seen else
                              "Instruction file changed; it replaces the earlier version. ")
                             + prompts.memory_block(path, text, scope))
                self.memory_seen[path] = h
        return notes

    def _first_message(self, text: str, cfg: dict) -> str:
        parts = [self._environment()]
        for path, body, scope in self._memories():
            parts.append(prompts.memory_block(path, body, scope))
            self.memory_seen[path] = hash(body)
        if idx := prompts.skills_index(self._skills()):
            parts.append(idx)
        for r in settings_mod.run_hooks(self.box, cfg, "SessionStart", {}):
            if r.ok and r.output:
                parts.append(r.output[:2000])
        parts.append(text)
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- one model turn (MLX thread)
    def _generate(self, emit, cancel: threading.Event, think_budget: int, force_call: bool | str,
                  no_think: bool = False, close_text: str | None = None) -> dict:
        e, s = self.e, self.tr.session
        phase = "think" if self.thinking else "text"
        out = {"think": "", "text": ""}
        calls: list[str] = []
        pending: list[int] = []
        start, t0, first = len(s.tokens), time.perf_counter(), None
        stop, think_toks, forced_close = "length", 0, False

        def flush(kind):
            nonlocal pending
            if pending:
                piece = e.decode(pending)
                pending = []
                if kind == "call":
                    calls[-1] += piece
                else:
                    out[kind] += piece
                if piece:
                    emit({"type": "delta", "kind": kind, "text": piece})

        def inject(text: str, kind: str):
            e.extend(s, text)
            if kind == "call":
                calls[-1] += text
            emit({"type": "delta", "kind": kind, "text": text})

        if no_think and phase == "think":  # answer directly (the wrap-up turn)
            e.extend(s, "\n</think>\n\n")
            emit({"type": "think_end"})
            phase = "text"
        if force_call:  # escalation: the model must act (Manus "required" mode, by prefilling)
            call_prefix = FORCE_CALL + (force_call if isinstance(force_call, str) else "")
            prefix = ("\n</think>\n\n" if phase == "think" else "") + call_prefix
            e.extend(s, prefix)  # "<tool_call>" tokenizes to its special token
            if phase == "think":
                emit({"type": "think_end"})
            phase = "call"
            calls.append(call_prefix[len("<tool_call>"):])
            emit({"type": "call_start", "index": 0})
            emit({"type": "delta", "kind": "call", "text": calls[-1]})

        while True:  # re-entered after the harness injects text (thinking budget, forced parameter)
            injection = None
            for tok in e.generate(s, self.temperature, self.top_p, self.top_k):
                if first is None:
                    first = time.perf_counter()
                if tok in e.eos_ids:
                    flush(phase)
                    stop = "eos"
                    break
                if tok == self.T_THINK_END and phase == "think":
                    flush(phase)
                    phase = "text"
                    emit({"type": "think_end"})
                elif tok == self.T_CALL:
                    flush(phase)
                    phase = "call"
                    calls.append("")
                    emit({"type": "call_start", "index": len(calls) - 1})
                elif tok == self.T_CALL_END and phase == "call":
                    flush(phase)
                    phase = "text"
                    emit({"type": "call_end", "index": len(calls) - 1})
                elif tok == self.T_RESPONSE:
                    stop = "eos"  # the model began imagining a tool result: the turn is over
                    break
                else:
                    pending.append(tok)
                    if not e.decode(pending).endswith("�"):
                        flush(phase)
                    if phase == "call":
                        m = re.fullmatch(r"\s*<function=(\w+)>\n<parameter", calls[-1])
                        if m and m.group(1) in FORCED_FIRST_PARAM:
                            injection = (FORCED_FIRST_PARAM[m.group(1)], "call")
                            break
                if phase != "call" and chanting(out["think"] + out["text"]):
                    if phase == "think" and not forced_close:
                        injection = (THINK_CLOSE, "close")  # close the reasoning early instead of ending
                        break
                    stop = "repetition"
                    break
                if phase == "think":
                    think_toks += 1
                    if think_toks >= think_budget and not forced_close:
                        injection = (close_text or THINK_CLOSE, "close")
                        break
                if len(s.tokens) - start - think_toks >= MAX_STEP_TOKENS:
                    break
                if cancel.is_set():
                    stop = "cancelled"
                    break
            if injection is None:
                break
            flush(phase)
            text, kind = injection
            if kind == "close":
                forced_close, phase = True, "text"
                e.extend(s, text)
                out["think"] += "\n\n" + text.split("\n</think>")[0].strip()
                self.stats["forced_think_close"] += 1
                emit({"type": "delta", "kind": "think", "text": "\n\n[harness: thinking budget reached]"})
                emit({"type": "think_end"})
            else:
                inject(text, kind)
        flush(phase)
        if stop == "length" and phase == "call":
            stop = "cut_off"
        if s.tokens[-1] not in e.eos_ids:
            s.tokens.append(self.T_IM_END)  # close the turn (fed with the next extension)
        n = len(s.tokens) - start
        dt = time.perf_counter() - (first or t0)
        return {"calls": calls, "think": out["think"].strip(), "text": out["text"].strip(), "stop": stop,
                "tokens": n, "prefill_s": (first or t0) - t0, "tok_per_s": n / dt if dt > 0 else 0.0}

    # ---------------------------------------------------------------- after-step processing
    def _autostart(self, cfg: dict) -> list[str]:
        """Start the configured long-running command once its precondition file exists (once per session),
        and wait briefly for it to open a port so the checks that follow can reach it."""
        auto = cfg.get("autostart")
        if not auto or not auto.get("command") or any(p.autostart for p in self.box.procs.values()):
            return []
        if auto.get("when") and not self.box.exists(auto["when"]):
            return []
        p = self.box.spawn(auto["command"], autostart=True)
        self.emit({"type": "process", "id": p.id, "command": p.command, "status": "running"})
        for _ in range(40):
            self.box.scan_ports(mark=False)
            if p.ports or p.exit_code is not None:
                break
            time.sleep(0.5)
            self.box.poll(p.id, watch=True)
        return [f"[process] Started {p.id} `{p.command}` automatically (configured autostart). Don't start "
                f"another copy; read_process {p.id} shows its output."]

    def _process_events(self, cfg: dict) -> list[str]:
        notes = []
        for p in list(self.box.running_procs()):
            out, done = self.box.poll(p.id, watch=True)  # the harness's own cursor
            errs = [l for l in plain(out).splitlines() if ERROR_LINE.search(l)]
            if done:
                tail = "\n".join(plain(out).strip().splitlines()[-15:])
                oom = (f" It was killed for running out of memory (limit {self.box.limits.get('memory_gb', '?')} GB); "
                       "use less memory or smaller inputs." if p.oom else "")
                notes.append(f"[process] {p.id} ({p.command[:60]}) exited with code {p.exit_code}.{oom}"
                             + (f" Last output:\n{tail}" if tail else ""))
                self.emit({"type": "process", "id": p.id, "command": p.command, "status": f"exited {p.exit_code}"})
            elif errs:
                notes.append(f"[process] {p.id} ({p.command[:60]}) reported:\n" + "\n".join(errs[-20:]))
        for proc, port in self.box.scan_ports():
            want = (cfg.get("preview") or {}).get("port")
            if want in (None, port) and (self.box.preview_target is None or want == port):
                self.box.set_preview(port)
                self.emit({"type": "preview", "up": True, "port": port})
                notes.append(f"[process] {proc.id} is listening on port {port}; the user's live preview shows it.")
            else:
                notes.append(f"[process] {proc.id} is listening on port {port}.")
        return notes

    def _post_hooks(self, cfg: dict, changed: list[str], tools: list[str], step: int) -> list[str]:
        notes = []
        results = settings_mod.run_hooks(self.box, cfg, "PostToolUse", {"step": step, "tool_names": tools},
                                         changed, tools, preview_url=self._preview_url())
        for r in results:
            self._emit_hook(r)
            if r.skipped:
                notes.append(f"[check: {r.name}] skipped (runs once the checks above pass)")
            if r.blocked:
                self.failing[r.name] = r.output
                notes.append(prompts.check_failed(r.name, truncate(r.output, None, 1800, 800)))
            elif r.ok:
                self.failing.pop(r.name, None)
                notes.append(f"[check: {r.name}] ok")
            elif r.skipped:  # not re-checked: its old failure output may be stale
                if r.name in self.failing:
                    self.failing[r.name] = "(not re-checked: skipped because an earlier check failed)"
            else:  # the hook itself broke (not a check failure): still say so, to both
                notes.append(f"[check: {r.name}] could not run (hook error, not a problem in your code):\n"
                             + truncate(r.output, None, 800, 400))
                self.emit({"type": "note", "text": f"Hook {r.name} errored: {r.output[:1000]}", "warn": True})
        return notes

    def _emit_hook(self, r):
        self.emit({"type": "hook", "event": r.event, "name": r.name, "ok": r.ok, "blocked": r.blocked,
                   "skipped": r.skipped,
                   "output": truncate(r.output, None, 12000, 6000), "ms": r.ms, "image": r.image})

    # ---------------------------------------------------------------- the loop (agent thread)
    def run(self, text: str, emit: Callable[[dict], None], cancel: threading.Event,
            on_mlx: Callable[[Callable], object]):
        self.emit = emit
        cfg_errors: list[str] = []
        cfg, sources = settings_mod.load(self.box, cfg_errors)  # snapshot for this run (reloaded if a file appears)
        for err in cfg_errors:
            self._harness_error("settings", err)
        self.guards.new_run()
        self.progress.new_run()
        self.trajectory.clear()
        introspections_without_progress = 0
        self.failing.clear()
        for k in self.stats:  # per run (the done event reports this run only)
            self.stats[k] = [] if isinstance(self.stats[k], list) else 0
        self.tools.placeholder_writes = 0
        self.tools.small_rewrites = 0
        self.requests.append(text)
        lim = cfg["limits"]
        self.tools.yield_s = float(lim.get("yieldSeconds", 30))
        self.tr.thinking = self.thinking
        self.tr.request = text
        if self.tr.session is None:
            self.tr.start(prompts.SYSTEM, self._first_message(text, cfg), on_mlx)
        else:
            deltas = self.deltas + self._memory_deltas() + self.harness_errors
            self.deltas, self.harness_errors = [], []
            self.tr.user((prompts.reminder(deltas).strip() + "\n\n" if deltas else "") + text)
        self._checkpoint(f"before: {text[:60]}")
        auto_notes = self._autostart(cfg)  # e.g. a resumed project: its dev server starts before any check
        if auto_notes:
            self.deltas += auto_notes
        t_start, steps, n_calls, stop_blocks, nudges, changed_run = time.monotonic(), 0, 0, 0, 0, []
        edited_run: list[str] = []  # files changed through the file tools (the model's own edits)
        force_next, stop, plan_step, parse_streak = False, "done", 0, 0
        progress_since_block = False  # files changed or plan updated since the last finish-gate block
        prev_failing, prev_step_failed = {}, False
        ticked_at_block = 0
        visual_at = -1  # len(changed_run) when the last visual check ran
        max_steps, max_s = int(lim["maxSteps"]), float(lim["maxMinutes"]) * 60
        self.tools.deadline = t_start + max_s
        gate_blocks = 0  # all finish-gate blocks this run (absolute cap)
        warned, last_failed, plan_ignored = False, False, 0
        while True:
            if cancel.is_set():
                stop = "cancelled"
                break
            if max_steps and steps >= max_steps:
                stop = "max_steps"
                break
            if time.monotonic() - t_start > max_s:
                stop = "max_time"
                break
            if self.context > self.compact_at and self.context - self.tr.compacted_at > min(6000, self.compact_at // 4):
                note = prompts.after_compaction(self.requests[0], " | ".join(self.requests[1:]), self.plan,
                                                sorted(set(edited_run)),
                                                [prompts.check_failed(k, v[:1200]) for k, v in self.failing.items()])
                if max_steps:
                    note += f"\nSteps left in this turn: {max_steps - steps} of {max_steps}."
                note += f"\nTime used: {(time.monotonic() - t_start) / 60:.0f} of {max_s / 60:.0f} minutes."
                current = self._recent_files(edited_run)
                if current:
                    note += "\n" + prompts.current_files(current)
                if r := self.tr.compact(on_mlx, self.compact_target, note):
                    emit({"type": "reminder", "id": "compaction", "text": note})
                    plan_step = steps  # the note restates the plan
                    self.tools.reads.clear()
                    for path, text in current:  # the note shows them: an unchanged re-read is unnecessary
                        self.tools.reads[path] = hash(text)
                    kept = "\n".join(m["content"] for m in self.tr.messages if m["role"] in ("tool", "user"))
                    for path in list(self.memory_in_tool):  # deliver again only if masking removed it
                        if f"Contents of {path} (" not in kept:
                            self.memory_seen.pop(path, None)
                            self.memory_in_tool.discard(path)
                    self.stats["compactions"] += 1
                    emit({"type": "compact", "before": r[0], "after": r[1]})
            if self.context > self.max_context:
                stop = "context"
                break
            steps += 1
            emit({"type": "turn_start", "step": steps, "thinking": self.thinking})
            # More room to think at the start and right after a failure; routine steps stay short
            # (in a validation run 60% of generated tokens were thinking, hitting the budget 13/24 turns).
            self.think_budget = int(lim["thinkBudget"])
            budget = self.think_budget * (2 if steps == 1 or last_failed else 1)
            r = on_mlx(lambda: self._generate(emit, cancel, budget, force_next))
            force_next = False
            emit({"type": "turn_end", "context": self.context,
                  **{k: r[k] for k in ("tokens", "tok_per_s", "prefill_s", "stop")}})
            calls = [parse(raw, SCHEMAS) for raw in r["calls"]]
            finish = next((c for c in calls if c.name == "finish"), None)
            calls = [c for c in calls if c.name != "finish"]
            if r["stop"] == "cut_off" and calls:  # never execute a call that was cut off mid-way
                c = calls[-1]
                c.error = (f"Error: this {c.name or 'tool'} call was cut off at the reply limit ({MAX_STEP_TOKENS} "
                           "tokens) and was not executed. Next: split the work, e.g. write a smaller file or several "
                           "files, or use edit for parts.")
            self.tr.assistant(r["think"], r["text"] if not finish else (r["text"] + "\n" + finish.args["result"]).strip(),
                              [{"name": c.name or "invalid", "arguments": c.args} for c in calls])
            if r["stop"] == "cancelled":
                if calls:  # keep the transcript well formed for the next message
                    self.tr.tool_results(["[interrupted by user]"] * len(calls))
                stop = "cancelled"
                break

            # ---------------------------------------------------- no tool call: the model wants to finish
            if not calls:
                if r["stop"] == "repetition":
                    self.tr.user(prompts.repetition_notice())
                    force_next = True
                    continue
                if r["stop"] in ("length", "cut_off"):
                    stop = "length"
                    break
                if self.steering:
                    self.tr.user(self.steering.popleft())
                    continue
                if not finish and announces_more(r["text"]) and nudges < 2:
                    nudges += 1
                    self.stats["nudges"] += 1
                    self.tr.user(prompts.continue_nudge(prompts.next_step(self.plan)))
                    continue
                if changed_run:  # Stop hooks: the finish gate
                    results = settings_mod.run_hooks(self.box, cfg, "Stop", {"stop_blocks": stop_blocks},
                                                     sorted(set(changed_run)), preview_url=self._preview_url())
                    for h in results:
                        self._emit_hook(h)
                        if h.skipped:
                            continue
                        if h.blocked:
                            self.failing[h.name] = h.output
                        elif h.ok:
                            self.failing.pop(h.name, None)
                    failures = [prompts.check_failed(h.name, truncate(h.output, None, 1800, 800)) for h in results if h.blocked]
                    # Built-in: success criteria, checked by running their verify commands (skipped for small
                    # changes made without a plan).
                    need_criteria = bool(self.plan) or steps > SMALL_TASK_STEPS
                    results_v = self._verify_criteria([c["verify"] for c in prompts.criteria(self.plan) if c["verify"]],
                                                      "Stop") if need_criteria else {}
                    if need_criteria and (gate := prompts.criteria_gate(self.plan, results_v)):
                        failures.append(gate)
                        emit({"type": "hook", "event": "Stop", "name": "success criteria", "ok": False,
                              "blocked": True, "skipped": False, "output": gate, "ms": 0, "image": None})
                    if failures and progress_since_block:  # a block after real work doesn't use up the cap
                        stop_blocks, progress_since_block = 0, False
                    if failures and gate_blocks >= MAX_GATE_BLOCKS:
                        stop = "stop_hook_exhausted"
                        break
                    if failures and stop_blocks < int(lim["maxStopBlocks"]):
                        gate_blocks += 1
                        stop_blocks += 1
                        self.stats["stop_blocks"] += 1
                        self.tr.user(prompts.finish_gate(failures))
                        continue
                    if failures:
                        stop = "stop_hook_exhausted"
                    elif self._visual_due(changed_run, visual_at):  # everything passes: look at it once
                        visual_at = len(changed_run)
                        if self._visual_check(on_mlx, emit):
                            continue
                break

            # ---------------------------------------------------- execute the calls
            outputs, changed, names, introspect = [], [], [], None
            ticked_before = {c["text"] for c in prompts.criteria(self.plan) if c["done"]}
            for i, c in enumerate(calls):
                self.call_seq += 1
                cid = f"c{self.call_seq}"
                self.stats["alias_hits"] += c.aliases
                emit({"type": "tool_call", "id": cid, "name": c.name, "args": c.args, "aliases": c.aliases})
                t0 = time.perf_counter()
                if not c.ok:
                    self.stats["parse_errors"] += 1
                    parse_streak += 1
                    ok, out = False, c.error
                    if parse_streak >= 2:  # two malformed calls in a row: prefill the call skeleton next turn
                        force_next = True
                elif cancel.is_set():
                    ok, out = False, "[interrupted by user]"
                else:
                    parse_streak = 0
                    res = self.tools.run(c.name, c.args, cancel)
                    ok, out = res.ok, res.text
                    if c.ignored:
                        out += f"\n(Note: ignored unknown parameter(s): {', '.join(c.ignored)}.)"
                    changed += res.changed
                    edited_run += res.changed
                    if c.name == "edit" and not ok:
                        self.stats["edit_failures"] += 1
                        if self.guards.edit_failed(rel(c.args.get("file_path", ""))):
                            out += ("\nThis edit has failed repeatedly. Next: read_file the file again, or rewrite it "
                                    "completely with write_file.")
                n_calls += 1
                names.append(c.name or "")
                proc = self.box.procs.get(str(c.args.get("process_id", "")).strip()) if c.name == "read_process" else None
                waiting = proc is not None and ok and proc.exit_code is None
                if not waiting and (what := self.guards.after_call(f"{c.name}:{to_json(c.args)}", c.name or "", None, ok, out)):
                    level = self.guards.escalate()
                    self.stats["stuck"] += 1
                    emit({"type": "note", "text": f"Harness: stuck ({what}); escalation level {level}."})
                    if level == 1:
                        out += prompts.reminder([prompts.stuck_warning(what, self.plan)])
                    else:
                        introspect = introspect or what  # ask the model to assess its trajectory
                ref = None
                if len(out) > 300:  # a full copy, referenced from the compacted history
                    ref = f"{STATE}/out/{cid}.txt"
                    try:
                        self.box.write(ref, out)
                    except Exception as exc:
                        ref = None
                        self._harness_error("could not save the full output copy", exc)
                emit({"type": "tool_result", "id": cid, "ok": ok, "output": out, "ms": (time.perf_counter() - t0) * 1000})
                self.trajectory.append(f"{steps}: {summarize_call_text(c.name, c.args)} -> "
                                       f"{'ok' if ok else 'FAILED'}: {out.strip().splitlines()[0][:110] if out.strip() else ''}")
                if c.name == "update_plan" and ok:
                    out += self._verify_new_ticks(ticked_before)
                outputs.append((out, ref, ok))
                if c.name == "update_plan":
                    plan_step, plan_ignored = steps, 0
            last_failed = any(not k for _, _, k in outputs)
            # Files changed by anything this step (tools, shell commands, installs), from the shadow repo.
            try:
                changed = sorted(set(changed) | set(self.box.dirty()))
            except Exception as exc:
                changed = sorted(set(changed))
                self._harness_error("could not list changed files (checks may miss shell edits)", exc)
            ticked = len(prompts.criteria_status(self.plan)[0])
            if changed or ticked > ticked_at_block:  # real work, not just re-sending the plan
                progress_since_block, ticked_at_block = True, max(ticked, ticked_at_block)
            if changed:
                changed_run += changed
                emit({"type": "files", "tree": self.box.tree()})

            # ---------------------------------------------------- after the batch: one reminder block
            notes = []
            if changed:
                cfg_errors = []
                new_cfg, new_sources = settings_mod.load(self.box, cfg_errors)
                for err in cfg_errors:
                    if err not in "\n".join(self.harness_errors):
                        self._harness_error("settings", err)
                if new_sources != sources:  # e.g. a project template brought its settings: adopt them
                    cfg, sources, lim = new_cfg, new_sources, new_cfg["limits"]
                    emit({"type": "note", "text": f"Harness: loaded settings from {', '.join(sources)}."})
                notes += self._autostart(cfg)
                notes += self._post_hooks(cfg, changed, names, steps)
            notes += self._process_events(cfg)
            while self.box.evicted:
                pid_, cmd_ = self.box.evicted.pop(0)
                msg_ = (f"[process] {pid_} (`{cmd_[:70]}`) was stopped to make room: at most {MAX_LIVE} background "
                        "processes run at once. Stop processes you no longer need with kill_process.")
                notes.append(msg_)
                emit({"type": "note", "text": msg_, "warn": True})
            notes += self._new_data_notes(changed, edited_run)
            if self.failing:
                notes.append(prompts.fix_mode(sorted(self.failing)))
            only_waiting = all(self._waiting(c) for c in calls)
            if not only_waiting and self.guards.after_step(bool(changed), not self.failing):
                notes.append(prompts.explore_warning(6))
            if self.plan and steps - plan_step >= PLAN_EVERY:
                plan_step = steps
                plan_ignored += 1
                if plan_ignored >= 2 and not force_next:
                    # Reminders were ignored twice: the next turn starts with update_plan already open.
                    force_next = "update_plan>\n<parameter=plan>\n"
                    notes.append(prompts.plan_reminder(self.plan, PLAN_EVERY) + "\nUpdate it now: your next reply "
                                 "starts with update_plan.")
                else:
                    notes.append(prompts.plan_reminder(self.plan, PLAN_EVERY))
            step_limit_near = bool(max_steps) and steps >= 0.8 * max_steps
            if not warned and (step_limit_near or time.monotonic() - t_start > 0.8 * max_s):
                warned = True
                minutes_left = max(0, (max_s - (time.monotonic() - t_start)) / 60)
                notes.append(prompts.budget_warning(max_steps - steps, "steps") if step_limit_near
                             else prompts.budget_warning(round(minutes_left), "minutes"))
            # Semantic loop and progress detection (beyond exact repeats).
            fixed = self.progress.checks_fixed(set(prev_failing) - set(self.failing))
            recovered = self.progress.call_outcomes(
                [(f"{c.name}:{to_json(c.args)}", k) for c, (_, _, k) in zip(calls, outputs)])
            # Problem signals: outputs of commands and edits plus check/process notes (not file reads,
            # searches, the plan, or the harness's own reminders, which quote code and earlier errors).
            signals = [o for c, (o, _, _) in zip(calls, outputs)
                       if c.name not in ("read_file", "grep_search", "glob", "update_plan")]
            signals += [n for n in notes if n.startswith(("[check:", "[process]"))]
            issue_msg = idle_msg = None
            if not only_waiting:  # waiting on a running job is neither progress nor idling
                issue_msg, idle_msg = self.progress.record(
                    steps, signals, [summarize_call_text(c.name, c.args) for c in calls], self.plan, fixed, recovered)
            prev_failing, prev_step_failed = dict(self.failing), not all(k for _, _, k in outputs)
            if self.progress.last_progress == steps:
                introspections_without_progress = 0
            for msg in (issue_msg, idle_msg):
                if not msg:
                    continue
                level = self.guards.escalate()
                self.stats["stuck"] += 1
                what = "the same issue keeps coming back" if msg is issue_msg else "no measurable progress"
                emit({"type": "note", "text": f"Harness: {what}; escalation level {level}.", "warn": True})
                if level == 1 and msg != "escalate":
                    notes.append(msg)
                else:  # the progress check gets the concrete issue and the fixes tried, not just a label
                    introspect = introspect or (msg if msg != "escalate" else what)
            notes += self.harness_errors
            self.harness_errors = []
            mem = self._memory_deltas()
            if mem:
                self.memory_in_tool |= {p for p in self.memory_seen if p.startswith("/workspace/")}
            notes += mem + self.deltas
            self.deltas = []
            while self.steering:
                msg = self.steering.popleft()
                self.requests.append(msg)  # durable: restated in the compaction note
                notes.append(f"The user says (address this now; it may change the plan): {msg}")
            last_out, last_ref, last_ok = outputs[-1]
            block = prompts.reminder(notes)
            outputs[-1] = (last_out + block, last_ref, last_ok)
            if block:  # observability: exactly what the model was told after this step
                emit({"type": "reminder", "id": cid, "text": block.strip()})
            self.tr.tool_results([o for o, _, _ in outputs], [k for _, _, k in outputs], [r for _, r, _ in outputs])
            if changed:
                self._checkpoint(f"step {steps}: " + ", ".join(changed)[:100])
            if introspect and stop != "stuck":
                if introspections_without_progress >= MAX_INTROSPECTIONS:
                    stop = "stuck"
                    emit({"type": "note", "text": f"Harness: {MAX_INTROSPECTIONS} progress checks without real "
                          "progress in between; stopping and reporting.", "warn": True})
                else:
                    introspections_without_progress += 1
                    verdict = self._introspect(introspect, steps, t_start, emit, cancel, on_mlx)
                    if verdict == "needs_user":
                        stop = "needs_user"
            if stop in ("stuck", "needs_user"):
                break

        if stop in ("max_steps", "max_time", "stuck") and self.tr.open:
            self._wrap_up(cfg, changed_run, emit, cancel, on_mlx)
        self.tr.close_open_turn()  # limits/cancel left a generation prompt open: close it consistently
        if stop == "cancelled":
            self.deltas.append(prompts.interrupted(
                [f"{p.id} `{p.command[:40]}`" for p in self.box.running_procs()]))
        unchecked = prompts.self_reported(self.plan)
        if unchecked and stop == "done":
            emit({"type": "note", "warn": True, "text": "Self-reported (no verify command, not checked by the harness): "
                  + "; ".join(u[:80] for u in unchecked)})
        self.stats["placeholder_writes"] = self.tools.placeholder_writes
        self.stats["small_rewrites"] = self.tools.small_rewrites
        emit({"type": "done", "stop": stop, "steps": steps, "tool_calls": n_calls, "context": self.context,
              "stats": {k: (len(v) if isinstance(v, list) else v) for k, v in self.stats.items()}})

    def _verify_criteria(self, commands: list[str], event: str) -> dict[str, tuple[bool, str]]:
        """Run criteria verify commands in the sandbox (exit 0 = the criterion holds)."""
        results = {}
        for cmd in dict.fromkeys(commands):
            t0 = time.perf_counter()
            code, out = self.box.sh(f"cd /workspace && export {PLAIN_ENV}; {cmd}", timeout=VERIFY_TIMEOUT)
            results[cmd] = (code == 0, out)
            self.emit({"type": "hook", "event": event, "name": f"criterion: {cmd[:60]}", "ok": code == 0,
                       "blocked": code != 0, "skipped": False, "output": truncate(out, None, 3000, 1000),
                       "ms": (time.perf_counter() - t0) * 1000, "image": None})
        return results

    def _verify_new_ticks(self, ticked_before: set[str]) -> str:
        """The model just ticked criteria: run their verify commands now and say whether they really pass."""
        new = [c for c in prompts.criteria(self.plan) if c["done"] and c["text"] not in ticked_before and c["verify"]]
        if not new:
            return ""
        res = self._verify_criteria([c["verify"] for c in new], "PostToolUse")
        lines = []
        for c in new:
            ok, out = res[c["verify"]]
            lines.append(f"[criterion verified] {c['text'][:100]}" if ok else
                         f"[criterion NOT verified] {c['text'][:100]}\nits verify command failed:\n"
                         + truncate(out.strip(), None, 1200, 600) + "\nUntick it and fix the problem.")
        return prompts.reminder(lines)

    def _visual_due(self, changed_run: list[str], visual_at: int) -> bool:
        return (bool(self.box.preview_target) and self.stats["visual_checks"] < MAX_VISUAL_CHECKS
                and len(changed_run) != visual_at and bool(changed_run))

    def _visual_check(self, on_mlx, emit) -> bool:
        """Show the model a screenshot of the served page (it is a vision model) and ask whether it matches
        the request. Returns True if the check was shown (the loop continues for the model's answer)."""
        shot = "/tmp/bonsai/visual.png"
        code, out = self.box.sh(f"node /opt/bonsai/tools/check-page.mjs {self._preview_url()} {shot} 3000", timeout=90)
        try:
            data = self.box.read(shot)
        except Exception as exc:
            self._harness_error("visual check: no screenshot", f"{exc}; {out[-300:]}")
            return False
        import io
        import tempfile
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((1024, 1024))
        f = tempfile.NamedTemporaryFile(prefix="bonsai-visual-", suffix=".png", delete=False)
        f.close()  # Allow Windows tools to reopen the screenshot path.
        img.save(f.name)
        self.stats["visual_checks"] += 1
        emit({"type": "hook", "event": "Stop", "name": "visual check", "ok": True, "blocked": False, "skipped": False,
              "output": "Screenshot shown to the model for a visual comparison with the request.", "ms": 0,
              "image": shot})
        self.tr.close_open_turn()
        self.tr.user_with_image(prompts.visual_check(self.requests[0] if self.requests else ""), f.name, on_mlx)
        return True

    def _preview_url(self) -> str | None:
        """URL (inside the sandbox) of what the user's preview shows, if anything is being served."""
        return f"http://localhost:{self.box.preview_target}/" if self.box.preview_target else None

    def _waiting(self, c) -> bool:
        """A read_process on a process that is still running: waiting, not working or idling."""
        if c.name != "read_process":
            return False
        p = self.box.procs.get(str(c.args.get("process_id", "")).strip())
        return p is not None and p.exit_code is None

    def _introspect(self, reason: str, steps: int, t_start: float, emit, cancel, on_mlx) -> str:
        """Instead of a hard stop: the model assesses its own trajectory and commits to a new approach
        (or asks the user). Returns "continue" or "needs_user"."""
        self.stats["introspections"] += 1
        self.tr.close_open_turn()
        self.tr.user(prompts.introspection(reason, self.trajectory, self.plan, (time.monotonic() - t_start) / 60, steps))
        emit({"type": "note", "text": f"Harness: progress check ({reason}); asking the model to assess its trajectory."})
        emit({"type": "turn_start", "step": steps, "thinking": self.thinking})
        r = on_mlx(lambda: self._generate(emit, cancel, self.think_budget, False, close_text=ANSWER_CLOSE))
        emit({"type": "turn_end", "context": self.context,
              **{k: r[k] for k in ("tokens", "tok_per_s", "prefill_s", "stop")}})
        calls = [parse(raw, SCHEMAS) for raw in r["calls"]]
        self.tr.assistant(r["think"], r["text"], [{"name": c.name or "invalid", "arguments": c.args} for c in calls])
        if calls:
            self.tr.tool_results(["[not executed: answer the progress check first]"] * len(calls))
            self.tr.close_open_turn()
        self.progress.after_check(steps)  # idle steps count afresh from here, whatever the answer
        if r["stop"] == "cancelled":
            return "continue"  # the loop sees the cancel; don't parse a half answer
        v = prompts.parse_introspection(r["text"])
        emit({"type": "introspection", **{k.lower().replace(" ", "_"): v.get(k, "") for k in prompts.INTROSPECTION_FIELDS}})
        if prompts.needs_user(v.get("NEED USER", "")):
            return "needs_user"
        if v.get("ASSESSMENT", "").lower().startswith("progressing") and v.get("EVIDENCE"):
            self.guards.level = 1  # benefit of the doubt, but the next trigger goes straight to a check
        approach = v.get("NEW APPROACH") or "Try a different approach than before."
        self.tr.user(prompts.harness_message(f"Continue with your new approach:\n{approach}"))
        self.guards.history.clear()
        return "continue"

    def _wrap_up(self, cfg, changed_run, emit, cancel, on_mlx):
        """Out of steps/time: run the finish checks for the record and let the model report to the user in
        one turn without tools (instead of ending silently)."""
        checks = []
        if changed_run:
            for h in settings_mod.run_hooks(self.box, cfg, "Stop", {"wrap_up": True}, sorted(set(changed_run)),
                                            preview_url=self._preview_url()):
                self._emit_hook(h)
                checks.append(f"[check: {h.name}] " + ("ok" if h.ok else "skipped (an earlier check failed)"
                                                        if h.skipped else "failed:\n" + h.output[:800]))
        # The pending generation prompt is replaced by the wrap-up request.
        self.tr.close_open_turn()
        self.tr.user(prompts.step_limit(checks))
        emit({"type": "turn_start", "step": 0, "thinking": self.thinking})
        r = on_mlx(lambda: self._generate(emit, cancel, 0, False, no_think=True))
        emit({"type": "turn_end", "context": self.context,
              **{k: r[k] for k in ("tokens", "tok_per_s", "prefill_s", "stop")}})
        calls = [parse(raw, SCHEMAS) for raw in r["calls"]]
        self.tr.assistant(r["think"], r["text"], [{"name": c.name or "invalid", "arguments": c.args} for c in calls])
        if calls:  # not executed: keep the transcript well formed
            self.tr.tool_results(["[not executed: the step limit is reached]"] * len(calls))

    def _new_data_notes(self, changed: list[str], written: list[str]) -> list[str]:
        """A data file appeared through a download or command (not written by the model): show a sample right
        away, so a parser is written for what is really in the file (the model tended to skip looking)."""
        fresh = [p for p in changed if p not in self.peeked and p not in written]
        if not fresh:
            return []
        try:
            big = {p: n for p, n in self.box.sizes(fresh).items() if n >= DATA_PEEK_BYTES}
        except Exception as exc:
            self._harness_error("could not check sizes of new files", exc)
            return []
        notes = []
        for path in list(big)[:3]:
            self.peeked.add(path)
            notes.append(f"[data] New file {path}:\n{self.box.peek(path)}\nStudy its real structure (look at more "
                         "records if needed) before writing code that parses it.")
        return notes

    def _recent_files(self, edited: list[str]) -> list[tuple[str, str]]:
        """Current contents of the files the model changed most recently, for the compaction note: compaction
        removes their bodies from the history, and without them an edit needs an extra read_file (or tempts a
        full rewrite from memory)."""
        out, total = [], 0
        for path in list(dict.fromkeys(reversed(edited))):
            if len(out) >= RECENT_FILES_MAX:
                break
            try:
                text = self.box.read(path).decode(errors="replace")
            except Exception as exc:
                if "No such file" not in str(exc):
                    self._harness_error(f"could not read {path} for the compaction note", exc)
                continue
            if len(text) > RECENT_FILE_CHARS or total + len(text) > RECENT_FILES_CHARS:
                continue
            out.append((path, text))
            total += len(text)
        return out

    def _checkpoint(self, message: str):
        try:
            if cid := self.box.checkpoint(message[:120]):
                self.emit({"type": "checkpoint", "id": cid, "message": message[:120]})
            else:
                self._harness_error("checkpoint failed", "git commit in the shadow repo returned no id")
        except Exception as exc:
            self._harness_error("checkpoint failed", exc)
