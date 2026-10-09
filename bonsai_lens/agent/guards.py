"""Deterministic loop guards (no extra model calls). Thresholds from OpenHands' stuck detector,
opencode's doom loop and Gemini CLI / Qwen Code's loop detection; escalation as in Gemini CLI
(recover on the first detection, stop on a repeat).

Ladder:  1 nudge (reminder)  ->  2 force (next turn is prefilled with a tool call)  ->  3 stop ("stuck")
The ladder resets after a step that changes files and leaves every check passing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SAME_CALL_SAME_RESULT = 4
SAME_CALL_SAME_ERROR = 3
ALTERNATION_STEPS = 6
EXPLORE_STEPS = 6
CHANT_CHUNK, CHANT_REPEATS, CHANT_WINDOW = 50, 10, 5000
SAME_EDIT_FAILURES = 2


def _sig(text: str) -> str:
    """A result's identity: its body (after any `Exit code … / Output:` header) with numbers normalised,
    so two different failing commands don't look alike just because both say "Exit code: 1"."""
    t = text.strip()
    if "\nOutput:\n" in t:
        t = t.split("\nOutput:\n", 1)[1]
    return re.sub(r"\d+(\.\d+)?", "#", t)[:300]


def chanting(text: str) -> bool:
    """The last 50-char chunk occurs 10+ times in the recent window (repetitive degeneration)."""
    w = text[-CHANT_WINDOW:]
    if len(w) < CHANT_CHUNK * CHANT_REPEATS:
        return False
    return w.count(w[-CHANT_CHUNK:]) >= CHANT_REPEATS


_MORE = re.compile(r"(next,? i('ll| will)|now i('ll| will)|i('ll| will) now|let me (now )?\w+)[^.\n]*[.:]?\s*$", re.I)


def announces_more(text: str) -> bool:
    """A reply that ends by announcing work it didn't do (Gemini CLI's next-speaker check, as a regex)."""
    tail = text.strip()[-200:]
    return bool(tail) and (tail.endswith(":") or bool(_MORE.search(tail)))


@dataclass
class Guards:
    history: list[tuple[str, bool, str]] = field(default_factory=list)  # (call key, ok, result signature)
    level: int = 0  # escalation ladder position
    explore: int = 0
    edit_failures: dict[str, int] = field(default_factory=dict)

    def after_call(self, key: str, name: str, path: str | None, ok: bool, result: str) -> str | None:
        """Record one tool call; returns what is repeating, if anything."""
        self.history = (self.history + [(key, ok, _sig(result))])[-12:]
        h = self.history
        last = h[-1]
        if name == "update_plan":
            return None
        if sum(1 for e in h if e == last) >= (SAME_CALL_SAME_ERROR if not ok else SAME_CALL_SAME_RESULT):
            self.history = []
            return f"the same {name} call returned the same {'error' if not ok else 'result'} several times"
        if not ok and sum(1 for k, o, g in h if not o and g == last[2] and k.split(":", 1)[0] == name) \
                >= SAME_CALL_SAME_ERROR:
            self.history = []
            return f"{name} keeps failing with the same error"
        if len(h) >= ALTERNATION_STEPS:
            keys = [k for k, _, _ in h[-ALTERNATION_STEPS:]]
            if len(set(keys)) == 2 and keys[0::2] == [keys[0]] * len(keys[0::2]) and keys[1::2] == [keys[1]] * len(keys[1::2]):
                self.history = []
                return "alternating between the same two calls"
        return None

    def edit_failed(self, path: str) -> bool:
        """True when edits to `path` failed SAME_EDIT_FAILURES times in a row."""
        self.edit_failures[path] = self.edit_failures.get(path, 0) + 1
        return self.edit_failures[path] >= SAME_EDIT_FAILURES

    def after_step(self, changed_files: bool, checks_ok: bool) -> bool:
        """Returns True when the model has explored EXPLORE_STEPS steps without changing a file."""
        if changed_files:
            self.edit_failures.clear()
            self.explore = 0
            if checks_ok:
                self.level = 0
            return False
        self.explore += 1
        if self.explore >= EXPLORE_STEPS:
            self.explore = 0
            return True
        return False

    def new_run(self):
        """Per user message: the escalation level and streaks start over."""
        self.level, self.explore = 0, 0
        self.history.clear()
        self.edit_failures.clear()

    def escalate(self) -> int:
        self.level += 1
        return self.level


# ---------------------------------------------------------------- semantic loops and progress
ISSUE_REPEAT = 4  # the same problem coming back this many times despite fixes = looping on one issue
NO_PROGRESS_WARN, NO_PROGRESS_ESCALATE = 30, 60  # steps without measurable progress
# Case-sensitive on purpose: exception classes are CamelCase (KeyError, ModuleNotFoundError), and "not found"
# / "failed" only count in error contexts (a log line "val_error: 0.31" or "0 failed" is not a problem).
_ISSUE_LINE = re.compile(
    r"(\b[A-Z][A-Za-z]*(Error|Exception)\b:.*|\berror TS\d+.*|^\s*(error|Error|ERROR|fatal|FATAL)\b[:!].*|"
    r"\[check: [^\]]+\] failed|^FAILED .*|\b[1-9]\d* failed\b.*|command not found.*|No such file or directory.*|"
    r"killed: .*)")


def issue_signatures(text: str) -> list[str]:
    """Fingerprints of the problems in an output: error lines with numbers, paths and quoted values
    normalised, so the same problem matches across attempts even if line numbers or values move."""
    sigs = []
    for line in text.splitlines():
        m = _ISSUE_LINE.search(line)
        if not m:
            continue
        s = m.group(0).strip()
        if s.lower().startswith("exit code"):
            continue  # too generic on its own; the error lines carry the identity
        s = re.sub(r"(/[\w.\-]+)+", "<path>", s)
        s = re.sub(r"(['\"]).*?\1", "<val>", s)
        s = re.sub(r"\d+(\.\d+)?", "#", s)
        sigs.append(s[:160])
    return list(dict.fromkeys(sigs))[:5]


@dataclass
class Progress:
    """Tracks recurring issues and measurable progress across a run."""
    issues: dict[str, list[int]] = field(default_factory=dict)  # signature -> steps it appeared in
    attempts: dict[str, list[str]] = field(default_factory=dict)  # signature -> calls tried while it was open
    last_progress: int = 0
    criteria_done: int = 0
    steps_done: int = 0
    warned_issue: set[str] = field(default_factory=set)
    warned_idle: int = 0

    idle_base: int = 0  # idle steps are counted from max(last_progress, idle_base)
    failed_keys: set[str] = field(default_factory=set)  # call keys that failed (for "recovered")
    fixed_checks_seen: set[str] = field(default_factory=set)  # checks that went red -> green at least once

    def new_run(self, step: int = 0):
        self.issues.clear()
        self.attempts = {}
        self.warned_issue.clear()
        self.failed_keys.clear()
        self.fixed_checks_seen.clear()
        self.last_progress, self.idle_base, self.warned_idle = step, step, 0

    def after_check(self, step: int):
        """A progress check (introspection) happened: count idle steps afresh from here, so the detector
        keeps working whatever the model answered."""
        self.idle_base, self.warned_idle = step, 0

    def call_outcomes(self, keyed: list[tuple[str, bool]]) -> bool:
        """Feed (call key, ok) pairs; True if a call that failed before now succeeds (a real recovery)."""
        recovered = any(ok and key in self.failed_keys for key, ok in keyed)
        for key, ok in keyed:
            if ok:
                self.failed_keys.discard(key)
            else:
                self.failed_keys.add(key)
        return recovered

    def checks_fixed(self, names: set[str]) -> bool:
        """True if a check went from failing to passing for the first time (flip-flopping doesn't count)."""
        new = names - self.fixed_checks_seen
        self.fixed_checks_seen |= names
        return bool(new)

    def record(self, step: int, outputs: list[str], calls: list[str], plan: str, fixed_checks: bool,
               recovered: bool) -> tuple[str | None, str | None]:
        """Feed one step. Returns (recurring-issue warning, no-progress warning)."""
        from .prompts import criteria_status, plan_steps_done  # local: prompts imports nothing from here
        done, _ = criteria_status(plan)
        steps_done = plan_steps_done(plan)
        if len(done) > self.criteria_done or steps_done > self.steps_done or fixed_checks or recovered:
            self.last_progress, self.warned_idle = step, 0
        self.criteria_done, self.steps_done = max(self.criteria_done, len(done)), max(self.steps_done, steps_done)
        issue_msg = None
        seen_now = {s for o in outputs for s in issue_signatures(o)}
        for sig in seen_now:
            self.issues.setdefault(sig, []).append(step)
            self.attempts.setdefault(sig, [])
            self.attempts[sig] += [f"step {step}: {c}" for c in calls]  # what was tried while it was open
            if len(self.issues[sig]) >= ISSUE_REPEAT and sig not in self.warned_issue:
                self.warned_issue.add(sig)
                tried = self.attempts[sig][-8:]
                issue_msg = (f"The same problem has come back {len(self.issues[sig])} times (steps "
                             f"{', '.join(map(str, self.issues[sig][-6:]))}): {sig}\n"
                             + ("Fixes tried since it first appeared:\n" + "\n".join(f"- {t}" for t in tried) + "\n"
                                if tried else "")
                             + "These attempts are not working. Stop and find the root cause: re-read the full error "
                               "and the code it points to, check the API you rely on (e.g. its documentation or "
                               "signature), list 3 possible causes, and try a different approach than before.")
        idle = step - max(self.last_progress, self.idle_base)
        idle_msg = None
        if idle >= NO_PROGRESS_WARN and self.warned_idle == 0:
            self.warned_idle = 1
            idle_msg = (f"No measurable progress for {idle} steps: no success criterion verified, no plan step "
                        "completed, no failing check fixed. Write down what is blocking you, decide whether the "
                        "current approach can work, and change it if not.")
        elif idle >= NO_PROGRESS_ESCALATE and self.warned_idle == 1:
            self.warned_idle = 2
            idle_msg = "escalate"
        return issue_msg, idle_msg


def summarize_call_text(name: str | None, args: dict) -> str:
    a = args or {}
    arg = a.get("file_path") or a.get("command") or a.get("pattern") or a.get("process_id") or ""
    return f"{name} {str(arg).splitlines()[0][:80] if arg else ''}".strip()
