"""Prompt layers (see docs/coding-agent-architecture.md §1). Nothing project-specific lives here:
environment and project knowledge arrive through AGENTS.md memory files, skills and hooks.

  system turn   tool schemas (rendered by the chat template) + SYSTEM: byte-stable, sent once
  first user    <environment>, memory files, <available_skills>, then the request
  reminders     <system-reminder> blocks at the end of the last tool response (the chat template
                treats a plain user message as a new query, so mid-task notes never become one)
"""

from __future__ import annotations

import re

SYSTEM = """You are Bonsai Code, a coding agent. You work by calling tools inside a sandboxed workspace; the user \
sees your plan, the files you change, the terminal and, when the project serves something on a port, a live \
preview. Be autonomous: carry each request through to a verified, working result, then report.

# Context you receive
- <environment> describes the workspace. "Contents of <path>" blocks are instruction files (AGENTS.md): follow \
them. When they conflict, the more specific wins: the user's message > project AGENTS.md > global AGENTS.md > \
this prompt. The tool-call format below cannot be overridden.
- <available_skills> lists guides by name, description and file. When a request matches a description, read that \
file with read_file BEFORE writing code for that part. Read each skill once; it stays in context.
- <system-reminder> blocks come from the harness, not the user: check results, process events, plan reminders, \
warnings. Act on them; don't reply to them.

# How to work
1. Understand: read the files you will change. Never guess file contents, APIs or names. Before writing code that \
reads existing data (a download, a dataset, a log, an export), look at a sample first: its format, structure, \
fields and encoding, then write the parser for what is really there.
2. Plan: for anything bigger than a small change, call update_plan with 3-8 short steps, plus "Success criteria:":
   concrete checks that prove the request is done, each with a command that exits 0 only when it holds:
   "- [ ] training loss falls below 4.0 — verify: `uv run python check_loss.py`" or
   "- [ ] dashboard renders without errors — verify: `check-page http://localhost:8000/`". The harness runs these
   commands itself when you tick a criterion and before you finish; a failing command blocks finishing. Keep
   exactly one step "(in progress)" and mark steps done only after they ran successfully. Skip the plan for
   trivial requests.
3. Act: make the change in small, complete steps. Do what was asked, no more. Prefer well-tested libraries over \
re-implementing algorithms yourself unless the user asks for that.
4. Verify as you go: run what you built on a small sample right after building it, and fix what fails before
   the next step. Tick each success criterion as soon as its verify command passes, adding the evidence
   (e.g. "- [x] loss falls below 4.0 (step 900: 3.71) — verify: `...`"). Never tick what you have not run.
5. Long jobs (training, big downloads) run in the background: wait with read_process wait_seconds instead of
   polling every few seconds, and check progress against the criteria.

# Tools
- Usually one tool call per reply, then wait for its result. Several edit calls in one reply are fine for \
changes to different places.
- Before a call, write at most one short sentence. Think briefly: decide the next action and take it. Never \
draft file contents in your thinking.
- Use the dedicated tools: read_file to read, write_file and edit to change files, glob and grep_search to find \
things. Use run_shell_command for installs, builds and one-off commands; each call is a fresh shell (cd and \
variables don't persist; chain with &&). A command still running after 30 s continues in the background as a \
process id: check it with read_process, stop it with kill_process. Don't add & or nohup.
- Don't re-read a file you just read or wrote; its content is in context.

# Editing
- New file, or more than half of an existing file changes: write_file with the COMPLETE content.
- Everything else (fixes, renames, adding or changing a function): edit, once per place that changes. A rewrite of \
a 100-line file costs about ten times the tokens of an edit. Copy old_string exactly from the file, without line \
numbers, with 2-3 lines of context so it matches once.
- If edit fails, read_file again and retry with the exact current text, or rewrite the file with write_file.
- Never write "...", "rest unchanged", TODOs or stubs. Close every tag, string and bracket. Use only APIs, \
packages and names that exist.
- Edit files in place; never create copies like main_v2.py.

# Checks
After your edits the harness runs the project's checks and appends the result as a <system-reminder>:
  [check: NAME] failed:
  <errors>
Fix the first real error, then continue. The same checks run when you finish; failures come back to you instead \
of ending the turn.

# Finishing
A reply without a tool call ends your turn. End only when every success criterion is verified and ticked, and the \
checks pass (the harness refuses to finish otherwise). There is no rush: work as long as needed. Then reply in 2-5 \
plain sentences: what you built, the evidence for each criterion, and how to use it. If something truly cannot be \
done, say exactly what and why.

# When stuck
If the same error survives two fixes: stop, list the 3 most likely causes, and try the most likely with a \
different approach (re-read the file, simplify the code, or rewrite the file completely). Don't repeat a call \
that already failed. Don't read the sources of installed libraries. If the request is ambiguous and the files \
can't answer it, ask one short question instead of guessing.

# Example
user: Make the report title say "Orders".
assistant: I'll update the title in report/config.py.
<tool_call>
<function=edit>
<parameter=file_path>
report/config.py
</parameter>
<parameter=old_string>
REPORT = {
    "title": "Dashboard",
    "rows": 50,
</parameter>
<parameter=new_string>
REPORT = {
    "title": "Orders",
    "rows": 50,
</parameter>
</function>
</tool_call>

# Style
Plain, direct and short. No greetings or filler. Refer to files by path. Never paste large file contents into \
replies."""

MEMORY_MAX_CHARS = 8000


def environment(files: list[str], processes: list[str], notes: list[str], resources: str = "") -> str:
    if files:
        shown = "\n".join(f"  {f}" for f in files[:80])
        more = f"\n  … {len(files) - 80} more" if len(files) > 80 else ""
        ws = f"Files ({len(files)}, newest first):\n{shown}{more}"
    else:
        ws = "Files: none (empty workspace)"
    procs = ("Background processes: " + "; ".join(processes)) if processes else "Background processes: none"
    extra = "".join(f"\n{n}" for n in notes)
    return (f"<environment>\nWorkspace: /workspace (tool paths are relative to it)\nShell: bash (login shell, fresh "
            f"per command, pipefail on, no TTY, stdin empty)\n"
            + (f"Resources: {resources}\n" if resources else "") + f"{ws}\n{procs}{extra}\n</environment>")


def memory_block(path: str, text: str, scope: str) -> str:
    body = text.strip()
    if len(body) > MEMORY_MAX_CHARS:
        body = body[:MEMORY_MAX_CHARS] + "\n[… truncated]"
    label = {"global": "global instructions: the environment",
             "project": "project instructions; override global"}[scope]
    return f"Contents of {path} ({label}):\n{body}"


def skills_index(skills: list[dict]) -> str:
    if not skills:
        return ""
    rows = "\n".join(f"- {s['name']}: {s['description']} (file: {s['path']})" for s in skills)
    return f"<available_skills>\n{rows}\n</available_skills>"


def reminder(lines: list[str]) -> str:
    body = "\n".join(l.strip() for l in lines if l and l.strip())
    return f"\n<system-reminder>\n{body}\n</system-reminder>" if body else ""


def check_failed(name: str, output: str) -> str:
    out = output.strip() or "(no output)"
    return f"[check: {name}] failed:\n{out}"


def harness_message(body: str) -> str:
    """A user-role message from the harness (only when there is no tool response to attach to).
    Tagged and explicit, like Codex's <turn_aborted>, so it isn't mistaken for a new request."""
    return f"<system-reminder>\nHarness, not the user (your request is unchanged):\n{body.strip()}\n</system-reminder>"


_CRITERIA_HEADER = re.compile(r"^[#*\s]*(success|acceptance) criteria\s*:?[*\s]*$", re.I)


_VERIFY = re.compile(r"verify:\s*`([^`]+)`", re.I)


def criteria(plan: str) -> list[dict]:
    """Success criteria from the plan: [{text, done, verify}]. `verify` is the command written as
    verify: `command` (None if the criterion has no command the harness can run)."""
    out, inside = [], False
    for line in plan.splitlines():
        s = line.strip()
        if not s:
            continue
        if _CRITERIA_HEADER.match(s):
            inside = True
            continue
        if inside and s.startswith(("- [", "* [")):
            text = s.split("]", 1)[-1].strip()
            m = _VERIFY.search(text)
            out.append({"text": text, "done": s[3:4].lower() == "x", "verify": m.group(1).strip() if m else None})
        elif inside and s.startswith("#"):
            inside = False
    return out


def criteria_status(plan: str) -> tuple[list[str], list[str]]:
    """(ticked, open) success criteria texts."""
    cs = criteria(plan)
    return [c["text"] for c in cs if c["done"]], [c["text"] for c in cs if not c["done"]]


def plan_steps_done(plan: str) -> int:
    """Ticked plan steps (outside the success criteria)."""
    n, inside = 0, False
    for line in plan.splitlines():
        s = line.strip()
        if _CRITERIA_HEADER.match(s):
            inside = True
        elif s.startswith(("- [x]", "- [X]", "* [x]")) and not inside:
            n += 1
    return n


def criteria_gate(plan: str, results: dict[str, tuple[bool, str]]) -> str | None:
    """The built-in finish check. `results` maps a criterion's verify command to (passed, output), from the
    harness running it. A criterion with a verify command counts only if the command passes (the tick
    alone is not enough); one without a command must at least be ticked."""
    if not plan:
        return ("[check: success criteria] failed:\nThere is no plan. Call update_plan with your steps and a "
                "'Success criteria:' checklist (each criterion with `verify: \\`command\\``), then finish.")
    cs = criteria(plan)
    if not cs:
        return ("[check: success criteria] failed:\nYour plan has no 'Success criteria:' checklist. Add concrete "
                "criteria, each with a command that proves it, e.g. `- [ ] page renders without errors — verify: "
                "\\`check-page http://localhost:8000/\\``, then finish.")
    problems = []
    if not any(c["verify"] for c in cs):
        problems.append("No criterion has a verify command, so nothing can be checked. Add `verify: \\`command\\`` "
                        "(exit code 0 = passed) to every criterion that a command can prove.")
    for c in cs:
        if c["verify"]:
            ok, out = results.get(c["verify"], (False, "not run"))
            if not ok:
                problems.append(f"- {c['text']}\n  verify command failed:\n" + "\n".join(
                    "    " + l for l in out.strip().splitlines()[-12:]))
        elif not c["done"]:
            problems.append(f"- {c['text']} (not ticked, and no verify command)")
    if problems:
        return ("[check: success criteria] failed:\n" + "\n".join(problems)
                + "\nMake each verify command pass (fix the code, not the command), then finish.")
    return None


def self_reported(plan: str) -> list[str]:
    """Ticked criteria the harness could not check (no verify command)."""
    return [c["text"] for c in criteria(plan) if c["done"] and not c["verify"]]


def visual_check(request: str) -> str:
    return ("<system-reminder>\nHarness, not the user: all checks pass. Above is a screenshot of the page the "
            "user sees in the preview right now. Compare it with the request:\n" + request[:800] + "\n"
            "Answer first with one line, VISUAL: ok, or VISUAL: problems: <what is missing or wrong>. If there are "
            "problems, fix them now with tools; if it is ok, finish with your summary.\n</system-reminder>")


INTROSPECTION_FIELDS = ("ASSESSMENT", "EVIDENCE", "ROOT CAUSE", "NEW APPROACH", "NEED USER")


def introspection(reason: str, trajectory: list[str], plan: str, elapsed_min: float, steps: int) -> str:
    done, open_ = criteria_status(plan)
    crit = (f"{len(done)} of {len(done) + len(open_)} success criteria verified" if done or open_
            else "no success criteria defined yet")
    return harness_message(
        f"Progress check: {reason}. You have worked {steps} steps ({elapsed_min:.0f} min); {crit}.\n"
        "Recent trajectory (step: call -> result):\n" + "\n".join(trajectory[-20:]) + "\n\n"
        "Assess honestly whether you are moving toward the goal. Answer WITHOUT tool calls, in exactly this form:\n"
        "ASSESSMENT: progressing | looping | blocked\n"
        "EVIDENCE: what has actually been achieved (verified facts only)\n"
        "ROOT CAUSE: why the recent steps did not move forward\n"
        "NEW APPROACH: the next 1-3 concrete actions, different from what you already tried\n"
        "NEED USER: no | <one short question, only if you cannot continue without the user>")


def needs_user(answer: str) -> bool:
    """A real question for the user ("No, I can continue." or the template echo are not)."""
    a = answer.strip().lower()
    return bool(a) and not a.startswith(("no", "none", "n/a", "-", "<")) and "?" in a


def parse_introspection(text: str) -> dict[str, str]:
    out, key = {}, None
    for line in text.splitlines():
        head = line.split(":", 1)[0].strip().strip("*").upper()
        if head in INTROSPECTION_FIELDS and ":" in line:
            key = head
            out[key] = line.split(":", 1)[1].strip()
        elif key and line.strip():
            out[key] += "\n" + line.strip()
    return out


def finish_gate(failures: list[str]) -> str:
    return harness_message("You tried to finish, but these checks fail:\n" + "\n".join(failures)
                           + "\nFix the first real error, then finish.")


def continue_nudge(next_step: str) -> str:
    return harness_message("Your last reply said more work follows but made no tool call."
                           + (f" Next plan step: {next_step}." if next_step else "") + " Make that tool call now.")


def repetition_notice() -> str:
    return harness_message("You started repeating the same text. Stop deliberating and make the next concrete "
                           "tool call now.")


def fix_mode(names: list[str]) -> str:
    return f"Checks failing: {', '.join(names)}. Fix these before new work; don't edit unrelated files."


def interrupted(running: list[str]) -> str:
    s = "The previous turn was interrupted by the user on purpose. Tool calls may have partially executed."
    return s + (f" Still running: {', '.join(running)}." if running else "")


def next_step(plan: str) -> str:
    """The plan step marked in progress, else the first unchecked one."""
    lines = [l.strip() for l in plan.splitlines() if l.strip().startswith(("- [", "* ["))]
    cur = next((l for l in lines if "in progress" in l.lower()), None) or next((l for l in lines if "[ ]" in l), "")
    return cur.split("]", 1)[-1].replace("(in progress)", "").strip() if cur else ""


def plan_reminder(plan: str, steps: int) -> str:
    return f"Plan (not updated for {steps} steps; mark finished steps done, keep one in progress):\n{plan}"


def stuck_warning(what: str, plan: str) -> str:
    s = (f"You are repeating yourself ({what}). Stop: list the 3 most likely causes, pick one, and try a "
         "different approach (re-read the file, simplify, or rewrite it).")
    return s + (f"\nPlan:\n{plan}" if plan else "")


def explore_warning(n: int) -> str:
    return (f"You have spent {n} steps without changing any file. Stop investigating; make the next change now "
            "with what you know and let the checks tell you whether it works.")


def budget_warning(left: int, unit: str) -> str:
    return f"{left} {unit} left in this turn: finish the current plan step, then wrap up."


def masked_output(tool: str, arg: str, first_line: str, ref: str) -> str:
    return f"[output cleared to save context: {tool} {arg}: {first_line[:120]}; full output: {ref}]"


def step_limit(checks: list[str]) -> str:
    status = ("\n".join(checks) if checks else "No checks are configured for this project.")
    return harness_message("This run is being stopped (time limit reached, or the harness detected a loop); no more "
                           "tool calls will run. Check "
                           f"status:\n{status}\nReply now, without tool calls, with 2-4 sentences for the user: "
                           "what works, what is left to do.")


def current_files(files: list[tuple[str, str]]) -> str:
    """Current on-disk contents (read_file format), so edits after compaction need no extra read."""
    parts = ["Current contents of the files you changed most recently (use these for edit; no read_file needed):"]
    for path, text in files:
        lines = text.split("\n")
        if text.endswith("\n"):
            lines = lines[:-1]
        body = "\n".join(f"{i + 1:6}\t{l}" for i, l in enumerate(lines))
        parts.append(f"=== {path} ({len(lines)} lines) ===\n{body}")
    return "\n".join(parts)


def after_compaction(task: str, latest: str, plan: str, changed: list[str], failing: list[str]) -> str:
    parts = ["Earlier steps were compacted: they are summarised as one line per tool call above, and their "
             "outputs were removed. Files on disk are current; read_file a file before editing it. Continue from "
             "the plan; don't redo finished steps.", f"Task: {task[:1500]}"]
    if latest and latest != task:
        parts.append(f"Later user messages (still apply): {latest[:800]}")
    if plan:
        parts.append(f"Plan:\n{plan}")
    if changed:
        parts.append("Files changed in this task: " + ", ".join(changed[:40]))
    if failing:
        parts.append("Checks still failing: " + ", ".join(failing))
    return "\n".join(parts)
