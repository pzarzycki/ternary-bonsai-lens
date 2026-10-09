# Bonsai Code: harness architecture (v3)

This document is the design for a complete refactor of the coding agent. It merges four research specs (tool surface, loop control, prompt architecture, and a deep read of the Codex source) plus earlier reports. Supporting material is in [coding-agent-research.md](coding-agent-research.md) and [coding-agent-improvement-plan.md](coding-agent-improvement-plan.md).

**References.** The primary reference is the **OpenAI Codex CLI** (`codex-rs`, read at commit `822e58cc`). Where Codex depends on a strong model, the **Claude Code** pattern is used instead. Qwen Code (Qwen's own CLI) supplies names, because our model's tool-call format comes from Qwen3-coder.

**Rule: the harness is generic.** Nothing about React, Vite, npm, TypeScript or 3D lives in `bonsai_lens/agent/`. Project and environment knowledge comes from three places, which are files in the sandbox image or in the project:
- **Memory files** (`AGENTS.md`): Codex and the agents.md standard; Claude Code's `CLAUDE.md`.
- **Skills** (`SKILL.md`): Codex and Claude Code, following agentskills.io.
- **Settings and hooks** (`.bonsai/settings.json`): Claude Code and Codex hooks.

**Engine constraints.**
- The model keeps a recurrent state plus a KV cache. Appending is cheap; any rewrite means a re-prefill, which costs about 1 s per 1k tokens.
- Generation runs at 30–40 tok/s with a 56k context cap.
- The model is weak (2-bit 27B), so tool calls must be simple, strictly checked by the harness, and have plain string parameters.

## 1. Layers

| Layer | Contents | Where | Sent |
|---|---|---|---|
| 1 | Tool schemas | system turn (rendered by the chat template) | once (byte-stable) |
| 2 | Generic system prompt | system turn, after the tools | once (byte-stable) |
| 3 | `<environment>` | first user message | once; later changes go out as deltas (Codex "world state") |
| 4 | Memory files | first user message, from broad to specific | once; a newly appearing or changed file goes out as a delta |
| 5 | Skills index (name, description, path) | first user message | once |
| 6 | Skill bodies | `read_file` of a `SKILL.md` | on demand; never masked |
| 7 | Reminders `<system-reminder>…</system-reminder>` | **inside the last tool response** | when an event fires |
| 8 | Finish-gate failures | standalone user message | when a Stop hook blocks |

Placement follows the chat template. A plain user message counts as a new query, so earlier reasoning is dropped. Reminders therefore go inside `<tool_response>`, as Claude Code does with PostToolUse `additionalContext`. The finish gate is the one place where a new query is intended.

**Precedence:** the user's message, then the project `AGENTS.md`, then the global `AGENTS.md`, then the system prompt. Tool-format rules cannot be overridden.

## 2. Tools (9): Qwen Code names, plain strings, one optional integer

| Tool | Params (schema order) | Notes |
|---|---|---|
| `read_file` | file_path, offset? | 400-line window, `cat -n` numbering, header "lines a-b of N", paging hint |
| `write_file` | file_path, content | whole file; the harness forces `file_path` to come first |
| `edit` | file_path, old_string, new_string | exact match, then tiered fallbacks (Codex `seek_sequence` + Gemini): trailing whitespace, trim, indentation-insensitive, Unicode punctuation, unescape. Multi-edit = several calls in one reply |
| `run_shell_command` | command | fresh `bash -lc` in /workspace, stdin is /dev/null. If it is still running after the yield time (30 s), it becomes background process `pN` (Codex unified exec, without stdin) |
| `read_process` | process_id | new output since the last read, plus status (running / exited N / listening on port P); waits up to 30 s |
| `kill_process` | process_id | kills the process group |
| `glob` | pattern | newest first; skips node_modules, .git and dist |
| `grep_search` | pattern, path | ripgrep; matches grouped by file, with line numbers; capped |
| `update_plan` | plan | markdown checklist; replies "Plan updated" (Codex) |

**Aliases, accepted silently and logged.**
- Parameters: `path`/`file`/`filename` → `file_path`, `old_str` → `old_string`, `cmd` → `command`.
- Tools: `bash`/`shell` → `run_shell_command`, `edit_file`/`str_replace` → `edit`, `list_files` → `glob`.
- A hallucinated `finish`/`attempt_completion` call is treated as finishing, and its message becomes the reply.

**Excluded:** apply_patch (Codex itself withholds it from unknown models), MultiEdit, think, ask_user, web, Task, and a separate skill tool. Fewer tools works better: GitHub cut from 40 to 13 tools and gained 2–5 points.

**Observation format** (Codex header style):
- Shell result: `Exit code: N · Wall time: 1.2s` then `Output:` and the output. An empty result reads `(no output)`.
- Truncation: about 8k chars, head plus tail, with `[… N lines omitted; full output: /tmp/bonsai/out/<id>.txt]`.
- Errors: `Error: <what>. <evidence>. Next: <one action>.`

## 3. Processes (generic, no dev-server tool)

- **Registry.** Each process gets an id `pN`, its own `setsid` process group, a log at `/tmp/bonsai/proc/pN.log` and a read cursor. At most 4 run at once.
- **Ports.** After each step, the harness reads `/proc/net/tcp` (state `0A`) and maps listening sockets to processes. It reports once: `[process] p1 is listening on port 5173`. The UI preview attaches to `settings.preview.port`, or else to the first port that opens.
- **Watching.** New error-looking lines from running processes are added as a reminder, capped. When a process exits, the reminder gives its exit code and last lines.
- **Autostart.** When the file named in `settings.autostart.when` exists, the harness runs `settings.autostart.command` as a registry process.

## 4. Hooks (Claude Code contract, trimmed)

**Events:**
- `SessionStart`: its stdout goes into the context.
- `PostToolUse`: a matcher selects tools, an `if` glob selects changed files, and `"batch": true` runs the hook once per step over all changed files.
- `Stop`: the finish gate.

**Command contract:**
- Hooks run in the container with cwd `/workspace`.
- stdin is JSON: `{event, tool_name, tool_input, changed_files, step, stop_blocks}`.
- Exit codes:
  - **0:** pass.
  - **2:** fail or block. stdout and stderr are sent to the model as `[check: NAME] failed:` followed by the output.
  - **anything else:** a hook error, shown to the user only.
- `{files}` in a command is replaced with the matching files, shell-quoted.

**Settings merge** (later files win for scalars, arrays are concatenated):
1. built-in defaults
2. `/opt/bonsai/settings.json` (image)
3. `/workspace/.bonsai/settings.json` (project, copied with the template)
4. `/workspace/.bonsai/settings.local.json`

A snapshot of the settings is taken at the start of each run. A built-in rule denies writes to `.bonsai/settings*.json`, so the agent can't switch off its own gate.

**Schema:**
```json
{
  "limits": {"maxSteps": 5000, "maxMinutes": 1440, "thinkBudget": 800, "maxStopBlocks": 3},
  "autostart": {"when": "package.json", "command": "npm run dev"},
  "preview": {"port": 5173},
  "hooks": {
    "SessionStart": [{"command": "…"}],
    "PostToolUse": [{"name": "typecheck", "matcher": "write_file|edit", "if": "*.{ts,tsx}", "batch": true, "command": "…"}],
    "Stop": [{"name": "typecheck", "command": "…"}]
  }
}
```

The project's own checks are hook commands defined in the template's `.bonsai/settings.json`:
- typecheck: `tsc`
- imports: `check-imports.mjs`
- page: `check-page.mjs`, which takes an optional screenshot

## 5. Loop

### Event log

There is one invariant: `session.tokens == tokenize(render(log))`. Every injection goes through a single `append()`. The three drift bugs of v2 are fixed:
- the plan and explore reminders went into the KV cache but not the transcript;
- the injected `</think>` was not recorded;
- the forced `file_path` was not recorded in the arguments.

### Turn loop (per user message)

1. Snapshot the settings.
2. Append the user message. For the first message, include `<environment>`, the memory files and the skills; afterwards include pending deltas.
3. Take a checkpoint.
4. Loop:
   1. **Guards before the step.** Steps and wall time: at 80% of a budget, add a "wrap up" reminder. Context: compact once it passes `COMPACT_AT`.
   2. **Generate** with the thinking budget, the chanting check and any forced prefix. Record what the harness injected.
   3. **Mid-run user messages.** Queued messages are appended as `<system-reminder>The user says: …</system-reminder>` (Codex/Claude Code queue).
   4. **No calls in the reply:**
      - if the reply announces more work, send "Please continue." (at most 2 times);
      - otherwise run the Stop hooks; a block becomes a user message, up to `maxStopBlocks` times;
      - otherwise finish.
   5. **Calls in the reply:** validate each one (schema, aliases, deny rules), execute, and collect the changed files.
   6. **After the batch of calls:**
      - run the PostToolUse hooks (batched);
      - process events (ports, exits, errors);
      - run the guards (stuck, explore, repeated failure);
      - recite the plan every 8 steps;
      - add deltas (new `AGENTS.md`, restore).

      All of this goes into a single `<system-reminder>` at the end of the last tool response.
   7. Take a checkpoint if files changed. Extend the generation prompt.
5. **Stop reasons:** `done | cancelled | max_steps | max_time | context | stuck | stop_hook_exhausted | length`.

### Guards (deterministic; no extra LLM calls)

| Trigger | Threshold | Source |
|---|---|---|
| Same call and same result | ×4 | OpenHands |
| Same call and same error | ×3 | OpenHands; opencode doom loop |
| A-B alternation | 6 steps | OpenHands |
| Text-only turns in a row | 3 | OpenHands monologue |
| 50-char chunk repeated | ×10 in the last 5000 chars | Gemini CLI / Qwen Code |
| Steps without a file change | 6 | ours |
| Same edit failing on the same file | ×2 | ours: "re-read or use write_file" |
| Parse errors in a row | 2 | ours: force the XML skeleton |

**Escalation ladder** (Gemini CLI style; resets when a file change makes the checks pass):
1. **Nudge:** a reminder naming the repetition, "list 3 causes, try a different approach", plus the plan.
2. **Force:** the next turn starts with the prefill `<tool_call>\n<function=`.
3. **Stop:** reason `stuck`; hand back to the user with the last error.

**Thinking budget.** 1500 tokens per turn, then the harness injects a closing sentence and `</think>` and records the injection. If the chanting check fires inside `<think>`, close the thinking early.

### Context (stable prefix; compaction drops old turns)

Compaction triggers at 32k tokens. It runs again only once the context has grown 6k past the previous compaction (hysteresis), and it aims to get under 18k. No model call is used.

**How it works:**
- Old turns are **dropped**. Each tool call in them becomes one line of an action log, e.g. `- edit src/X.tsx: ok` or `- run_shell_command npm run build: Exit code: 1 (full output: /tmp/bonsai/out/c12.txt)`.
- The log is attached as a reminder to the kept message just before the dropped turns, so events stay in order.
- Nothing call-shaped remains in the history. Validation showed the model imitates any masked call it sees, whether it holds a placeholder or blanked arguments.

**Never dropped:**
- the system turn and all user messages;
- the newest turns, up to about 6k tokens of output and at least 2 turns;
- SKILL.md reads and the newest `update_plan`, kept as stripped-down calls.

**After compaction**, a note goes into the last message containing:
- the task (the first request) and later user messages, including steered ones;
- the plan, the files edited and checks still failing;
- the number of steps left.

The read cache is cleared. Project memory files that were removed get delivered again.

**Success criteria (built-in finish check):** the plan carries a "Success criteria:" checklist of verifiable checks. The finish gate refuses to end a run that changed files while that list is missing or has unticked items. The agent ticks each criterion with its evidence as it verifies it. The 3-block cap counts only consecutive blocks with no progress (no file change and no plan update) in between, so long runs aren't cut off.

**Loops and progress (no step limit):** runs end on verified success criteria, on a confirmed loop, on the 24 h limit, or when you press Stop. On top of exact-repeat detection there are two detectors:
- **Recurring issue:** an error fingerprint (exception type and message, compiler error code, failed check, with numbers, paths and values normalised) seen 4 times while fixes were being tried.
- **No progress:** 30 steps with no criterion verified, no plan step done, no failing check fixed, and no failed command recovered. Waiting on a running process doesn't count.

The first trigger adds a reminder. Further triggers start an **introspection turn**: the model receives its recent trajectory and must answer, without tools, ASSESSMENT / EVIDENCE / ROOT CAUSE / NEW APPROACH / NEED USER.
- "progressing" with evidence resets the counters.
- Otherwise the run continues with the stated new approach.
- A question for the user pauses the run (stop reason `needs_user`).
- 3 introspections without progress in between stop the run with a wrap-up summary (`stuck`).

**Long runs:** the limit is 24 h per request, with no step limit. `read_process wait_seconds` (up to 3600) lets the agent sleep through training instead of polling. Waiting on a running process counts neither as stuck nor as exploring.

**Step or time limit:** the Stop hooks run for the record. Then the model gets a single wrap-up turn, without thinking or tools, to report status to the user.

### Interruption

- **Mid-run messages** go into a queue and are delivered at the next step boundary. They are added to the task's request list, so compaction restates them. Messages still queued when a run ends start the next run.
- **Stop** closes the turn cleanly: `<|im_end|>` is appended and every pending call gets the result `[interrupted by user]`.
- **Restore** rewinds only the workspace. The next step gets a delta: "workspace restored to X; re-read files".

## 6. Code layout

```
bonsai_lens/agent/
  __init__.py     CodeAgent, MAX_CONTEXT
  tools.py        schemas, aliases, validation, handlers (generic)
  parser.py       Qwen3-coder XML parsing, partial-call streaming helpers
  edits.py        tiered matching for `edit`
  prompts.py      system prompt + environment/memory/skills/reminder templates
  settings.py     settings discovery, merge, hook runner
  guards.py       budgets, stuck/loop/chanting detectors, escalation ladder
  context.py      event log, render invariant, masking
  loop.py         CodeAgent.run: the turn loop
bonsai_lens/sandbox.py   container, exec, files, process registry, ports, checkpoints
sandbox/
  Dockerfile
  bonsai/AGENTS.md, bonsai/settings.json, bonsai/skills/*/SKILL.md, bonsai/tools/*.mjs
  template/   (overlay onto the Vite react-ts starter: AGENTS.md, .bonsai/settings.json, vite.config.ts, …)
```

## 7. Validation

- **Unit tests** (no model, no Docker): parser, aliases, edit tiers, settings merge, hook result handling, guards, masking, and the render invariant against the real chat template.
- **Sandbox integration:** processes and ports, hooks running in the container, checkpoints.
- **End-to-end:** the "Night Approach" WebGL task, compared with the v2 baseline (run 2: 40/40 steps used, 1 runtime error and 2 layout problems left). Measured:
  - steps used and wall time
  - failures by type: parse, alias hits, edit failures
  - guard triggers
  - checks passing at the end
  - the final screenshot
