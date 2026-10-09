# Building a good coding agent: research notes for the Code tab

Scope: how open coding agents are structured (loop, tools, context, prompts), and what that means for
ours. Our constraints are a local 2-bit 27B model at about 15–40 tok/s, a 56k context cap, Qwen3-coder
XML tool calls, and a Docker sandbox that builds React apps with a live preview.

## Recommendations, ranked by expected impact (✅ = implemented in `bonsai_lens/agent.py`)

1. ✅ **Ground truth after every edit, and a check before the agent may finish.**
   - After every `write_file`/`edit_file` on TS/TSX, the harness type-checks and appends the errors to that tool result.
   - It also appends new error lines from the dev-server log.
   - When the model wants to finish, the harness runs the type check. If it fails, the errors go back to the model instead of ending the turn (at most 2 retries).
   - Why:
     - In SWE-agent's ablation, linter-gated edits reached 18.0% vs 15.0% without them.
     - Aider auto-lints and auto-tests.
     - Anthropic's guidance asks for "ground truth from the environment" at each step.
2. ✅ **Two edit tools: whole-file write, and exact replace with tolerant matching.**
   - `write_file` writes the complete content. Bolt and Aider's whole-file format are the most format-robust options for weak models.
   - `edit_file` replaces one exact match. If that fails, it falls back to whitespace-insensitive line matching. Aider found flexible patching cut edit errors 9×.
   - Errors are actionable: "0 matches, the closest lines are 40–52: …", or "3 matches at lines 12, 80, 133".
   - No line-number edits and no unified diffs: Qwen2.5-Coder produced well-formed diffs only 72% of the time.
3. ✅ **Observation masking rather than summarisation, applied in batches.**
   - When the context passes a threshold, old tool outputs and old `write_file` bodies are replaced with one-line placeholders, and old reasoning is dropped. The last few observations stay verbatim.
   - The context is then re-prefilled once. Batching keeps the KV-cache prefix stable between compactions.
   - Why:
     - In *The Complexity Trap* (arXiv 2508.21433), masking matched or beat LLM summarisation at about half the cost; summarisation made trajectories 13–15% longer.
     - SWE-agent's "keep the last 5 observations" also beat keeping the full history.
     - Qwen3-family failure modes (tool-call loops, incomplete calls) show up past about 30–40k tokens, so a smaller working context is safer.
4. ✅ **Sampling.** The Qwen3.5 card for thinking mode with code: `temperature 0.6, top_p 0.95, top_k 20, min_p 0`.
5. ✅ **Informative, capped tool output.**
   - Output is cut to head plus tail, with "N chars omitted".
   - An empty result says "(no output)".
   - `read_file` returns numbered lines and the total line count. SWE-agent's windowed viewer beat showing the whole file.
   - Use quiet npm flags.
6. ✅ **Loop controls.**
   - A step limit.
   - A repeated identical call is flagged back to the model.
   - A malformed call gets an error that restates the expected parameters, and the error stays in context (the Manus approach).
7. ✅ **Small toolset; the harness owns the dev server.**
   - Vite is started automatically once `package.json` exists, and the prompt says never to restart it. Bolt's rule is "do not re-run the dev command"; HMR picks up the changes.
   - Paths are accepted with or without `/workspace/`.
8. ✅ **A short, byte-stable system prompt.**
   - No timestamps, and the tool order is fixed.
   - The section order follows bolt, Codex and Qwen Code: environment, workflow, editing rules, scope and quality, if stuck, style.
9. ✅ **A plan/todo tool** (Codex `update_plan`, Qwen `todo_write`). Ours is shown in the Plan tab.
10. ✅ **A parser as tolerant as Qwen's own.**
    - A value may end at `</parameter>`, at the next `<parameter=`, or at `</function>`.
    - Exactly one leading and one trailing newline is stripped.
    - Values are coerced to the schema's types.
    - Near-miss parameter names (`file_path` → `path`, `old_str` → `old_string`) are mapped to the real ones.

Not done yet (ideas):
- Capture browser-console errors from the preview, e.g. a tiny injected script that posts errors to the server.
- "Recitation": re-show the plan near the end of the context.
- Few-shot tool-call examples in the system prompt.

## What the prompts have in common

- **Section order:**
  1. Role.
  2. Environment and constraints (bolt `<system_constraints>`).
  3. Rules.
  4. Workflow: understand → plan → implement → verify.
  5. Tool usage.
  6. Tone and final message.
  7. A closing reminder.
- **Web-app builders (bolt.new, bolt.diy, open-lovable, llama-coder):**
  - Install dependencies first.
  - Never scaffold again or recreate config files.
  - Never restart the dev server.
  - Write full files with no `...` placeholders, and close every tag and bracket.
  - Split big files into components (bolt.diy refactors anything over 250 lines).
  - When editing an existing app, stay in scope.
  - Aim for a polished, responsive UI with no external assets.
- **Verification:**
  - Codex: "Persist until the task is fully handled end-to-end."
  - Gemini: build it and make sure there are no compile errors.
  - Qwen: run the project's own build and lint before reporting the task complete.
  - Cline: a reply without a tool call means the task is complete.
- **Weaker models:**
  - mini-swe-agent: one action per response, a canned error when the format is wrong, a note that `cd` does not persist between calls, head-and-tail truncation.
  - Qwen Code: different tool-call examples for each model family.
  - Aider: "Do what they ask, but no more."

## Sources

- SWE-agent ACI: https://arxiv.org/abs/2405.15793
- mini-swe-agent: https://github.com/SWE-agent/mini-swe-agent
- The Complexity Trap: https://arxiv.org/abs/2508.21433
- OpenHands condenser: https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents
- OpenHands CodeAct prompt: https://github.com/All-Hands-AI/OpenHands/blob/0.59.0/openhands/agenthub/codeact_agent/prompts/system_prompt.j2
- Anthropic: https://www.anthropic.com/research/building-effective-agents ·
  https://www.anthropic.com/engineering/writing-tools-for-agents ·
  https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Aider: https://aider.chat/docs/more/edit-formats.html · https://aider.chat/docs/unified-diffs.html ·
  https://aider.chat/2024/12/21/polyglot.html · https://aider.chat/docs/usage/lint-test.html
- Manus: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- Qwen Code prompts: https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/core/prompts.ts
- Qwen3-Coder tool parser: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py
- Qwen3.5 sampling: https://huggingface.co/Qwen/Qwen3.5-27B
- Qwen tool-loop issue: https://github.com/ggml-org/llama.cpp/issues/20164
- Codex prompt: https://github.com/openai/codex/blob/main/codex-rs/core/gpt_5_1_prompt.md
- Gemini CLI: https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/prompts/snippets.ts
- Cline: https://github.com/cline/cline
- bolt.new: https://github.com/stackblitz/bolt.new/blob/main/app/lib/.server/llm/prompts.ts
- bolt.diy: https://github.com/stackblitz-labs/bolt.diy/blob/main/app/lib/common/prompts/optimized.ts
- open-lovable: https://github.com/firecrawl/open-lovable
- llama-coder: https://github.com/Nutlope/llamacoder/blob/main/lib/prompts.ts
- E2B fragments: https://github.com/e2b-dev/fragments/blob/main/lib/prompt.ts
