# Coding agent: improvement plan from popular harnesses

This plan comes from three research passes: terminal CLIs, harnesses that top the benchmarks, and web-app builders. It also draws on what we saw in our own runs (counter, todo, coffee shop, pomodoro, tip calculator). Background and sources for the first iteration are in [coding-agent-research.md](coding-agent-research.md).

## What the successful harnesses do

| Harness | Mechanisms worth borrowing |
|---|---|
| **Qwen Code** (built for Qwen3-coder, the same XML format as our model) | XML tool-call examples chosen per model family; loop detection (identical calls ×5, A-B alternation, the same file read 8× in 15 calls, repeated 50-character text chunks ×10); microcompaction ("[Old tool result content cleared]", keeps the last 5); a read cache (unchanged re-reads return a placeholder); read-before-edit enforcement; a todo list re-injected each turn |
| **Gemini CLI** | Cascade of edit matchers (exact → flexible → regex → fuzzy) plus un-escaping; tool-output masking with head/tail preview and a file pointer; a "next speaker" check that sends "Please continue." |
| **Codex CLI** | `apply_patch` with a 4-pass context matcher; background processes polled through session ids; AGENTS.md memory |
| **opencode** | 9 edit replacers with a guard against matches that are too large; LSP diagnostics returned with every edit; "did you mean" file suggestions; doom-loop threshold of 3 |
| **SWE-agent / OpenHands** | Edits gated by a linter (+3 points); 100-line read windows; keep the last N observations; a stuck detector (same action and error ×3, ping-pong ×6); critic or verifier based on running the code |
| **Factory / Warp** (Terminal-Bench) | Few tools with simple schemas; an environment bootstrap at turn 0; a forced plan; fail-fast timeouts |
| **Replit / Lovable / Same.dev / bolt** | Runtime errors from the preview fed back to the model; Playwright checks of the page; a screenshot attached to results; a cap of 3 fix loops; checkpoints in a separate git repo (Cline, Roo); shadcn + Tailwind templates with design rules |
| **v0** | "LLM Suspense": fixes hallucinated imports and icons in the stream without calling a model; autofixers after the stream |

Several approaches don't pay off at our size and speed (2-bit 27B, 30–40 tok/s, one GPU):
- splitting work between an architect model and an editor model;
- trained critics;
- summarizing old context with the model;
- fixed best-of-N sampling;
- subagents;
- `apply_patch` (opencode only uses it for GPT models).

## Iteration 2 (implemented)

1. **Headless browser check.** Chromium in the sandbox renders WebGL in software (SwiftShader). `sandbox/check.mjs` reports uncaught errors, `console.error`, failed requests, the Vite error overlay, a blank `#root` and the canvas count, and saves a screenshot.
   - Runs after each clean edit and again in the finish gate.
   - The UI shows a card with the screenshot.
   - Basis: Replit, Lovable, Same.dev, and the common thread among top benchmark entries, which is verification by running the code.
2. **Errors from the open preview.** A Vite plugin in the template's `vite.config.ts` hooks `onerror`, `unhandledrejection`, `console.error` and `vite:error`. Errors are logged to `/tmp/browser.log` and added to edit results. Basis: bolt's preview error alerts.
3. **Deterministic import checker** (`sandbox/check-imports.mjs`). Flags missing relative files, packages that aren't installed, and named exports that don't exist, with the closest real names (e.g. `PlaneTakeof` → `PlaneTakeoff`). Basis: v0's LLM Suspense and open-lovable's package detection.
4. **Finish gate.** Requires tsc, imports and the headless page check to pass, with up to 3 rounds of sending errors back (Same.dev's cap).
5. **Stuck detection.** Triggers on the same call with the same result ×3, the same error ×3, or ping-pong between two calls over 6 steps. It injects "list 3 causes, try a different approach" plus the current plan. Basis: OpenHands and Qwen Code.
6. **Text-repetition detection.** If the last 50-character chunk appears 10 or more times in the recent reasoning or prose, the turn ends with a nudge to act. Basis: Qwen Code and Gemini CLI.
7. **Session bootstrap.** The first user message gets an `<environment>` block with the workspace files and dev-server status. Basis: Factory and Gemini CLI.
8. **Plan recitation.** If the plan hasn't been updated for 8 steps, it is appended to the tool result. Basis: Claude Code todo reminders, Manus, Qwen Code.
9. **Read cache.** An unchanged re-read returns a short placeholder; compaction or a restore clears the cache. A missed read suggests similar filenames. Basis: Qwen Code and opencode.
10. **Edit feedback.** A successful edit shows the edited lines with numbers (SWE-agent, Anthropic). Matching now also handles over-escaped `\n` (Gemini CLI's editCorrector).
11. **"Please continue."** If a reply ends by announcing more work ("Next, I will…") and has no tool call, the harness sends "Please continue." Gemini CLI's next-speaker check, done with a regex instead of a model call.
12. **Checkpoints** in a shadow git repo (`/ckpt`, outside `/workspace`). One is made before each message and after every step that changes files. The History pane offers Restore, and the agent is told about a restore. Basis: Cline and Roo.
13. **Template and prompt.**
    - Pre-installed: Tailwind v4, lucide-react, three, @react-three/fiber, drei.
    - Design rules: 3–5 colours, 2 fonts, tokens (v0, Lovable).
    - 3D guidance.
    - An XML tool-call example (Qwen Code).

## Backlog (ranked)

1. **Screenshot to the model at the finish gate.** Our model is a VLM. Ask it a yes/no question ("blank, broken or unstyled?") rather than asking for an aesthetic judgment.
   - Needs one re-prefill with the image, because images enter only at prefill.
   - Keep only the latest image.
2. **More edit matchers**: indentation-flexible with re-indent, block-anchor Levenshtein, and a guard against matches that are too large (opencode, Gemini CLI).
3. **Read windows**: `read_file` offset/limit with "Showing lines X–Y of Z", and A/B 150–200 lines against today's 600 (SWE-agent).
4. **Masking with pointers**: masked outputs keep a head/tail preview and a path to the full output saved in the container (Gemini CLI). Also mask by recency (last 5–8) regardless of total size.
5. **Mid-stream fixes for icons and imports** (v0 LLM Suspense) instead of after the write.
6. **Smoke test of interactions**: click the main button and assert that the DOM changed. This catches "Potemkin" UIs that render but don't work (Replit).
7. **Best-of-2 when the final check still fails**: resample one turn at a higher temperature and keep the variant that passes (selection by execution only).
8. **shadcn/ui components** in the template, with a short component index in the prompt (Lovable).
9. **`ask_user` tool** for ambiguous requests (Cline, v0).
10. **Grammar-constrained tool-call wrapper**: constrain only the XML structure, never the reasoning (the "Constraint Tax" study).
