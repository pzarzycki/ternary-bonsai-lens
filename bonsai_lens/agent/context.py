"""The transcript: structured messages kept in lock-step with the model's append-only KV cache.

Single path for every injection: the harness builds the complete content (tool output + reminder)
first, then appends it to `messages` AND extends the session from the same text, so nothing the
model saw is missing from the transcript when it is re-rendered after masking.

The incremental tails below are byte-identical to the chat template's rendering
(tests/test_agent.py checks this against models/*/chat_template.jinja).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from . import prompts

PROTECT_TAIL_CHARS = 24_000  # newest tool outputs kept verbatim when masking (~6k tokens)


def _safe(s: str) -> str:
    """Text from tools must not tokenize into chat-control tokens."""
    return (s.replace("<|", "<​|").replace("<tool_response>", "<tool_response​>")
            .replace("</tool_response>", "</tool_response​>").replace("<tool_call>", "<tool_call​>")
            .replace("</tool_call>", "</tool_call​>"))


class Transcript:
    def __init__(self, engine, render: Callable[[list[dict]], str], thinking: bool):
        self.e, self.render, self.thinking = engine, render, thinking
        self.messages: list[dict] = []
        self.session = None
        self.request = ""  # the latest real user request (re-stated after compaction)
        self.open = False  # an assistant generation prompt is pending in the cache
        self.images: list[str] = []  # host paths of images in the transcript, in order
        self.compacted_at = 0  # context size right after the last compaction (hysteresis)

    @property
    def tokens(self) -> int:
        return len(self.session.tokens) if self.session else 0

    def gen_prompt(self) -> str:
        return "<|im_start|>assistant\n" + ("<think>\n" if self.thinking else "<think>\n\n</think>\n\n")

    def start(self, system: str, first_user: str, on_mlx):
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": first_user}]
        prompt = self.render(self.messages)
        self.session = on_mlx(lambda: self.e.new_session(prompt))
        self.open = True

    def user(self, text: str):
        """A user-role message (a real request, or the finish gate). The template treats it as a new query."""
        self.messages.append({"role": "user", "content": text})
        self.e.extend(self.session, f"\n<|im_start|>user\n{text}<|im_end|>\n{self.gen_prompt()}")
        self.open = True

    def user_with_image(self, text: str, image_path: str, on_mlx):
        """A user message carrying an image. Images enter the model only at prefill, so this re-prefills
        the whole transcript once (with every image still in it)."""
        self.messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]})
        self.images.append(image_path)
        prompt = self.render(self.messages)
        self.session = on_mlx(lambda: self.e.new_session(prompt, images=list(self.images)))
        self.open = True

    def close_open_turn(self):
        """A run ended right after a generation prompt (cancel, limits): close that assistant turn as an
        empty one, exactly as the template renders it, so cache and transcript stay identical."""
        if self.open and self.session is not None:
            self.e.extend(self.session, ("\n</think>\n\n" if self.thinking else "") + "<|im_end|>")
            self.assistant("", "", [])

    def assistant(self, think: str, text: str, calls: list[dict]):
        """Record the turn the model generated (its tokens are already in the cache)."""
        self.messages.append({"role": "assistant", "content": text, "reasoning_content": think,
                              "tool_calls": [{"function": c} for c in calls]})
        self.open = False

    def tool_results(self, outputs: list[str], oks: list[bool] | None = None, refs: list[str | None] | None = None):
        outs = [_safe(o) for o in outputs]
        for k, o in enumerate(outs):  # `ok`/`ref` are extra keys the template ignores (used by compaction)
            m = {"role": "tool", "content": o, "ok": oks[k] if oks else True}
            if refs and refs[k]:
                m["ref"] = refs[k]
            self.messages.append(m)
        body = "".join(f"\n<tool_response>\n{o}\n</tool_response>" for o in outs)
        self.e.extend(self.session, f"\n<|im_start|>user{body}<|im_end|>\n{self.gen_prompt()}")
        self.open = True

    # ------------------------------------------------------------------ compaction
    def _turns(self, msgs: list[dict]) -> list[tuple[int, list[int]]]:
        """(assistant index, [its tool result indexes]) for every model turn."""
        turns, cur = [], None
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                cur = (i, [])
                turns.append(cur)
            elif m["role"] == "tool" and cur is not None:
                cur[1].append(i)
            elif m["role"] in ("user", "system"):
                cur = None
        return turns

    @staticmethod
    def _log_line(call: dict, result: dict | None) -> str:
        name, a = call.get("name", "tool"), call.get("arguments", {})
        arg = a.get("file_path") or a.get("command") or a.get("pattern") or a.get("process_id") or ""
        arg = str(arg).replace("\n", " ")[:90]
        content = (result or {}).get("content", "")
        first = content.split("\n", 1)[0][:100]
        if name in ("write_file", "edit", "update_plan", "read_file", "glob", "grep_search"):
            status = "ok" if (result or {}).get("ok", True) else f"failed: {first}"
        else:
            status = first
        ref = (result or {}).get("ref")
        return f"- {name} {arg}: {status}" + (f" (full output: {ref})" if ref else "")

    def compacted(self, protect_chars: int = PROTECT_TAIL_CHARS) -> list[dict]:
        """Codex / Claude Code-style compaction without a model call: old turns are DROPPED and replaced by
        a one-line-per-call action log, appended (as a reminder) to the kept message right before them, so
        the order of events is preserved. Nothing call-shaped remains for the model to imitate (masked calls
        with blanked arguments were copied in validation).
        Kept: system, every user message, the newest turns (up to protect_chars of tool output, at least 2),
        skill loads (only the SKILL.md read itself), and the newest plan update (only that call)."""
        msgs = copy.deepcopy(self.messages)
        for m in msgs:  # images are not carried through compaction (they'd need a re-prefill with pixels)
            if isinstance(m.get("content"), list):
                m["content"] = "\n".join(p.get("text", "[a screenshot was shown here]") if p.get("type") == "text"
                                          else "[a screenshot was shown here]" for p in m["content"])
        turns = self._turns(msgs)
        if len(turns) <= 2:
            return msgs
        keep, budget = {t[0] for t in turns[-2:]}, protect_chars
        for ai, tools in reversed(turns):
            budget -= sum(len(msgs[i]["content"]) for i in tools) + len(str(msgs[ai].get("tool_calls", "")))
            if budget < 0:
                break
            keep.add(ai)
        # Pinned calls inside otherwise-dropped turns: (assistant index, call position).
        pinned, newest_plan = set(), None
        for ai, tools in turns:
            for k, tc in enumerate(msgs[ai].get("tool_calls", [])):
                f = tc["function"]
                if f.get("name") == "read_file" and str(f.get("arguments", {}).get("file_path", "")).endswith("SKILL.md"):
                    pinned.add((ai, k))
                if f.get("name") == "update_plan":
                    newest_plan = (ai, k)
        if newest_plan:
            pinned.add(newest_plan)
        turn_of = {ai: tools for ai, tools in turns}
        out, pending_log, changed = [], [], False

        def flush_log():
            if pending_log and out:
                block = prompts.reminder(["Earlier steps were compacted here (one line per tool call; outputs "
                                          "removed):"] + (pending_log if len(pending_log) <= 80 else
                                          pending_log[:20] + [f"- … {len(pending_log) - 60} more …"] + pending_log[-40:]))
                c = out[-1]["content"]
                if c.rstrip().endswith("</system-reminder>"):  # merge: one reminder block per message
                    k = c.rstrip().rfind("</system-reminder>")
                    out[-1]["content"] = c[:k].rstrip() + "\n" + block.strip()[len("<system-reminder>\n"):]
                else:
                    out[-1]["content"] = c + block
            pending_log.clear()

        recent = set(sorted(keep)[-2:])
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m["role"] != "assistant":
                if m["role"] == "user":
                    flush_log()
                out.append(m)
                i += 1
                continue
            tools = turn_of[i]
            calls = m.get("tool_calls", [])
            if i in keep:
                flush_log()
                if i not in recent:
                    m["reasoning_content"] = ""
                out.append(m)
                out += [msgs[t] for t in tools]
            else:
                changed = True
                keep_k = [k for k in range(len(calls)) if (i, k) in pinned and k < len(tools)]
                for k, c in enumerate(calls):
                    if k not in keep_k:
                        pending_log.append(self._log_line(c["function"], msgs[tools[k]] if k < len(tools) else None))
                if keep_k:  # a stripped-down turn holding only the pinned calls (no prose, no reasoning)
                    flush_log()
                    out.append({"role": "assistant", "content": "", "reasoning_content": "",
                                "tool_calls": [calls[k] for k in keep_k]})
                    out += [msgs[tools[k]] for k in keep_k]
            i += 1 + len(tools)
        flush_log()
        return out if changed else msgs

    def compact(self, on_mlx, target_tokens: int, note: str) -> tuple[int, int] | None:
        """Compact until the rendered prompt is under target_tokens; re-prefill once. The compaction note
        goes into the last message (merged into its reminder block), so it doesn't start a new query."""
        before = self.tokens
        for protect in (PROTECT_TAIL_CHARS, PROTECT_TAIL_CHARS // 3, 0):
            msgs = self.compacted(protect)
            changed = msgs is not self.messages and msgs != self.messages
            c = msgs[-1]["content"]  # one reminder block per message: merge into an existing one
            if c.rstrip().endswith("</system-reminder>"):
                k = c.rstrip().rfind("</system-reminder>")
                msgs[-1]["content"] = c[:k].rstrip() + "\n" + note.strip() + "\n</system-reminder>"
            else:
                msgs[-1]["content"] = c + prompts.reminder([note])
            prompt = self.render(msgs)
            if len(self.e.tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
                break
        if not changed:  # nothing left to drop: don't pay for another prefill
            return None
        self.messages = msgs
        self.images = []
        self.session = on_mlx(lambda: self.e.new_session(prompt))
        self.open = True
        self.compacted_at = self.tokens
        return before, self.tokens
