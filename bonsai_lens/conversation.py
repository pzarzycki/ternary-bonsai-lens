"""Multi-turn conversation state on top of Engine, with token streaming (used by chat.py and server.py)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from .engine import Engine

IMAGE_ITEM = {"type": "image"}


class Conversation:
    def __init__(
        self,
        engine: Engine,
        system: str | None = None,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 32768,
    ):
        self.e = engine
        self.system = system
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        self.messages: list[dict] = [{"role": "system", "content": self.system}] if self.system else []
        self.images: list[str] = []  # every image in the conversation, in order
        self.session = None
        self.open_reply: str | None = None  # raw text of the last reply if it was cut off (resumable)

    def _render(self, messages):
        return self.e.render(messages, thinking=self.thinking, reasoning_effort=self.reasoning_effort)

    def stream(self, text: str, image: str | None = None, cancel: threading.Event | None = None) -> Iterator[dict]:
        """Add a user turn and generate the reply.

        Yields {"type": "delta", "text": ...} events, then a final {"type": "done", ...} with stats.
        When thinking is on, the reply begins inside the <think> block (the prompt opens it).
        """
        e = self.e
        content = [IMAGE_ITEM, {"type": "text", "text": text}] if image else text
        self._close_open_reply()
        if image:
            self.images.append(image)
            self.session = None  # images are injected at prefill only: re-prefill the whole history
        self.messages.append({"role": "user", "content": content})
        if self.session is None:
            self.session = e.new_session(self._render(self.messages), images=self.images)
        else:
            # Continue the cached session: append only the new user turn + generation prompt.
            tail = self._render([{"role": "user", "content": content}])
            tail = tail[tail.index("<|im_start|>user") :]  # drop the template's auto system block
            e.extend(self.session, "\n" + tail)
        yield from self._generate("", cancel)

    def resume(self, cancel: threading.Event | None = None) -> Iterator[dict]:
        """Continue the last reply after it hit max_tokens or was stopped (same events as stream)."""
        if self.open_reply is None or self.session is None:
            raise ValueError("the last reply is complete; nothing to continue")
        self.messages.pop()  # replaced by the merged reply
        prior, self.open_reply = self.open_reply, None
        yield from self._generate(prior, cancel)

    def _close_open_reply(self):
        """A cut-off reply stays open (resumable) until the next user turn; close it then."""
        if self.open_reply is not None and self.session is not None:
            self.session.tokens.append(self.e.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.open_reply = None

    def _generate(self, prior: str, cancel: threading.Event | None) -> Iterator[dict]:
        e, s = self.e, self.session
        start, t0, first = len(s.tokens), time.perf_counter(), None
        emitted, pending, stop = "", [], "length"
        for tok in e.generate(s, self.temperature, self.top_p, self.top_k):
            if first is None:
                first = time.perf_counter()
            if tok in e.eos_ids:
                stop = "eos"
                break
            # Detokenize only the not-yet-emitted tail; hold back incomplete UTF-8 sequences.
            pending.append(tok)
            piece = e.decode(pending)
            if not piece.endswith("\ufffd"):
                yield {"type": "delta", "text": piece}
                emitted += piece
                pending = []
            if len(s.tokens) - start >= self.max_tokens:
                break
            if cancel is not None and cancel.is_set():
                stop = "cancelled"
                break

        gen = [t for t in s.tokens[start:] if t not in e.eos_ids]
        text = e.decode(gen)
        if text[len(emitted) :]:
            yield {"type": "delta", "text": text[len(emitted) :]}
        raw = prior + text
        # A cut-off reply is left open so it can be resumed; the next user turn closes it.
        self.open_reply = raw if stop != "eos" else None
        reasoning, reply = "", raw
        if "</think>" in raw:
            reasoning, reply = raw.split("</think>", 1)
            reply = reply.lstrip()
        elif self.thinking:
            reasoning, reply = raw, ""  # still inside <think>
        self.messages.append({"role": "assistant", "content": reply, "reasoning_content": reasoning.strip()})
        now = time.perf_counter()
        decode_s = now - (first or t0)
        yield {
            "type": "done",
            "stop": stop,
            "resumable": stop != "eos",
            "tokens": len(gen),
            "prefill_s": (first or t0) - t0,
            "tok_per_s": len(gen) / decode_s if decode_s > 0 else 0.0,
            "prompt_tokens": start,
            "context": len(s.tokens),
        }
