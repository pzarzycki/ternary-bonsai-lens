#!/usr/bin/env python3
"""Terminal chat with Ternary Bonsai 2 27B (MLX).   uv run chat.py

Commands:
  /image <path>   attach an image to your next message
  /think          toggle thinking mode (default: off)
  /temp <t>       sampling temperature (0 = greedy; default 1.0)
  /system <text>  set system prompt (resets conversation)
  /reset          clear conversation
  /quit           exit
"""

import argparse
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from bonsai_lens.conversation import Conversation
from bonsai_lens.engine import DEFAULT_PACK, Engine

console = Console()


class Chat:
    def __init__(self, engine: Engine, max_tokens: int):
        self.conv = Conversation(engine, max_tokens=max_tokens)
        self.pending_image: str | None = None

    @property
    def images(self):
        return self.conv.images

    def reset(self):
        self.conv.reset()

    def user_turn(self, text: str):
        image, self.pending_image = self.pending_image, None
        if image and self.conv.session is not None:
            console.print("[dim](image attached mid-conversation: re-prefilling history)[/dim]")
        for ev in self.conv.stream(text, image=image):
            if ev["type"] == "delta":
                console.print(ev["text"], end="", markup=False, highlight=False)
            else:
                console.print(
                    f"\n[dim]{ev['tokens']} tok · prefill {ev['prefill_s']:.2f}s · {ev['tok_per_s']:.1f} tok/s"
                    f" · ctx {ev['context']}[/dim]"
                )

    def command(self, line: str) -> bool:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/quit":
            return False
        if cmd == "/reset":
            self.reset()
            console.print("[dim]conversation cleared[/dim]")
        elif cmd == "/think":
            self.conv.thinking = not self.conv.thinking
            console.print(f"[dim]thinking {'on' if self.conv.thinking else 'off'} (applies from next turn)[/dim]")
        elif cmd == "/temp":
            self.conv.temperature = float(arg)
            console.print(f"[dim]temperature = {self.conv.temperature}[/dim]")
        elif cmd == "/system":
            self.conv.system = arg or None
            self.reset()
            console.print("[dim]system prompt set; conversation cleared[/dim]")
        elif cmd == "/image":
            p = Path(arg).expanduser()
            if not p.is_file():
                console.print(f"[red]no such file: {p}[/red]")
            else:
                self.pending_image = str(p)
                console.print(f"[dim]image attached to next message: {p.name}[/dim]")
        else:
            console.print(__doc__)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=None)
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda", "llama"), default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    args = ap.parse_args()

    with console.status("Loading Ternary Bonsai 2 27B…"):
        t0 = time.perf_counter()
        engine = Engine(args.pack, backend=args.backend)
    console.print(f"[green]Loaded in {time.perf_counter() - t0:.1f}s.[/green] Type /help for commands.\n")

    chat = Chat(engine, args.max_tokens)
    ps = PromptSession(history=FileHistory(str(Path.home() / ".bonsai_chat_history")))
    while True:
        try:
            line = ps.prompt("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            if not chat.command(line):
                break
            continue
        console.print("[bold cyan]bonsai ›[/bold cyan] ", end="")
        try:
            chat.user_turn(line)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted (conversation reset)[/yellow]")
            chat.reset()


if __name__ == "__main__":
    main()
