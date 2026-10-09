"""`edit` matching: exact first, then tolerant tiers (Codex seek_sequence, Gemini CLI, opencode).

Tiers, tried in order until exactly one region matches:
  1. exact substring
  2. line-based: trailing whitespace ignored, then all surrounding whitespace ignored
     (the replacement is re-indented to the file's indentation)
  3. Unicode punctuation normalised to ASCII (dashes, quotes, ellipsis, nbsp)
  4. over-escaped text un-escaped (\\n, \\t, \\")
  5. line numbers copied from read_file output stripped
A match that is much larger than old_string is rejected (opencode's guard).
"""

from __future__ import annotations

import difflib
import re

_PUNCT = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                        "‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                        "„": '"', "…": "...", " ": " ", " ": " ", " ": " "})
_LINENO = re.compile(r"^ *\d+\t", re.M)


def _lines_match(text: str, old: str, norm) -> list[int]:
    """Start line indexes where old's lines equal the file's lines under `norm`."""
    tl, ol = text.split("\n"), old.strip("\n").split("\n")
    key = [norm(l) for l in ol]
    if not any(key):
        return []
    n = len(ol)
    return [i for i in range(len(tl) - n + 1) if [norm(l) for l in tl[i:i + n]] == key]


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _replace_lines(text: str, start: int, old: str, new: str) -> str:
    tl, ol = text.split("\n"), old.strip("\n").split("\n")
    nl = new.strip("\n").split("\n") if new.strip("\n") else []
    # Re-indent: shift the replacement by the difference between the file's and old_string's indent.
    first_old = next((l for l in ol if l.strip()), "")
    first_file = next((l for l in tl[start:start + len(ol)] if l.strip()), "")
    shift = len(_indent(first_file)) - len(_indent(first_old))
    if shift > 0:
        nl = [(" " * shift + l) if l.strip() else l for l in nl]
    elif shift < 0:
        nl = [l[min(-shift, len(_indent(l))):] for l in nl]
    return "\n".join(tl[:start] + nl + tl[start + len(ol):])


def apply_edit(text: str, old: str, new: str) -> tuple[str | None, str]:
    """Returns (new_text, note) or (None, error message for the model)."""
    if not old:
        return None, "Error: old_string is empty. Next: copy the exact lines to replace from read_file output."
    if old == new:
        return None, "Error: old_string and new_string are identical; nothing to change."
    n = text.count(old)
    if n == 1:
        return text.replace(old, new, 1), ""
    if n > 1:
        lines = [text.count("\n", 0, m.start()) + 1 for m in re.finditer(re.escape(old), text)]
        return None, (f"Error: old_string occurs {n} times (lines {', '.join(map(str, lines[:8]))}). "
                      "Next: include 2-3 more surrounding lines so it matches exactly once.")
    tiers = [
        ("ignoring trailing whitespace", old, new, str.rstrip),
        ("ignoring indentation", old, new, str.strip),
        ("after normalising punctuation", old.translate(_PUNCT), new, lambda l: l.translate(_PUNCT).strip()),
    ]
    if "\\n" in old and "\n" not in old.strip("\n"):
        un = lambda s: s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        tiers.append(("after un-escaping \\n", un(old), un(new), str.rstrip))
    if _LINENO.search(old) and all(_LINENO.match(l) for l in old.strip("\n").split("\n") if l.strip()):
        tiers.append(("after removing copied line numbers", _LINENO.sub("", old), _LINENO.sub("", new), str.rstrip))
    for label, o, nw, norm in tiers:
        if text.count(o) == 1:
            return text.replace(o, nw, 1), f" (matched {label})"
        hits = _lines_match(text, o, norm)
        if len(hits) == 1:
            return _replace_lines(text, hits[0], o, nw), f" (matched {label})"
        if len(hits) > 1:
            return None, (f"Error: old_string matches {len(hits)} places {label} (lines "
                          f"{', '.join(str(h + 1) for h in hits[:8])}). Next: include more surrounding lines.")
    return None, _closest(text, old)


def _closest(text: str, old: str) -> str:
    tl, ol = text.split("\n"), old.strip("\n").split("\n")
    first = next((l.strip() for l in ol if l.strip()), "")
    if not tl or not first:
        return "Error: old_string not found. Next: read_file the file and copy the exact text."
    scores = [difflib.SequenceMatcher(None, first, l.strip()).ratio() for l in tl]
    b = max(range(len(tl)), key=scores.__getitem__)
    lo, hi = max(0, b - 2), min(len(tl), b + len(ol) + 2)
    region = "\n".join(f"{j + 1:6}\t{tl[j]}" for j in range(lo, hi))
    return (f"Error: old_string not found. Most similar lines {lo + 1}-{hi}:\n{region}\n"
            "Next: copy the exact current text (without the line numbers), or read_file again.")


def around(text: str, snippet: str, ctx: int = 3) -> str:
    """The edited region with line numbers, so the model sees the result without re-reading."""
    s = snippet.strip("\n")
    i = text.find(s) if s.strip() else -1
    if i < 0:
        return ""
    first = text.count("\n", 0, i)
    last = first + s.count("\n")
    lines = text.split("\n")
    lo, hi = max(0, first - ctx), min(len(lines), last + ctx + 1)
    return "\n".join(f"{j + 1:6}\t{lines[j]}" for j in range(lo, hi))
