"""Bonsai Code: a generic coding-agent harness for the local model (docs/coding-agent-architecture.md).

Project and environment knowledge is not in this package: it comes from AGENTS.md memory files,
SKILL.md skills and .bonsai/settings.json hooks inside the sandbox.
"""

from .loop import COMPACT_AT, MAX_CONTEXT, CodeAgent

__all__ = ["CodeAgent", "MAX_CONTEXT", "COMPACT_AT"]
