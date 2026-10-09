"""Qwen3-coder XML tool calls: parsing, aliases, schema coercion.

    <tool_call>\\n<function=NAME>\\n<parameter=K>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>

Mirrors the official qwen3coder_tool_parser rules: a value ends at </parameter>, at the next
<parameter=, or at </function>; exactly one leading and one trailing newline belong to the format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_CALL = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)

# Near-miss names seen from Qwen-family models and other harnesses' conventions. Accepted silently,
# counted in `Call.aliases` so the logs show which convention the model prefers.
TOOL_ALIASES = {
    "bash": "run_shell_command", "shell": "run_shell_command", "execute_bash": "run_shell_command",
    "exec_command": "run_shell_command", "run_command": "run_shell_command",
    "edit_file": "edit", "str_replace": "edit", "replace": "edit", "str_replace_editor": "edit",
    "create_file": "write_file", "write": "write_file", "read": "read_file", "view": "read_file",
    "list_files": "glob", "list_dir": "glob", "ls": "glob", "find_files": "glob",
    "grep": "grep_search", "search": "grep_search", "rg": "grep_search", "search_files": "grep_search",
    "todo_write": "update_plan", "plan": "update_plan",
    "attempt_completion": "finish", "complete": "finish", "submit": "finish",
}
PARAM_ALIASES = {
    "path": "file_path", "filepath": "file_path", "filename": "file_path", "file": "file_path",
    "old_str": "old_string", "new_str": "new_string", "old": "old_string", "new": "new_string",
    "text": "content", "cmd": "command", "id": "process_id", "pid": "process_id", "query": "pattern",
    "regex": "pattern", "directory": "path", "dir": "path", "todos": "plan", "message": "result",
    "summary": "result",
}


def strip1(v: str) -> str:
    """Drop exactly one leading and one trailing newline (the format's own), keep the rest."""
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


@dataclass
class Call:
    raw: str
    name: str | None = None  # canonical tool name (after aliasing)
    args: dict = field(default_factory=dict)
    error: str | None = None
    aliases: list[str] = field(default_factory=list)  # e.g. ["bash->run_shell_command", "path->file_path"]
    ignored: list[str] = field(default_factory=list)  # parameters the tool doesn't have (dropped, reported)

    @property
    def ok(self) -> bool:
        return self.error is None


def parse(raw: str, schemas: dict[str, dict]) -> Call:
    """Parse the inside of one <tool_call> block against `schemas` {name: parameters-schema}."""
    m = _CALL.search(raw)
    if not m:
        return Call(raw, error="Error: malformed tool call. Next: use exactly <tool_call>\\n<function=NAME>\\n"
                    "<parameter=KEY>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>.")
    call = Call(raw)
    name = m.group(1).strip()
    if name not in schemas and name in TOOL_ALIASES:
        call.aliases.append(f"{name}->{TOOL_ALIASES[name]}")
        name = TOOL_ALIASES[name]
    call.name = name
    args = {k.strip(): strip1(v) for k, v in _PARAM.findall(m.group(2))}
    if name == "finish":  # not a tool: the model wanted to end its turn
        call.args = {"result": next(iter(args.values()), "")}
        return call
    spec = schemas.get(name)
    if spec is None:
        call.args = args
        call.error = f"Error: unknown tool {name!r}. Next: use one of: {', '.join(schemas)}."
        return call
    props = spec["properties"]
    for k in list(args):
        if k not in props:
            canon = PARAM_ALIASES.get(k)
            if canon == "path" and "file_path" in props:
                canon = "file_path"
            if canon in props and canon not in args:
                args[canon] = args.pop(k)
                call.aliases.append(f"{k}->{canon}")
    for k in [k for k in args if k not in props]:
        call.ignored.append(k)
        del args[k]
    for k, v in list(args.items()):
        typ = props.get(k, {}).get("type")
        if typ == "integer":
            try:
                args[k] = int(str(v).strip())
            except ValueError:
                call.args = args
                call.error = f"Error: parameter {k} must be an integer, got {v!r}."
                return call
    call.args = args
    missing = [k for k in spec.get("required", []) if k not in args]
    if missing:
        call.error = (f"Error: {name} is missing required parameter(s): {', '.join(missing)} "
                      f"(got: {', '.join(args) or 'none'}). Next: call {name} with {', '.join(props)}.")
    return call


def partial(raw: str) -> tuple[str | None, dict[str, str]]:
    """Name and parameters so far of a still-streaming call (for live UI views)."""
    m = re.search(r"<function=([^>\n]+)>", raw)
    params = {k.strip(): v.lstrip("\n") for k, v in _PARAM.findall(raw)}
    return (m.group(1).strip() if m else None), params


def to_json(args: dict) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)
