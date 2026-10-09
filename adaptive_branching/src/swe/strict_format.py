"""Validate the single Qwen SWE wire grammar, not a repaired/parsed reply.

The fixed thinking template already supplies ``<think>\n`` in the prompt.
One generated span must finish that thought, emit native bash calls, and end
with exactly one im_end. Shell commands are not XML: ordinary <, > and & are
legal, but reserved protocol delimiters inside a command are ambiguous and
must be encoded by the model instead of emitted literally.
"""

import re

_RESERVED = re.compile(r"<\||\|>|</?(?:think|tool_call|tool_response|function|parameter)(?=[\s=>/])")
_CALL = re.compile(
    r"<tool_call>\s*<function=bash>\s*<parameter=command>"
    r"(?P<command>.*?)</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


def format_failure(text: str) -> str | None:
    """Return a model-error reason; invalid caller inputs fail fast.

    Whitespace between grammar elements is insignificant. Multiple calls use
    the same grammar; no prose, code fences, JSON alternative or role markers
    are accepted outside the thought and command bodies.
    """
    if not isinstance(text, str):
        raise TypeError("generated span must be text")
    if "\x00" in text:
        return "nul"
    if not text.endswith("<|im_end|>"):
        return "missing_end"
    body = text[: -len("<|im_end|>")]
    if "<|" in body or "|>" in body:
        return "special_token"
    if body.count("</think>") != 1:
        return "thinking_boundary"
    thought, actions = body.split("</think>")
    if _RESERVED.search(thought):
        return "thinking_protocol_marker"
    actions = actions.strip()
    if not actions:
        return "missing_call"
    offset = 0
    while offset < len(actions):
        match = _CALL.match(actions, offset)
        if match is None:
            return "call_structure"
        command = match["command"]
        if not command.strip():
            return "empty_command"
        if _RESERVED.search(command):
            return "command_protocol_marker"
        next_offset = match.end()
        if next_offset <= offset:
            raise AssertionError("format parser failed to advance")
        offset = next_offset
        while offset < len(actions) and actions[offset].isspace():
            offset += 1
    return None
