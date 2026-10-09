"""Load the single set of active SWE prompts."""

import json
from pathlib import Path

_PROMPT_DIR = Path(__file__).with_name("thinking_prompts")
for _name in ("system_prompt.txt", "instance_prompt.txt"):
    _path = _PROMPT_DIR / _name
    if not _path.is_file():
        raise FileNotFoundError(_path)

SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
INSTANCE_PROMPT = (_PROMPT_DIR / "instance_prompt.txt").read_text(encoding="utf-8")
if not SYSTEM_PROMPT.strip() or not INSTANCE_PROMPT.strip():
    raise ValueError("thinking prompts must be nonempty")
if "THOUGHT" in SYSTEM_PROMPT or "THOUGHT" in INSTANCE_PROMPT:
    raise ValueError("thinking prompts must not request a separate THOUGHT section")
if INSTANCE_PROMPT.count("{problem_statement}") != 1:
    raise ValueError("instance prompt must contain exactly one problem_statement placeholder")


# UltraDataMiniSWE bash schema; dataset-only positional metadata is omitted.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the terminal, with Python version compatibility.\n",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'\n",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def parse_action(tool_calls: list) -> str:
    """Validate one native action. Malformed model arguments are format errors."""
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("expected exactly one native bash tool call")
    call = tool_calls[0]
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ValueError("tool call must have type=function")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "bash":
        raise ValueError("expected the bash function")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict) or set(arguments) != {"command"}:
        raise ValueError("bash arguments must contain only command")
    command = arguments["command"]
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise ValueError("bash command must be nonempty text without NUL")
    return command


def format_error_message(n_actions: int, finish_reason: str) -> str:
    if type(n_actions) is not int or n_actions < 0:
        raise ValueError("n_actions must be a nonnegative integer")
    if finish_reason not in {"stop", "length", "tool_calls"}:
        raise ValueError(f"unexpected finish reason: {finish_reason!r}")
    prefix = "Your response reached the output token limit. " if finish_reason == "length" else "Format error. "
    return prefix + (
        f"Received {n_actions} tool calls. Provide one or more native bash tool calls "
        'with arguments {"command": "your command"}. Do not write a bash code block in the response text.'
    )
