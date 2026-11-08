"""Scripted oracle and the one place prompts are rendered (FR-004, FR-006).

Public API:

* ``oracle_call(task) -> Call``: the one correct call (follows the tool doc).
* ``habit_call(task) -> Call``: what a model following habit would emit.
* ``render_prompt(task, follow_docs=False) -> str``: tool doc + request. With
  ``follow_docs=True`` exactly one extra line, ``FOLLOW_DOCS_LINE``, is added
  right after the reply-format line (the prompt-only baseline of FR-006).
* ``call_to_json(call) -> str``: the canonical JSON text of a call.
* ``to_chat_example(task) -> dict``: SFT trace ``{"messages": [user, assistant]}``
  whose assistant message is the oracle call as JSON.
"""

from __future__ import annotations

import copy
import json

from alien.generator import Call, Task

FOLLOW_DOCS_LINE = (
    "Follow the tool documentation exactly, even where it contradicts what the tool "
    "name or common practice suggests."
)

_INTRO = "You are an agent that calls tools. Read the documentation of the tool below."
_FORMAT_LINE = (
    'Reply with exactly one JSON object of the form {"name": "<tool name>", '
    '"arguments": {"<argument>": <value>}} and nothing else.'
)


def oracle_call(task: Task) -> Call:
    """The correct call, as the tool doc specifies. Returns a copy."""
    return copy.deepcopy(task.oracle_call)


def habit_call(task: Task) -> Call:
    """The call a model following habit (ignoring the twist) would emit. Returns a copy."""
    return copy.deepcopy(task.habit_call)


def render_prompt(task: Task, follow_docs: bool = False) -> str:
    lines = [
        _INTRO,
        "",
        f"### Tool: {task.tool.name}",
        *task.tool.doc.split("\n"),
        "",
        _FORMAT_LINE,
    ]
    if follow_docs:
        lines.append(FOLLOW_DOCS_LINE)
    lines += ["", f"Request: {task.request}"]
    return "\n".join(lines)


def call_to_json(call: Call) -> str:
    return json.dumps(call)


def to_chat_example(task: Task) -> dict:
    return {
        "messages": [
            {"role": "user", "content": render_prompt(task)},
            {"role": "assistant", "content": call_to_json(task.oracle_call)},
        ]
    }
