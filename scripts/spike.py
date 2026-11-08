"""Helpers for the Colab spike notebook (notebooks/spike.ipynb; AS-1, AS-2, AS-3, B-1).

Pure Python with no heavy imports at module level: the notebook imports this file on a T4,
the unit tests import it on the laptop. The spike runs once on Colab before anything else and
writes results/spike.json, which scripts/check_results.py then validates.

Public API:

* ``build_sft_texts(n=64, seed=0) -> list[dict]``: chat examples for the 20-step SFT.
* ``build_grpo_rows(n=8, seed=0) -> list[dict]``: conversational GRPO rows with a task_id.
* ``tasks_by_id(n=8, seed=0) -> dict[str, Task]``: the tasks behind those rows.
* ``fallback_exact_match(task, completion) -> float``: tiny reward used if alien.reward is absent.
* ``capture_first_step(completions, kwargs) -> dict``: what the first reward call received.
* ``loss_is_nan(losses) -> bool``; ``collect_package_versions(names) -> dict``.
* ``build_spike_results(...) -> dict``; ``write_results(path, doc)``.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import check_results
from alien.generator import Task
from alien.oracle import render_prompt, to_chat_example
from alien.splits import make_splits

MAX_CAPTURE_CHARS = 2000
_CONVERSATION_KEYS = ("prompts", "conversation", "messages")


def _sample_train_tasks(n: int, seed: int) -> list[Task]:
    train = make_splits(seed).train
    if not 1 <= n <= len(train):
        raise ValueError(f"n must be between 1 and {len(train)}, got {n}")
    return random.Random(f"spike:{seed}").sample(train, n)


def build_sft_texts(n: int = 64, seed: int = 0) -> list[dict]:
    """``n`` chat examples ``{"messages": [user, assistant]}`` from the train split."""
    return [to_chat_example(task) for task in _sample_train_tasks(n, seed)]


def tasks_by_id(n: int = 8, seed: int = 0) -> dict[str, Task]:
    """The ``n`` train tasks behind ``build_grpo_rows``, keyed by task id."""
    return {task.id: task for task in _sample_train_tasks(n, seed)}


def build_grpo_rows(n: int = 8, seed: int = 0) -> list[dict]:
    """Conversational GRPO rows ``{"prompt": [user message], "task_id": id}``.

    Same shape as ``alien.reward.to_grpo_rows``; built here so the spike does not depend on
    that module.
    """
    return [
        {"prompt": [{"role": "user", "content": render_prompt(task)}], "task_id": task.id}
        for task in _sample_train_tasks(n, seed)
    ]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    messages = [completion] if isinstance(completion, dict) else completion
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                return content if isinstance(content, str) else ""
    return ""


def fallback_exact_match(task: Task, completion: Any) -> float:
    """1.0 iff the completion text is one JSON call equal to the task's oracle call, else 0.0."""
    try:
        call = json.loads(_completion_text(completion).strip())
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if call == task.oracle_call else 0.0


def _stringify(value: Any) -> str:
    """Text of ``value`` that never raises and is never empty (the validator needs text)."""
    if isinstance(value, str):
        return value if value.strip() else repr(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _is_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(m, dict) and "role" in m for m in value)
    )


def capture_first_step(completions: Any, kwargs: Mapping[str, Any]) -> dict:
    """What the reward function received on the first GRPO step (B-1, AS-3).

    Returns ``{"completions_0", "kwargs_keys", "reward_sees_full_conversation"}``.
    ``completions_0`` is ``completions[0]`` as text, truncated to 2000 characters.
    ``reward_sees_full_conversation`` is True when ``completions[0]`` is a list of messages
    with more than one message, OR ``kwargs`` has a ``prompts``, ``conversation`` or
    ``messages`` entry whose first item is a list of role/content messages (the prompt side
    of the conversation reaches the reward). If False, score from the completion text alone
    (AS-3 impact).
    """
    first = completions[0] if isinstance(completions, Sequence) and len(completions) else None
    text = _stringify(first) if first is not None else "<no completions>"
    in_completion = _is_message_list(first) and len(first) > 1
    in_kwargs = False
    for key in _CONVERSATION_KEYS:
        value = kwargs.get(key)
        if isinstance(value, list) and value and _is_message_list(value[0]):
            in_kwargs = True
    return {
        "completions_0": text[:MAX_CAPTURE_CHARS],
        "kwargs_keys": sorted(str(key) for key in kwargs),
        "reward_sees_full_conversation": in_completion or in_kwargs,
    }


def loss_is_nan(losses: Iterable[float]) -> bool:
    """True when any loss is NaN or infinite."""
    return any(not math.isfinite(loss) for loss in losses)


def collect_package_versions(names: Iterable[str]) -> dict[str, str]:
    """Installed version of each package, or "not installed"."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_spike_results(
    *,
    install_ok: bool,
    sft_losses: Sequence[float],
    grpo_step_ok: bool,
    first_step: Mapping[str, Any],
    python_version: str,
    package_versions: Mapping[str, str],
    seed: int = 0,
    git_sha: str | None = None,
) -> dict:
    """The ``spike`` results document (shared contract); validate it with check_results."""
    return {
        "kind": "spike",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "seed": seed,
        "install_ok": install_ok,
        "sft_steps": len(sft_losses),
        "fp16_nan_loss": loss_is_nan(sft_losses),
        "grpo_step_ok": grpo_step_ok,
        "completions_0": first_step["completions_0"],
        "kwargs_keys": list(first_step["kwargs_keys"]),
        "reward_sees_full_conversation": first_step["reward_sees_full_conversation"],
        "python_version": python_version,
        "package_versions": dict(package_versions),
    }


def write_results(path: str | Path, doc: dict) -> None:
    """Write ``doc`` as JSON atomically (temp file, then rename).

    Raises ``ValueError`` listing every validator message, and writes nothing, when
    ``check_results.validate`` reports problems.
    """
    errors = check_results.validate(doc)
    if errors:
        raise ValueError("refusing to write invalid results:\n" + "\n".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)
