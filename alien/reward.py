"""Exact-match GRPO reward and a dead-reward monitor (FR-003, DEC-2, NG-5, E-2, E-4, AS-3).

Public API:

* ``exact_match_reward(task, completion) -> float``: 1.0 iff the completion parses to exactly
  one call equal (canonical form) to the oracle call, else 0.0. No partial credit (DEC-2).
  ``completion`` may be a string, a chat-message list or a single message dict (AS-3 is
  unvalidated, so both the last-message and full-conversation shapes are handled).
* ``to_grpo_rows(tasks) -> list[dict]``: conversational dataset rows for TRL GRPO.
* ``make_reward_func(tasks, monitor=None, num_generations=4)``: a TRL-style reward callable.
* ``RewardMonitor``: counts reward groups with and without variance (E-2).
* ``DeadRewardError(RuntimeError)``: every group so far had identical rewards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from alien.generator import Task
from alien.oracle import oracle_call, render_prompt
from alien.scoring import canonical, parse_calls

MIN_GENERATIONS = 4


class DeadRewardError(RuntimeError):
    """The reward gave no learning signal (E-2); run the rejection-sampling fallback."""


def _check_generations(num_generations: int) -> None:
    if num_generations < MIN_GENERATIONS:
        raise ValueError(
            f"num_generations must be at least {MIN_GENERATIONS} "
            f"(FR-003: GRPO needs at least 4 generations per prompt), got {num_generations}"
        )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    messages = [completion] if isinstance(completion, dict) else completion
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return _content_text(message.get("content"))
    return ""


def exact_match_reward(task: Task, completion: Any) -> float:
    calls = parse_calls(_completion_text(completion))
    if len(calls) == 1 and canonical(calls[0]) == canonical(oracle_call(task)):
        return 1.0
    return 0.0


def to_grpo_rows(tasks: Sequence[Task]) -> list[dict[str, Any]]:
    return [
        {"prompt": [{"role": "user", "content": render_prompt(t)}], "task_id": t.id}
        for t in tasks
    ]


class RewardMonitor:
    """Tracks whether GRPO reward groups ever vary; also holds B-1 diagnostics."""

    def __init__(self, num_generations: int = 4, patience_groups: int = 50) -> None:
        _check_generations(num_generations)
        self.num_generations = num_generations
        self.patience_groups = patience_groups
        self.groups_total = 0
        self.groups_with_variance = 0
        self.unmatched = 0
        self.last_kwargs_keys: list[str] | None = None
        self.last_completion_type: str | None = None

    def observe(self, rewards: Sequence[float]) -> None:
        n = self.num_generations
        for start in range(0, len(rewards) - n + 1, n):
            self.groups_total += 1
            if len(set(rewards[start:start + n])) > 1:
                self.groups_with_variance += 1

    def summary(self) -> dict[str, int]:
        return {
            "groups_total": self.groups_total,
            "groups_with_variance": self.groups_with_variance,
        }

    def is_dead(self) -> bool:
        return self.groups_total >= self.patience_groups and self.groups_with_variance == 0

    def raise_if_dead(self) -> None:
        if self.is_dead():
            raise DeadRewardError(
                f"All {self.groups_total} reward groups had identical rewards, so GRPO has "
                "no learning signal. Stop this run and run the rejection-sampling "
                "fallback (scripts/train_rft.py)."
            )


def _prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return _content_text(message.get("content"))
    return None


def make_reward_func(
    tasks: Sequence[Task],
    monitor: RewardMonitor | None = None,
    num_generations: int = 4,
) -> Callable[..., list[float]]:
    _check_generations(num_generations)
    if monitor is not None and monitor.num_generations != num_generations:
        raise ValueError(
            f"num_generations ({num_generations}) differs from the monitor's "
            f"({monitor.num_generations}); rewards would be grouped wrongly"
        )
    by_id = {t.id: t for t in tasks}
    by_prompt = {render_prompt(t): t for t in reversed(tasks)}

    def reward_fn(
        prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]:
        if monitor is not None:
            monitor.last_kwargs_keys = sorted(kwargs)
            monitor.last_completion_type = type(completions[0]).__name__ if completions else None
        task_ids = kwargs.get("task_id")
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            task = None
            if isinstance(task_ids, (list, tuple)) and i < len(task_ids):
                task = by_id.get(task_ids[i])
            if task is None and i < len(prompts):
                task = by_prompt.get(_prompt_text(prompts[i]))
            if task is None:
                if monitor is not None:
                    monitor.unmatched += 1
                rewards.append(0.0)
            else:
                rewards.append(exact_match_reward(task, completion))
        if monitor is not None:
            monitor.observe(rewards)
        return rewards

    return reward_fn
