"""Scorer for the four Alien Manual metrics with bootstrap CIs (FR-006, FR-007, E-4, E-5).

Public API:

* ``NoEligibleExamplesError(ValueError)``: a metric has no eligible examples (E-5).
* ``canonical(call) -> str``: canonical JSON of ``{"name", "arguments"}``; integral
  floats become ints (5 == 5.0), strings are left alone.
* ``parse_calls(text) -> list[Call]``: calls found in model output; never raises (deeply
  nested JSON counts as malformed; only the first 32768 characters are read).
* ``is_valid_output(task, calls) -> bool``: known tool and schema-valid arguments (an
  integral float is a valid integer, as in ``canonical``).
* ``ExampleScore`` and ``score_example(task, output_text)``: per-example TP/FP/FN.
* ``metrics_from_scores(scores)``: point values of the four metrics.
* ``evaluate(tasks, outputs, ...)``: the Metrics object (value, ci_low, ci_high, n).
* ``paired_gap(scores_a, scores_b, metric, ...)``: bootstrap gap between two models.

Matching follows FR-007: an oracle call matches at most one predicted call by canonical
equality; unmatched predicted calls are FP, unmatched oracle calls are FN. Output with no
parseable call is invalid and scores TP 0, FP 0, FN 1.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from alien.generator import Call, Task

METRIC_NAMES: tuple[str, ...] = (
    "prior_trap_rate", "tool_call_f1", "tool_call_validity", "normal_tool_f1",
)
_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()
# A real reply is at most 256 tokens. Each failed decode costs time in proportion to its
# offset in the text, so a megabyte of "{" would take minutes; only the head is parsed.
_MAX_PARSE_CHARS = 32768


class NoEligibleExamplesError(ValueError):
    """A metric has zero eligible examples; the message names the metric."""


def _normalise(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def canonical(call: Call) -> str:
    body = {"name": call["name"], "arguments": _normalise(call["arguments"])}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_call(obj: Any) -> bool:
    return (isinstance(obj, dict) and isinstance(obj.get("name"), str)
            and isinstance(obj.get("arguments"), dict))


def parse_calls(text: str) -> list[Call]:
    if not isinstance(text, str):
        return []
    text = text[:_MAX_PARSE_CHARS]
    fenced = _FENCE.findall(text)
    calls: list[Call] = []
    for segment in fenced or [text]:
        pos = segment.find("{")
        while pos != -1:
            try:
                obj, end = _DECODER.raw_decode(segment, pos)
            except ValueError:
                pos = segment.find("{", pos + 1)
                continue
            except RecursionError:  # absurdly nested JSON: malformed, drop the rest
                break
            if _is_call(obj):
                call = {"name": obj["name"], "arguments": obj["arguments"]}
                try:
                    canonical(call)  # the scorer compares calls this way; too deep is malformed
                    calls.append(call)
                except RecursionError:
                    pass
            pos = segment.find("{", end)
    return calls


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "integer":
        # Same rule as canonical(): an integral float such as 3.0 is the integer 3.
        return isinstance(_normalise(value), int) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    return False


def is_valid_output(task: Task, calls: list[Call]) -> bool:
    if not calls:
        return False
    known = {task.tool.name, task.tool.call_name}
    params = {p.name: p.type for p in task.tool.params}
    for call in calls:
        args = call["arguments"]
        if call["name"] not in known or set(args) != set(params):
            return False
        if not all(_type_ok(args[name], type_name) for name, type_name in params.items()):
            return False
    return True


@dataclass(frozen=True)
class ExampleScore:
    task_id: str
    is_trap: bool
    valid: bool
    tp: int
    fp: int
    fn: int
    prior_trap: bool


def score_example(task: Task, output_text: str) -> ExampleScore:
    calls = parse_calls(output_text)
    predicted = [canonical(c) for c in calls]
    oracle = canonical(task.oracle_call)
    tp = 1 if oracle in predicted else 0
    habit = canonical(task.habit_call)
    prior_trap = task.is_trap and habit != oracle and habit in predicted
    return ExampleScore(
        task_id=task.id,
        is_trap=task.is_trap,
        valid=is_valid_output(task, calls),
        tp=tp,
        fp=len(predicted) - tp,
        fn=1 - tp,
        prior_trap=prior_trap,
    )


def _pooled_f1(scores: Sequence[ExampleScore], name: str) -> float:
    tp = sum(s.tp for s in scores)
    denominator = 2 * tp + sum(s.fp for s in scores) + sum(s.fn for s in scores)
    if denominator == 0:
        raise NoEligibleExamplesError(name)
    return 2 * tp / denominator


def _metric(scores: Sequence[ExampleScore], name: str) -> float:
    if name == "prior_trap_rate":
        trap = [s for s in scores if s.is_trap]
        if not trap:
            raise NoEligibleExamplesError("prior_trap_rate: no trap examples")
        return sum(s.prior_trap for s in trap) / len(trap)
    if name == "normal_tool_f1":
        normal = [s for s in scores if not s.is_trap]
        if not normal:
            raise NoEligibleExamplesError("normal_tool_f1: no normal examples")
        return _pooled_f1(normal, name)
    if not scores:
        raise NoEligibleExamplesError(f"{name}: no examples")
    if name == "tool_call_validity":
        return sum(s.valid for s in scores) / len(scores)
    if name == "tool_call_f1":
        return _pooled_f1(scores, name)
    raise ValueError(f"unknown metric: {name}")


def _eligible_count(scores: Sequence[ExampleScore], name: str) -> int:
    if name == "prior_trap_rate":
        return sum(s.is_trap for s in scores)
    if name == "normal_tool_f1":
        return sum(not s.is_trap for s in scores)
    return len(scores)


def metrics_from_scores(scores: Sequence[ExampleScore]) -> dict[str, float]:
    return {name: _metric(scores, name) for name in METRIC_NAMES}


def _quantile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _bootstrap(scores: Sequence[ExampleScore], metrics: Sequence[str], n_resamples: int,
               seed: int) -> dict[str, list[float]]:
    """Per-metric resampled values; a resample with no eligible examples is skipped."""
    rng = random.Random(seed)
    n = len(scores)
    values: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_resamples):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        for name in metrics:
            try:
                values[name].append(_metric(sample, name))
            except NoEligibleExamplesError:
                continue
    for name, drawn in values.items():
        if not drawn:
            raise NoEligibleExamplesError(f"{name}: every bootstrap resample was empty")
    return values


def _interval(drawn: list[float], alpha: float) -> tuple[float, float]:
    ordered = sorted(drawn)
    return _quantile(ordered, alpha / 2), _quantile(ordered, 1 - alpha / 2)


def evaluate(tasks: Sequence[Task], outputs: Sequence[str], *, n_resamples: int = 1000,
             seed: int = 0, alpha: float = 0.05) -> dict[str, dict[str, float]]:
    if len(tasks) != len(outputs):
        raise ValueError(f"{len(tasks)} tasks but {len(outputs)} outputs")
    scores = [score_example(t, o) for t, o in zip(tasks, outputs)]
    points = metrics_from_scores(scores)
    drawn = _bootstrap(scores, METRIC_NAMES, n_resamples, seed)
    result = {}
    for name in METRIC_NAMES:
        low, high = _interval(drawn[name], alpha)
        value = points[name]
        result[name] = {
            "value": value,
            "ci_low": min(low, value),
            "ci_high": max(high, value),
            "n": _eligible_count(scores, name),
        }
    return result


def paired_gap(scores_a: Sequence[ExampleScore], scores_b: Sequence[ExampleScore], metric: str,
               *, higher_is_better: bool = True, n_resamples: int = 1000, seed: int = 0,
               alpha: float = 0.05) -> dict[str, float]:
    if [s.task_id for s in scores_a] != [s.task_id for s in scores_b]:
        raise ValueError("scores_a and scores_b are not aligned by task_id")
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric: {metric}")
    sign = 1 if higher_is_better else -1
    gap = sign * (_metric(scores_a, metric) - _metric(scores_b, metric))
    rng = random.Random(seed)
    n = len(scores_a)
    drawn: list[float] = []
    for _ in range(n_resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        try:
            a = _metric([scores_a[i] for i in picks], metric)
            b = _metric([scores_b[i] for i in picks], metric)
        except NoEligibleExamplesError:
            continue
        drawn.append(sign * (a - b))
    if not drawn:
        raise NoEligibleExamplesError(f"{metric}: every bootstrap resample was empty")
    low, high = _interval(drawn, alpha)
    return {"gap": gap, "ci_low": min(low, gap), "ci_high": max(high, gap)}
