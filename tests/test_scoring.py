"""Tests for the Alien Manual scorer (FR-006, FR-007, E-4, E-5).

Each test group names the production change that would make it fail.
"""

import copy
import json

import pytest

from alien.generator import Task
from alien.oracle import call_to_json, habit_call, oracle_call
from alien.scoring import (
    ExampleScore,
    NoEligibleExamplesError,
    canonical,
    evaluate,
    is_valid_output,
    metrics_from_scores,
    paired_gap,
    parse_calls,
    score_example,
)
from alien.splits import Splits, make_splits


@pytest.fixture(scope="module")
def splits() -> Splits:
    return make_splits(0)


@pytest.fixture(scope="module")
def pool(splits) -> list[Task]:
    return [*splits.train, *splits.dev, *splits.heldout]


def pick(pool: list[Task], twist: str | None, archetype: str | None = None) -> Task:
    for t in pool:
        if t.tool.twist == twist and (archetype is None or t.tool.archetype == archetype):
            return t
    raise LookupError((twist, archetype))


def dump(call: dict) -> str:
    return call_to_json(call)


def oracle_outputs(tasks: list[Task]) -> list[str]:
    return [dump(oracle_call(t)) for t in tasks]


def habit_outputs(tasks: list[Task]) -> list[str]:
    return [dump(habit_call(t)) for t in tasks]


def wrong_argument(call: dict) -> dict:
    bad = copy.deepcopy(call)
    key = next(iter(bad["arguments"]))
    bad["arguments"][key] = "WRONG"
    return bad


# Break caught: a scorer that gives partial credit for the right tool with a wrong
# argument, or that ignores extra or missing calls (FR-007 fixtures).
def test_fr007_fixture_exact_call_is_f1_one(pool):
    task = pick(pool, None)
    s = score_example(task, dump(task.oracle_call))
    assert (s.tp, s.fp, s.fn, s.valid) == (1, 0, 0, True)
    assert _f1([s]) == 1.0


def test_fr007_fixture_wrong_argument_is_one_fp_one_fn(pool):
    task = pick(pool, None)
    s = score_example(task, dump(wrong_argument(task.oracle_call)))
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert _f1([s]) == 0.0


def test_fr007_fixture_no_call_is_one_fn(pool):
    s = score_example(pick(pool, None), "I do not know.")
    assert (s.tp, s.fp, s.fn, s.valid) == (0, 0, 1, False)
    assert _f1([s]) == 0.0


def test_fr007_fixture_exact_plus_extra_call_is_two_thirds(pool):
    task = pick(pool, None)
    extra = wrong_argument(task.oracle_call)
    s = score_example(task, f"{dump(task.oracle_call)}\n{dump(extra)}")
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)
    assert round(_f1([s]), 3) == 0.667


def test_duplicate_correct_call_matches_oracle_only_once(pool):
    task = pick(pool, None)
    s = score_example(task, f"{dump(task.oracle_call)} {dump(task.oracle_call)}")
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def _f1(scores: list[ExampleScore]) -> float:
    """Pooled f1 through the real scorer; a trap twin keeps both populations non-empty."""
    twins = [ExampleScore(s.task_id + "-twin", not s.is_trap, s.valid, s.tp, s.fp, s.fn,
                          False) for s in scores]
    return metrics_from_scores([*scores, *twins])["tool_call_f1"]


# Break caught: counting a habit-name call as correct on a rename task, or flagging
# a prior-trap only for one twist kind.
def test_rename_task_with_habit_name_is_prior_trap(pool):
    task = pick(pool, "rename")
    s = score_example(task, dump(task.habit_call))
    assert s.prior_trap is True
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


@pytest.mark.parametrize("twist,archetype", [("flip_args", "transfer"), ("odd_unit", "set"),
                                              ("invert_op", "adjust")])
def test_argument_twist_with_habit_arguments_is_prior_trap(pool, twist, archetype):
    task = pick(pool, twist, archetype)
    s = score_example(task, dump(task.habit_call))
    assert s.prior_trap is True
    assert _f1([s]) == 0.0


def test_oracle_output_on_trap_task_is_not_prior_trap(pool):
    task = pick(pool, "rename")
    assert score_example(task, dump(task.oracle_call)).prior_trap is False


def test_habit_output_on_normal_task_is_never_prior_trap(pool):
    task = pick(pool, None)
    s = score_example(task, dump(task.habit_call))
    assert s.prior_trap is False
    assert s.tp == 1


# Break caught: parse_calls raising on garbage (E-4), or failing on fenced / multiple JSON.
def test_malformed_json_is_invalid_and_run_continues(pool):
    task = pick(pool, None)
    s = score_example(task, '{"name": "move_x", "arguments": {"a": ')
    assert (s.valid, s.tp, s.fp, s.fn, s.prior_trap) == (False, 0, 0, 1, False)


def test_parse_calls_never_raises_and_returns_empty():
    for text in ["", "   ", "null", "[1, 2]", '{"name": 3, "arguments": {}}',
                 '{"name": "t", "arguments": []}', "{{{", "```json\n```"]:
        assert parse_calls(text) == []


def test_parse_single_array_multiple_fenced_and_prose():
    one = {"name": "t", "arguments": {"a": 1}}
    two = {"name": "u", "arguments": {"b": "x"}}
    assert parse_calls(json.dumps(one)) == [one]
    assert parse_calls(json.dumps([one, two])) == [one, two]
    assert parse_calls(json.dumps(one) + "\n\n" + json.dumps(two)) == [one, two]
    assert parse_calls("```json\n" + json.dumps(one) + "\n```") == [one]
    assert parse_calls("Sure! Here you go:\n```json\n" + json.dumps(one)
                       + "\n```\nHope that helps {not json}.") == [one]
    assert parse_calls("I will call " + json.dumps(two) + " now.") == [two]


def test_prose_json_outside_fence_is_ignored_when_fence_present():
    one = {"name": "t", "arguments": {}}
    decoy = {"name": "decoy", "arguments": {}}
    text = json.dumps(decoy) + "\n```json\n" + json.dumps(one) + "\n```"
    assert parse_calls(text) == [one]


# Break caught: treating 5 and 5.0 as different, or 1 and True as the same.
def test_canonical_normalises_integral_floats_but_not_strings():
    a = {"name": "t", "arguments": {"n": 5, "s": "5"}}
    b = {"arguments": {"s": "5", "n": 5.0}, "name": "t"}
    assert canonical(a) == canonical(b) == '{"arguments":{"n":5,"s":"5"},"name":"t"}'
    assert canonical({"name": "t", "arguments": {"s": "5"}}) != canonical(
        {"name": "t", "arguments": {"s": "5.0"}})
    assert canonical({"name": "t", "arguments": {"n": True}}) != canonical(
        {"name": "t", "arguments": {"n": 1}})
    assert canonical({"name": "é", "arguments": {}}) == '{"arguments":{},"name":"é"}'


def test_float_five_matches_integer_oracle_argument(pool):
    task = pick(pool, "odd_unit", "set")
    call = oracle_call(task)
    key = list(call["arguments"])[1]
    text = dump(call).replace(f'"{key}": {call["arguments"][key]}',
                              f'"{key}": {call["arguments"][key]}.0')
    assert ".0" in text
    assert score_example(task, text).tp == 1


def test_bool_and_float_are_not_valid_integers(pool):
    task = pick(pool, None, "set")
    key = task.tool.params[1].name
    for bad in (True, 5.5, float("inf"), float("nan"), "5"):
        call = oracle_call(task)
        call["arguments"][key] = bad
        assert is_valid_output(task, [call]) is False
    assert is_valid_output(task, [oracle_call(task)]) is True


# Break caught (QA F-2): validity rejecting an integral float (110.0) that canonical() already
# treats as the integer 110, so an exactly-right call scored tp=1 but valid=False.
def test_f2_integral_float_argument_is_valid_and_correct_together(pool):
    task = pick(pool, "odd_unit", "set")
    call = oracle_call(task)
    key = task.tool.params[1].name
    assert isinstance(call["arguments"][key], int)
    text = dump(call).replace(f'"{key}": {call["arguments"][key]}',
                              f'"{key}": {call["arguments"][key]}.0')
    assert f'"{key}": {call["arguments"][key]}.0' in text
    s = score_example(task, text)
    assert (s.tp, s.fp, s.fn, s.valid) == (1, 0, 0, True)


def test_f2_integral_float_in_exponent_form_is_valid_but_fractional_is_not(pool):
    task = pick(pool, None, "set")
    key = task.tool.params[1].name
    text = '{"name": "%s", "arguments": {"%s": "x", "%s": 1e2}}'
    first = task.tool.params[0].name
    assert first != key
    exact = text % (task.tool.name, first, key)
    assert is_valid_output(task, parse_calls(exact)) is True
    fractional = exact.replace("1e2", "100.5")
    assert is_valid_output(task, parse_calls(fractional)) is False


def test_validity_rejects_unknown_tool_extra_missing_args_and_no_calls(pool):
    task = pick(pool, None, "set")
    call = oracle_call(task)
    assert is_valid_output(task, []) is False
    assert is_valid_output(task, [{**call, "name": "ghost_tool"}]) is False
    extra = copy.deepcopy(call)
    extra["arguments"]["bonus"] = "x"
    assert is_valid_output(task, [extra]) is False
    missing = copy.deepcopy(call)
    del missing["arguments"][task.tool.params[1].name]
    assert is_valid_output(task, [missing]) is False
    assert is_valid_output(task, [call, {**call, "name": "ghost_tool"}]) is False


def test_rename_task_accepts_both_names_as_valid(pool):
    task = pick(pool, "rename")
    assert is_valid_output(task, [task.habit_call]) is True
    assert is_valid_output(task, [task.oracle_call]) is True


# Break caught: returning 0 or NaN for an empty metric instead of aborting (E-5).
def test_zero_trap_tasks_raises_naming_prior_trap_rate(splits):
    normal = [t for t in splits.heldout if not t.is_trap][:5]
    with pytest.raises(NoEligibleExamplesError, match="prior_trap_rate"):
        evaluate(normal, oracle_outputs(normal))


def test_zero_normal_tasks_raises_naming_normal_tool_f1(splits):
    trap = [t for t in splits.heldout if t.is_trap][:5]
    with pytest.raises(NoEligibleExamplesError, match="normal_tool_f1"):
        evaluate(trap, oracle_outputs(trap))


def test_zero_examples_raises_and_is_a_value_error():
    with pytest.raises(ValueError):
        metrics_from_scores([])
    assert issubclass(NoEligibleExamplesError, ValueError)


def test_f1_with_zero_denominator_raises_naming_metric():
    only_valid_nothing = [ExampleScore("a", False, True, 0, 0, 0, False)]
    with pytest.raises(NoEligibleExamplesError, match="tool_call_f1"):
        metrics_from_scores(only_valid_nothing + [ExampleScore("b", True, True, 0, 0, 0, False)])


def test_evaluate_requires_aligned_lists(splits):
    with pytest.raises(ValueError):
        evaluate(splits.heldout[:3], ["{}"] * 2)


# Break caught: wrong metric populations, or CI shape not matching the contract.
def test_evaluate_oracle_outputs_on_heldout_gives_perfect_scores(splits):
    tasks = splits.heldout
    result = evaluate(tasks, oracle_outputs(tasks))
    n_trap = sum(t.is_trap for t in tasks)
    assert list(result) == ["prior_trap_rate", "tool_call_f1", "tool_call_validity",
                            "normal_tool_f1"]
    assert result["prior_trap_rate"] == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                                         "n": n_trap}
    for name in ("tool_call_f1", "tool_call_validity", "normal_tool_f1"):
        assert result[name]["value"] == 1.0
        assert result[name]["ci_low"] == result[name]["ci_high"] == 1.0
    assert result["tool_call_f1"]["n"] == result["tool_call_validity"]["n"] == len(tasks)
    assert result["normal_tool_f1"]["n"] == len(tasks) - n_trap
    assert 0 < n_trap < len(tasks)


def test_evaluate_habit_outputs_give_prior_trap_rate_one(splits):
    tasks = splits.heldout
    result = evaluate(tasks, habit_outputs(tasks))
    assert result["prior_trap_rate"]["value"] == 1.0
    assert result["normal_tool_f1"]["value"] == 1.0
    assert 0.0 < result["tool_call_f1"]["value"] < 1.0


def mixed_outputs(tasks: list[Task]) -> list[str]:
    half = len(tasks) // 2
    return oracle_outputs(tasks[:half]) + habit_outputs(tasks[half:])


def test_mixed_outputs_have_interior_ci_containing_the_value(splits):
    tasks = splits.heldout
    result = evaluate(tasks, mixed_outputs(tasks))
    m = result["prior_trap_rate"]
    assert 0.0 < m["ci_low"] <= m["value"] <= m["ci_high"] < 1.0
    assert m["value"] == pytest.approx(0.5, abs=0.15)
    f1 = result["tool_call_f1"]
    assert 0.0 < f1["ci_low"] <= f1["value"] <= f1["ci_high"] < 1.0


# Break caught: bootstrap that ignores the seed or uses global random state.
def test_same_seed_same_ci_and_different_seed_differs(splits):
    tasks = splits.heldout
    outs = mixed_outputs(tasks)
    a = evaluate(tasks, outs, seed=3)
    b = evaluate(tasks, outs, seed=3)
    c = evaluate(tasks, outs, seed=4)
    assert a == b
    assert a["prior_trap_rate"]["value"] == c["prior_trap_rate"]["value"]
    assert (a["prior_trap_rate"]["ci_low"], a["prior_trap_rate"]["ci_high"]) != (
        c["prior_trap_rate"]["ci_low"], c["prior_trap_rate"]["ci_high"])


# Break caught: gap with the wrong sign, or unpaired resampling.
def test_paired_gap_oracle_vs_habit_on_prior_trap_rate(splits):
    tasks = splits.heldout
    tuned = [score_example(t, o) for t, o in zip(tasks, oracle_outputs(tasks))]
    base = [score_example(t, o) for t, o in zip(tasks, habit_outputs(tasks))]
    gap = paired_gap(tuned, base, "prior_trap_rate", higher_is_better=False)
    assert gap["gap"] == 1.0
    assert gap["ci_low"] > 0.9
    assert gap["ci_low"] <= gap["gap"] <= gap["ci_high"]
    worse = paired_gap(base, tuned, "prior_trap_rate", higher_is_better=False)
    assert worse["gap"] == -1.0
    assert worse["ci_high"] < 0


def test_paired_gap_higher_is_better_direction(splits):
    tasks = splits.heldout
    tuned = [score_example(t, o) for t, o in zip(tasks, oracle_outputs(tasks))]
    base = [score_example(t, o) for t, o in zip(tasks, habit_outputs(tasks))]
    gap = paired_gap(tuned, base, "tool_call_f1")
    assert gap["gap"] > 0.0 and gap["ci_low"] > 0.0


def test_paired_gap_rejects_misaligned_scores_and_unknown_metric(splits):
    tasks = splits.heldout[:10]
    scores = [score_example(t, o) for t, o in zip(tasks, oracle_outputs(tasks))]
    with pytest.raises(ValueError, match="aligned"):
        paired_gap(scores, list(reversed(scores)), "tool_call_f1")
    with pytest.raises(ValueError, match="aligned"):
        paired_gap(scores, scores[:-1], "tool_call_f1")
    with pytest.raises(ValueError, match="unknown metric"):
        paired_gap(scores, scores, "accuracy")


# Break caught: resampling each list independently; identical models must give a zero-width CI.
def test_paired_gap_of_identical_mixed_models_is_exactly_zero(splits):
    tasks = splits.heldout
    scores = [score_example(t, o) for t, o in zip(tasks, mixed_outputs(tasks))]
    gap = paired_gap(scores, scores, "prior_trap_rate", higher_is_better=False)
    assert gap == {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0}


# Break caught (QA F-1): RecursionError escaping parse_calls on deeply nested JSON.
@pytest.mark.parametrize("unit", ['{"a":', '{"a":[', '[{"a":'])
def test_f1_deeply_nested_json_parses_to_no_calls_and_scores_zero(pool, unit):
    task = pick(pool, None)
    deep = unit * 20000
    assert len(deep) >= 100000
    assert parse_calls(deep) == []
    s = score_example(task, deep)
    assert (s.tp, s.fp, s.fn, s.valid, s.prior_trap) == (0, 0, 1, False, False)


def test_f1_call_before_a_deeply_nested_blob_is_still_scored(pool):
    task = pick(pool, None)
    s = score_example(task, dump(task.oracle_call) + "\n" + '{"a":' * 20000)
    assert (s.tp, s.fp, s.fn, s.valid) == (1, 0, 0, True)


def test_f1_evaluate_with_a_deeply_nested_output_scores_that_example_zero(pool):
    normal = pick(pool, None)
    trap = pick(pool, "rename")
    result = evaluate([normal, trap], [dump(normal.oracle_call), '{"a":' * 20000],
                      n_resamples=20)
    assert result["tool_call_validity"]["value"] == 0.5
    assert result["normal_tool_f1"]["value"] == 1.0
    assert result["prior_trap_rate"]["value"] == 0.0
    assert round(result["tool_call_f1"]["value"], 3) == 0.667


@pytest.mark.parametrize("filler", ['{"a":', '{"a":[', "{", "[", '{"a":1}', "```", "x"])
def test_f1_one_megabyte_output_returns_quickly_without_raising(pool, filler):
    import time

    task = pick(pool, None)
    big = filler * (1_000_000 // len(filler))
    start = time.monotonic()
    s = score_example(task, big)
    assert time.monotonic() - start < 5
    assert (s.tp, s.valid) == (0, False)


def test_f1_call_whose_argument_is_nested_too_deep_to_compare_is_malformed(pool):
    # raw_decode accepts this depth; canonical() would hit the recursion limit on it.
    task = pick(pool, None)
    deep = '{"name": "%s", "arguments": {"x": %s%s}}' % (
        task.tool.name, "[" * 3000, "]" * 3000)
    assert parse_calls(deep) == []
    s = score_example(task, deep)
    assert (s.tp, s.fp, s.fn, s.valid) == (0, 0, 1, False)
