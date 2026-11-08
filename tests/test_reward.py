"""Tests for the exact-match GRPO reward and its monitor (FR-003, DEC-2, NG-5, E-2, E-4, AS-3).

Each test group names the production change that would make it fail.
"""

import json

import pytest

from alien.oracle import call_to_json, habit_call, oracle_call, render_prompt
from alien.reward import (
    DeadRewardError,
    RewardMonitor,
    exact_match_reward,
    make_reward_func,
    to_grpo_rows,
)
from alien.splits import make_splits


@pytest.fixture(scope="module")
def tasks():
    splits = make_splits(0)
    return [*splits.train, *splits.dev, *splits.heldout]


@pytest.fixture(scope="module")
def trap(tasks):
    return next(t for t in tasks if t.is_trap)


@pytest.fixture(scope="module")
def other_trap(tasks, trap):
    return next(t for t in tasks if t.is_trap and t.id != trap.id)


def oracle_text(task) -> str:
    return call_to_json(oracle_call(task))


def assistant(content) -> dict:
    return {"role": "assistant", "content": content}


# Breaks caught: scoring partial credit (DEC-2/NG-5), accepting the habit call, or accepting
# anything that is not exactly one canonical-equal call (E-4).
def test_oracle_call_scores_one(trap):
    assert exact_match_reward(trap, oracle_text(trap)) == 1.0


def test_habit_call_on_trap_task_scores_zero(trap):
    assert habit_call(trap) != oracle_call(trap)
    assert exact_match_reward(trap, call_to_json(habit_call(trap))) == 0.0


def test_malformed_text_scores_zero(trap):
    assert exact_match_reward(trap, '{"name": "x", "arguments": {') == 0.0
    assert exact_match_reward(trap, "") == 0.0
    assert exact_match_reward(trap, "I would call the tool") == 0.0


def test_unknown_tool_name_scores_zero(trap):
    bad = {"name": "no_such_tool", "arguments": oracle_call(trap)["arguments"]}
    assert exact_match_reward(trap, json.dumps(bad)) == 0.0


def test_oracle_call_plus_extra_call_scores_zero(trap):
    both = oracle_text(trap) + "\n" + call_to_json(habit_call(trap))
    assert exact_match_reward(trap, both) == 0.0
    twice = oracle_text(trap) + "\n" + oracle_text(trap)
    assert exact_match_reward(trap, twice) == 0.0


def test_fenced_oracle_json_scores_one(trap):
    fenced = "```json\n" + oracle_text(trap) + "\n```"
    assert exact_match_reward(trap, fenced) == 1.0


# Breaks caught: ignoring message-list completions (AS-3 unvalidated), reading the wrong message
# of a conversation, or not handling content parts.
def test_message_list_uses_last_assistant_message(trap):
    messages = [
        assistant(call_to_json(habit_call(trap))),
        {"role": "user", "content": "try again"},
        assistant(oracle_text(trap)),
    ]
    assert exact_match_reward(trap, messages) == 1.0


def test_message_list_with_wrong_last_assistant_scores_zero(trap):
    messages = [assistant(oracle_text(trap)), assistant(call_to_json(habit_call(trap)))]
    assert exact_match_reward(trap, messages) == 0.0


def test_content_parts_shape_scores_one(trap):
    parts = [{"type": "text", "text": oracle_text(trap)}]
    assert exact_match_reward(trap, [assistant(parts)]) == 1.0


def test_single_message_dict_scores_one(trap):
    assert exact_match_reward(trap, assistant(oracle_text(trap))) == 1.0


@pytest.mark.parametrize("odd", [None, 42, {}, [], [None], [{"role": "assistant"}], b"x"])
def test_odd_completion_scores_zero_without_raising(trap, odd):
    assert exact_match_reward(trap, odd) == 0.0


# Breaks caught: misaligned task lookup, prompt-shape-dependent matching, unmatched examples
# scoring nonzero or not being counted, and B-1 diagnostics not being recorded.
def test_reward_func_uses_task_id_kwargs(tasks, trap, other_trap):
    monitor = RewardMonitor(num_generations=4)
    fn = make_reward_func(tasks, monitor)
    completions = [
        [assistant(oracle_text(trap))],
        [assistant(call_to_json(habit_call(trap)))],
        [assistant(oracle_text(other_trap))],
        [assistant("nonsense")],
    ]
    rewards = fn(
        prompts=["ignored"] * 4,
        completions=completions,
        task_id=[trap.id, trap.id, other_trap.id, other_trap.id],
    )
    assert rewards == [1.0, 0.0, 1.0, 0.0]
    assert monitor.unmatched == 0
    assert fn.__name__ == "reward_fn"


def test_reward_func_matches_string_prompts(tasks, trap, other_trap):
    fn = make_reward_func(tasks)
    prompts = [render_prompt(trap), render_prompt(other_trap)]
    completions = [oracle_text(trap), oracle_text(trap)]
    assert fn(prompts, completions) == [1.0, 0.0]


def test_reward_func_matches_message_list_prompts(tasks, trap, other_trap):
    fn = make_reward_func(tasks)
    prompts = [
        [{"role": "user", "content": render_prompt(trap)}],
        [{"role": "user", "content": render_prompt(other_trap)}],
    ]
    completions = [[assistant(oracle_text(trap))], [assistant(oracle_text(other_trap))]]
    assert fn(prompts, completions) == [1.0, 1.0]


def test_unknown_prompt_scores_zero_and_is_counted(tasks, trap):
    monitor = RewardMonitor(num_generations=4)
    fn = make_reward_func(tasks, monitor)
    rewards = fn(["some prompt nobody rendered"], [oracle_text(trap)])
    assert rewards == [0.0]
    assert monitor.unmatched == 1


def test_unknown_task_id_falls_back_to_prompt_then_counts(tasks, trap):
    monitor = RewardMonitor(num_generations=4)
    fn = make_reward_func(tasks, monitor)
    rewards = fn(
        [render_prompt(trap), "nobody"],
        [oracle_text(trap), oracle_text(trap)],
        task_id=["missing-id", "missing-id"],
    )
    assert rewards == [1.0, 0.0]
    assert monitor.unmatched == 1


def test_reward_func_records_kwargs_keys_and_completion_type(tasks, trap):
    monitor = RewardMonitor(num_generations=4)
    fn = make_reward_func(tasks, monitor)
    fn([render_prompt(trap)], [[assistant("x")]], task_id=[trap.id], trainer_state=object())
    assert monitor.last_kwargs_keys == ["task_id", "trainer_state"]
    assert monitor.last_completion_type == "list"
    fn([render_prompt(trap)], ["x"])
    assert monitor.last_kwargs_keys == []
    assert monitor.last_completion_type == "str"


def test_reward_func_feeds_monitor_in_groups(tasks, trap):
    monitor = RewardMonitor(num_generations=4)
    fn = make_reward_func(tasks, monitor)
    texts = [oracle_text(trap), "x", "x", "x", "x", "x", "x", "x"]
    fn([render_prompt(trap)] * 8, texts, task_id=[trap.id] * 8)
    assert monitor.summary() == {"groups_total": 2, "groups_with_variance": 1}


def test_reward_func_rejects_monitor_with_other_group_size(tasks):
    with pytest.raises(ValueError, match="num_generations"):
        make_reward_func(tasks, RewardMonitor(num_generations=8), num_generations=4)


def test_reward_func_rejects_fewer_than_four_generations(tasks):
    with pytest.raises(ValueError, match="FR-003"):
        make_reward_func(tasks, num_generations=2)


# Breaks caught: miscounting variance groups, declaring death too early/late (E-2), a single
# varying group not rescuing the run, partial groups counted, and the FR-003 floor missing.
def test_monitor_counts_variance_groups():
    monitor = RewardMonitor(num_generations=4)
    monitor.observe([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert monitor.summary() == {"groups_total": 3, "groups_with_variance": 1}


def test_monitor_dead_after_patience_all_zero_groups():
    monitor = RewardMonitor(num_generations=4, patience_groups=50)
    monitor.observe([0.0] * (4 * 49))
    assert not monitor.is_dead()
    monitor.raise_if_dead()
    monitor.observe([0.0] * 4)
    assert monitor.is_dead()
    with pytest.raises(DeadRewardError, match="scripts/train_rft.py"):
        monitor.raise_if_dead()


def test_dead_reward_error_is_a_runtime_error():
    assert issubclass(DeadRewardError, RuntimeError)


def test_one_varying_group_among_dead_groups_is_not_dead():
    monitor = RewardMonitor(num_generations=4, patience_groups=50)
    monitor.observe([0.0] * (4 * 25) + [0.0, 1.0, 0.0, 0.0] + [0.0] * (4 * 24))
    assert monitor.summary() == {"groups_total": 50, "groups_with_variance": 1}
    assert not monitor.is_dead()
    monitor.raise_if_dead()


def test_monitor_ignores_trailing_partial_group():
    monitor = RewardMonitor(num_generations=4)
    monitor.observe([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert monitor.summary() == {"groups_total": 1, "groups_with_variance": 0}


def test_monitor_rejects_fewer_than_four_generations():
    with pytest.raises(ValueError, match="FR-003"):
        RewardMonitor(num_generations=2)


# Breaks caught: wrong dataset column names or prompt format that TRL conversational GRPO needs.
def test_to_grpo_rows_shape(tasks):
    rows = to_grpo_rows(tasks[:2])
    assert len(rows) == 2
    assert list(rows[0]) == ["prompt", "task_id"]
    assert rows[0] == {
        "prompt": [{"role": "user", "content": render_prompt(tasks[0])}],
        "task_id": tasks[0].id,
    }
    assert rows[0]["prompt"][0]["role"] == "user"
    assert rows[1]["task_id"] == tasks[1].id
