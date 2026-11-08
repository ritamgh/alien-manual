"""Tests for the Alien Manual generator, oracle and splits (FR-001, FR-004).

Each test group names the production change that would make it fail.
"""

import dataclasses
import hashlib
import json

import pytest

from alien.generator import ALIAS_VERBS, DOMAINS, HABIT_VERBS, TWISTS, Param, Task, Tool
from alien.oracle import (
    FOLLOW_DOCS_LINE,
    habit_call,
    oracle_call,
    render_prompt,
    to_chat_example,
)
from alien.splits import (
    LeakError,
    Splits,
    assert_no_leak,
    make_splits,
    splits_from_json,
    splits_to_json,
)


@pytest.fixture(scope="module")
def splits() -> Splits:
    return make_splits(11)


@pytest.fixture(scope="module")
def twist_splits() -> Splits:
    return make_splits(11, holdout="twist")


def all_tasks(s: Splits) -> list[Task]:
    return [*s.train, *s.dev, *s.heldout]


def tool_names(tasks: list[Task]) -> set[str]:
    names = set()
    for t in tasks:
        names.add(t.tool.name)
        names.add(t.tool.call_name)
    return names


def domains(tasks: list[Task]) -> set[str]:
    return {t.tool.domain for t in tasks}


# Break caught: a generator with fewer than six domains or a changed twist list.
def test_constants_have_expected_literals():
    assert TWISTS == ("flip_args", "invert_op", "odd_unit", "rename")
    assert len(DOMAINS) == 12
    assert {"inventory", "scheduling", "fluids"} <= set(DOMAINS)


# Break caught: nondeterminism (global random, set ordering) or ignoring the seed.
def test_same_seed_is_identical_and_different_seed_differs():
    a = splits_to_json(make_splits(5))
    b = splits_to_json(make_splits(5))
    c = splits_to_json(make_splits(6))
    assert a == b
    assert a != c


# Break caught: a twist implementation that leaves habit == oracle, in any family.
def test_every_twisted_task_has_habit_call_different_from_oracle(splits):
    for seed in (1, 2, 3, 11):
        s = make_splits(seed)
        twisted = [t for t in all_tasks(s) if t.tool.twist is not None]
        assert len(twisted) > 600
        for t in twisted:
            assert t.is_trap is True
            assert habit_call(t) != oracle_call(t), t.id
    seen = {t.tool.twist for t in all_tasks(splits)} - {None}
    assert seen == set(TWISTS)


# Break caught: normal tools whose habit call drifts from the oracle call.
def test_normal_tasks_have_equal_habit_and_oracle(splits):
    normal = [t for t in all_tasks(splits) if t.tool.twist is None]
    assert len(normal) > 150
    for t in normal:
        assert t.is_trap is False
        assert habit_call(t) == oracle_call(t), t.id


# Break caught: flip_args that swaps the wrong things, e.g. changes the item.
def test_flip_args_swaps_src_and_dst_only(splits):
    tasks = [t for t in all_tasks(splits) if t.tool.twist == "flip_args"]
    assert tasks
    for t in tasks:
        _, src, dst = [p.name for p in t.tool.params]
        h, o = habit_call(t), oracle_call(t)
        assert h["name"] == o["name"] == t.tool.name
        assert h["arguments"][src] == o["arguments"][dst]
        assert h["arguments"][dst] == o["arguments"][src]
        assert h["arguments"][src] != h["arguments"][dst]
        assert "ends up in" in t.tool.doc


# Break caught: invert_op that does not negate the signed delta.
def test_invert_op_negates_the_delta(splits):
    tasks = [t for t in all_tasks(splits) if t.tool.twist == "invert_op"]
    assert tasks
    for t in tasks:
        delta = t.tool.params[1].name
        h, o = habit_call(t), oracle_call(t)
        assert h["arguments"][delta] != 0
        assert o["arguments"][delta] == -h["arguments"][delta]
        assert h["arguments"][t.tool.params[0].name] == o["arguments"][t.tool.params[0].name]


# Break caught: odd_unit that applies a factor other than the one in the doc.
def test_odd_unit_scales_value_by_the_documented_factor(splits):
    tasks = [t for t in all_tasks(splits) if t.tool.twist == "odd_unit"]
    assert tasks
    for t in tasks:
        value = t.tool.params[1].name
        h, o = habit_call(t), oracle_call(t)
        assert t.tool.unit_factor in (10, 100, 1000)
        assert o["arguments"][value] == h["arguments"][value] * t.tool.unit_factor
        assert str(t.tool.unit_factor) in t.tool.doc


# Break caught: rename that changes arguments or reuses the habit name.
def test_rename_changes_only_the_tool_name(splits):
    tasks = [t for t in all_tasks(splits) if t.tool.twist == "rename"]
    assert tasks
    for t in tasks:
        h, o = habit_call(t), oracle_call(t)
        assert h["name"] == t.tool.name
        assert o["name"] == t.tool.call_name != t.tool.name
        assert h["arguments"] == o["arguments"]
        assert f"`{t.tool.call_name}`" in t.tool.doc


# Break caught: name collisions, e.g. an alias equal to a habit verb or a reused noun.
def test_names_are_unique_and_alias_verbs_do_not_overlap_habit_verbs(splits):
    habit = {v for verbs in HABIT_VERBS.values() for v in verbs}
    assert habit.isdisjoint(ALIAS_VERBS)
    tools = {t.tool.name: t.tool for t in all_tasks(splits)}
    nouns = [n.split("_", 1)[1] for n in tools]
    assert len(set(nouns)) == len(nouns)
    ids = [t.id for t in all_tasks(splits)]
    assert len(set(ids)) == len(ids)
    assert splits.train[0].id.startswith("train-")


# Break caught: wrong split sizes or dev sharing domains with the other splits (B-2).
def test_split_sizes_and_dev_pool_is_separate(splits):
    assert len(splits.heldout) == 240
    assert len(splits.train) == 600
    assert len(splits.dev) == 100
    dev_d, train_d, held_d = domains(splits.dev), domains(splits.train), domains(splits.heldout)
    assert len(dev_d) == 3 and len(train_d) == 6 and len(held_d) == 3
    assert dev_d.isdisjoint(train_d) and dev_d.isdisjoint(held_d)
    assert train_d.isdisjoint(held_d)
    assert tool_names(splits.dev).isdisjoint(tool_names(splits.train) | tool_names(splits.heldout))


# Break caught: held-out family leaking into train, or held-out lacking traps/normals.
def test_twist_holdout_holds_one_twist_family_and_both_kinds(twist_splits):
    splits = twist_splits
    held = splits.heldout_twist
    assert held in TWISTS
    heldout_twists = {t.tool.twist for t in splits.heldout}
    assert heldout_twists == {held, None}
    assert sum(t.is_trap for t in splits.heldout) == 140
    assert sum(not t.is_trap for t in splits.heldout) == 100
    train_twists = {t.tool.twist for t in splits.train}
    assert held not in train_twists
    assert train_twists == (set(TWISTS) - {held}) | {None}
    assert tool_names(splits.train).isdisjoint(tool_names(splits.heldout))


# Break caught: the held-out family is not varied by seed (always the same family).
def test_heldout_family_varies_with_seed():
    families = {make_splits(seed, holdout="twist").heldout_twist for seed in range(12)}
    assert len(families) >= 3


# Break caught: the default hold-out sharing a tool (name or alias) between train, dev and
# held-out, or changing the 48 / 24 / 24 tool counts.
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_default_holdout_splits_share_no_tool_and_keep_the_sizes(seed):
    s = make_splits(seed)
    assert (len(s.train), len(s.dev), len(s.heldout)) == (600, 100, 240)
    pools = {"train": s.train, "dev": s.dev, "heldout": s.heldout}
    assert {n: len({t.tool.name for t in ts}) for n, ts in pools.items()} == {
        "train": 48, "dev": 24, "heldout": 24}
    for a, b in (("train", "dev"), ("train", "heldout"), ("dev", "heldout")):
        assert tool_names(pools[a]) & tool_names(pools[b]) == set(), (a, b)


# Break caught: a twist family missing from train, dev or held-out under the default hold-out
# (the first run's held-out twist family was never seen in training), or a split with no
# normal tools.
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_default_holdout_has_every_twist_family_in_every_split(seed):
    s = make_splits(seed)
    assert s.heldout_twist is None
    for name in ("train", "dev", "heldout"):
        tasks = getattr(s, name)
        assert {t.tool.twist for t in tasks} == {None, *TWISTS}, name


# Break caught: holdout="twist" drifting from the split of the first run. The hash is of
# splits_to_json(make_splits(seed)) taken before holdout existed; the ids and tool names were
# read off the same code.
def test_twist_holdout_reproduces_the_first_run_split_exactly():
    s0 = make_splits(0, holdout="twist")
    assert s0.heldout_twist == "flip_args"
    assert hashlib.sha256(splits_to_json(s0).encode()).hexdigest() == (
        "1264b480bc290d462cf13aa687c17f29c710ce7c95265b5a322dcfda16e9f29b")
    assert [(t.id, t.tool.name, t.tool.call_name, t.tool.domain, t.tool.twist)
            for t in (s0.train[0], s0.dev[1], s0.heldout[-1])] == [
        ("train-0000", "boost_gornemde", "boost_gornemde", "botany", "invert_op"),
        ("dev-0001", "read_runru", "grab_runru", "logistics", "rename"),
        ("heldout-0239", "carry_bipom", "carry_bipom", "navigation", "flip_args"),
    ]
    s3 = make_splits(3, holdout="twist")
    assert s3.heldout_twist == "invert_op"
    assert hashlib.sha256(splits_to_json(s3).encode()).hexdigest() == (
        "e6cf4af3dde615abfafe1965f812e5d5ecf560b377916fb5f20e746d69d555c7")


# Break caught: a typo such as holdout="twists" silently falling through to a hold-out mode.
def test_unknown_holdout_is_rejected():
    with pytest.raises(ValueError, match="holdout"):
        make_splits(0, holdout="twists")


# Break caught: a JSON round trip that fails on heldout_twist None (default) or loses it (twist).
@pytest.mark.parametrize("holdout, expected", [("tools", None), ("twist", "flip_args")])
def test_json_round_trip_keeps_heldout_twist_for_both_holdouts(holdout, expected):
    s = make_splits(0, holdout=holdout)
    back = splits_from_json(splits_to_json(s))
    assert back.heldout_twist == expected
    assert back == s


def test_assert_no_leak_passes_on_clean_splits(splits):
    assert_no_leak(splits)


# Break caught: assert_no_leak that does not detect a held-out tool in train.
def test_leaky_tool_raises_leak_error(splits):
    leaky = dataclasses.replace(splits, train=[*splits.train, splits.heldout[0]])
    with pytest.raises(LeakError):
        assert_no_leak(leaky)


# Break caught: assert_no_leak that ignores a held-out alias name of a rename tool.
def test_leaky_alias_name_raises_leak_error():
    splits = next(s for s in (make_splits(seed, holdout="twist") for seed in range(30))
                  if s.heldout_twist == "rename")
    train_task = splits.train[0]
    other = next(t for t in splits.heldout if t.tool.twist == "rename")
    clone = dataclasses.replace(train_task.tool, name=other.tool.call_name)
    leaky_task = dataclasses.replace(train_task, tool=clone)
    leaky = dataclasses.replace(splits, train=[*splits.train, leaky_task])
    with pytest.raises(LeakError):
        assert_no_leak(leaky)


# Break caught: assert_no_leak that ignores a held-out domain in train.
def test_leaky_domain_raises_leak_error(splits):
    held_domain = splits.heldout[0].tool.domain
    train_task = splits.train[0]
    clone = dataclasses.replace(train_task.tool, name="zz_fresh", call_name="zz_fresh",
                                domain=held_domain)
    leaky_task = dataclasses.replace(train_task, tool=clone)
    leaky = dataclasses.replace(splits, train=[*splits.train, leaky_task])
    with pytest.raises(LeakError):
        assert_no_leak(leaky)


# Break caught: assert_no_leak that ignores the held-out twist family in train.
def test_leaky_twist_raises_leak_error(twist_splits):
    splits = twist_splits
    train_task = splits.train[0]
    clone = dataclasses.replace(train_task.tool, name="zz_fresh", call_name="zz_fresh",
                                twist=splits.heldout_twist)
    leaky_task = dataclasses.replace(train_task, tool=clone)
    leaky = dataclasses.replace(splits, train=[*splits.train, leaky_task])
    with pytest.raises(LeakError):
        assert_no_leak(leaky)


# Break caught: dev tools or domains shared with held-out (peeking, B-2).
def test_leaky_dev_raises_leak_error(splits):
    leaky = dataclasses.replace(splits, dev=[*splits.dev, splits.heldout[0]])
    with pytest.raises(LeakError):
        assert_no_leak(leaky)


FIXTURE_TOOL = Tool(
    name="move_zorvak",
    domain="inventory",
    archetype="transfer",
    params=(
        Param("item", "string", "The crate to move."),
        Param("src", "string", "A bay identifier."),
        Param("dst", "string", "A bay identifier."),
    ),
    doc="Moves a crate between two bays.\nNote: `src` is where it ends up.\nParameters:\n"
        "- item (string): The crate to move.",
    twist="flip_args",
    call_name="move_zorvak",
    unit_factor=1,
)
FIXTURE_TASK = Task(
    id="fixture-0001",
    tool=FIXTURE_TOOL,
    request="Move the crate C-14 from bay-3 to bay-7.",
    oracle_call={"name": "move_zorvak",
                 "arguments": {"item": "C-14", "src": "bay-7", "dst": "bay-3"}},
    habit_call={"name": "move_zorvak",
                "arguments": {"item": "C-14", "src": "bay-3", "dst": "bay-7"}},
    is_trap=True,
)


# Break caught: any change to the prompt layout (FR-006 needs one fixed layout).
def test_render_prompt_literal_layout():
    expected = (
        "You are an agent that calls tools. Read the documentation of the tool below.\n"
        "\n"
        "### Tool: move_zorvak\n"
        "Moves a crate between two bays.\n"
        "Note: `src` is where it ends up.\n"
        "Parameters:\n"
        "- item (string): The crate to move.\n"
        "\n"
        'Reply with exactly one JSON object of the form {"name": "<tool name>", '
        '"arguments": {"<argument>": <value>}} and nothing else.\n'
        "\n"
        "Request: Move the crate C-14 from bay-3 to bay-7."
    )
    assert render_prompt(FIXTURE_TASK) == expected


# Break caught: the follow-docs variant differing by more or less than one line (FR-006).
def test_follow_docs_adds_exactly_one_line(splits):
    for task in (FIXTURE_TASK, splits.heldout[0], splits.train[5]):
        base = render_prompt(task).split("\n")
        tuned = render_prompt(task, follow_docs=True).split("\n")
        assert len(tuned) == len(base) + 1
        assert FOLLOW_DOCS_LINE in tuned and FOLLOW_DOCS_LINE not in base
        tuned.remove(FOLLOW_DOCS_LINE)
        assert tuned == base
        assert "\n" not in FOLLOW_DOCS_LINE


# Break caught: a training trace that does not end with the oracle call as JSON.
def test_chat_example_ends_with_oracle_call_json():
    ex = to_chat_example(FIXTURE_TASK)
    assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
    assert ex["messages"][0]["content"] == render_prompt(FIXTURE_TASK)
    assert ex["messages"][1]["content"] == (
        '{"name": "move_zorvak", "arguments": {"item": "C-14", "src": "bay-7", "dst": "bay-3"}}'
    )


# Break caught: to_dict/from_dict losing a field (e.g. tuple params, unit_factor).
def test_task_and_splits_json_round_trip(splits):
    d = FIXTURE_TASK.to_dict()
    assert sorted(d) == ["habit_call", "id", "is_trap", "oracle_call", "request", "tool"]
    assert d["tool"]["params"][1] == {"name": "src", "type": "string",
                                      "description": "A bay identifier."}
    assert Task.from_dict(json.loads(json.dumps(d))) == FIXTURE_TASK
    again = splits_from_json(splits_to_json(splits))
    assert again == splits
    assert splits_to_json(again) == splits_to_json(splits)


# Break caught: hardness changing the calls (the E-1 hook must only change docs).
def test_hardness_changes_docs_but_not_calls():
    base, hard = make_splits(4, hardness=0), make_splits(4, hardness=1)
    assert [t.id for t in base.heldout] == [t.id for t in hard.heldout]
    for a, b in zip(base.heldout, hard.heldout):
        assert (a.request, a.oracle_call, a.habit_call) == (b.request, b.oracle_call, b.habit_call)
    assert any(a.tool.doc != b.tool.doc for a, b in zip(base.heldout, hard.heldout))
    twisted = [t for t in hard.heldout if t.is_trap]
    assert all("Note: despite" not in t.tool.doc and "Note: `" not in t.tool.doc
               for t in twisted)
    assert all(habit_call(t) != oracle_call(t) for t in twisted)
    assert_no_leak(hard)
