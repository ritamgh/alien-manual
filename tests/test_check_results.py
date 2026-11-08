"""Tests for scripts/check_results.py, the results-file gate (FR-006, AS-1..3, E-2, E-5).

Each rule has a mutation of a valid document that must produce an error naming the field.
A validator that accepted a bad value (or named the wrong field) would fail the matching case.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_results
from check_results import validate, validate_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DELETE = object()


def metric(value: float, low: float, high: float, n: int = 200) -> dict:
    return {"value": value, "ci_low": low, "ci_high": high, "n": n}


def metrics(scale: float = 0.0) -> dict:
    """A valid Metrics object; `scale` shifts the values so models differ."""
    return {
        "prior_trap_rate": metric(0.60 - scale, 0.52 - scale, 0.68 - scale, 140),
        "tool_call_f1": metric(0.50 + scale, 0.44 + scale, 0.56 + scale),
        "tool_call_validity": metric(0.90, 0.86, 0.94),
        "normal_tool_f1": metric(0.80, 0.72, 0.88, 100),
    }


def envelope(kind: str) -> dict:
    return {
        "kind": kind,
        "schema_version": 1,
        "created_at": "2026-11-07T18:30:00Z",
        "git_sha": "4d15b5d",
        "seed": 7,
    }


def models4() -> dict:
    return {
        "base": metrics(0.0),
        "prompt_only": metrics(0.05),
        "sft": metrics(0.20),
        "tuned": metrics(0.30),
    }


def gap(value: float, low: float, high: float) -> dict:
    return {"gap": value, "ci_low": low, "ci_high": high}


def base_probe() -> dict:
    return {
        **envelope("base_probe"),
        "n_examples": 100,
        "split": "dev",
        "model": "CohereLabs/tiny-aya-global",
        "model_revision": "abc123",
        "metrics": metrics(),
        "prior_trap_rate": 0.60,
        "headroom": "ok",
        "trap_example": {
            "task_id": "dev-0042",
            "request": "Send 5 units to the depot",
            "model_output": '{"name": "ship", "arguments": {"qty": 5}}',
            "habit_call": {"name": "ship", "arguments": {"qty": 5}},
            "oracle_call": {"name": "ship", "arguments": {"qty": 50}},
        },
    }


def eval_baselines() -> dict:
    return {
        **envelope("eval_baselines"),
        "n_examples": 200,
        "split": "heldout",
        "model_revision": "abc123",
        "decoding": {"max_new_tokens": 256, "temperature": 0.0},
        "models": {"base": metrics(0.0), "prompt_only": metrics(0.05)},
        "gpu_seconds": 5400.5,
        "within_budget": True,
    }


def sft() -> dict:
    return {
        **envelope("sft"),
        "model_revision": "abc123",
        "checkpoint": {"hub_repo": "someone/alien-sft", "revision": None},
        "loss_first": 2.31,
        "loss_last": 0.42,
        "steps": 300,
        "models": {"base": metrics(0.0), "prompt_only": metrics(0.05), "sft": metrics(0.20)},
    }


def grpo() -> dict:
    return {
        **envelope("grpo"),
        "path": "grpo",
        "checkpoint": {"hub_repo": "someone/alien-grpo", "revision": "9f8e7d"},
        "generations_per_prompt": 4,
        "reward_signal": {"groups_total": 64, "groups_with_variance": 40},
        "models": models4(),
    }


def rft() -> dict:
    doc = grpo()
    doc.update(envelope("rft"))
    doc["path"] = "rft"
    doc["reward_signal"] = {"groups_total": 64, "groups_with_variance": 0}
    return doc


def spike() -> dict:
    return {
        **envelope("spike"),
        "install_ok": True,
        "sft_steps": 5,
        "fp16_nan_loss": False,
        "grpo_step_ok": True,
        "completions_0": '[{"role": "assistant", "content": "..."}]',
        "kwargs_keys": ["prompts", "completions", "completion_ids"],
        "reward_sees_full_conversation": True,
        "python_version": "3.12.4",
        "package_versions": {"torch": "2.11.0", "trl": "1.7.1"},
    }


def final() -> dict:
    return {
        **envelope("final"),
        "models": models4(),
        "tuned_path": "grpo",
        "sc001": {
            "verdict": "met",
            "gaps": {
                "vs_base": {
                    "prior_trap_rate": gap(0.30, 0.20, 0.40),
                    "tool_call_f1": gap(0.30, 0.22, 0.38),
                },
                "vs_prompt_only": {
                    "prior_trap_rate": gap(0.25, 0.15, 0.35),
                    "tool_call_f1": gap(0.25, 0.17, 0.33),
                },
            },
        },
        "report_path": "report.md",
    }


BUILDERS = {
    "base_probe": base_probe,
    "eval_baselines": eval_baselines,
    "sft": sft,
    "grpo": grpo,
    "rft": rft,
    "spike": spike,
    "final": final,
}


def mutated(kind: str, path: str, value) -> dict:
    """Return a valid `kind` document with the dotted `path` replaced (or deleted)."""
    doc = copy.deepcopy(BUILDERS[kind]())
    *parents, last = path.split(".")
    node = doc
    for key in parents:
        node = node[key]
    if value is DELETE:
        del node[last]
    else:
        node[last] = value
    return doc


# ---------------------------------------------------------------- valid documents


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_valid_document_has_no_errors(kind):
    assert validate(BUILDERS[kind]()) == []


def test_final_with_failed_threshold_is_valid_when_verdict_says_not_met():
    doc = final()
    doc["sc001"]["gaps"]["vs_base"]["prior_trap_rate"] = gap(0.08, 0.02, 0.14)
    doc["sc001"]["verdict"] = "not_met"
    assert validate(doc) == []


def test_low_headroom_probe_is_valid_when_marked_low():
    doc = base_probe()
    doc["metrics"]["prior_trap_rate"] = metric(0.04, 0.01, 0.08, 140)
    doc["prior_trap_rate"] = 0.04
    doc["headroom"] = "low"
    assert validate(doc) == []


def test_prior_trap_rate_exactly_at_threshold_is_ok_headroom():
    doc = base_probe()
    doc["metrics"]["prior_trap_rate"] = metric(0.10, 0.05, 0.15, 140)
    doc["prior_trap_rate"] = 0.10
    assert validate(doc) == []


def test_rft_fallback_with_zero_variance_groups_is_valid():
    doc = grpo()
    doc["path"] = "rft"
    doc["reward_signal"] = {"groups_total": 64, "groups_with_variance": 0}
    assert validate(doc) == []


# ---------------------------------------------------------------- envelope and kind


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("schema_version", 2, "schema_version: expected 1, got 2"),
        ("schema_version", True, "schema_version: expected 1"),
        ("schema_version", DELETE, "schema_version: missing"),
        ("created_at", "yesterday", "created_at: expected an ISO8601"),
        ("created_at", 20261107, "created_at: expected an ISO8601"),
        ("created_at", DELETE, "created_at: missing"),
        ("git_sha", 1234, "git_sha: expected a string or null"),
        ("seed", "7", "seed: expected an integer"),
        ("seed", 7.5, "seed: expected an integer"),
        ("seed", True, "seed: expected an integer"),
    ],
)
def test_envelope_rules(path, value, expected):
    errors = validate(mutated("sft", path, value))
    assert any(e.startswith(expected) for e in errors), errors


def test_git_sha_null_and_offset_timestamp_are_accepted():
    doc = sft()
    doc["git_sha"] = None
    doc["created_at"] = "2026-11-07T18:30:00+00:00"
    assert validate(doc) == []


@pytest.mark.parametrize("kind_value", ["bogus", None, 3])
def test_unknown_kind_names_the_allowed_kinds(kind_value):
    doc = sft()
    doc["kind"] = kind_value
    errors = validate(doc)
    assert len(errors) == 1
    assert errors[0].startswith("kind: expected one of ")
    for allowed in BUILDERS:
        assert allowed in errors[0]


def test_missing_kind_names_the_allowed_kinds():
    doc = sft()
    del doc["kind"]
    errors = validate(doc)
    assert errors[0].startswith("kind: expected one of base_probe, eval_baselines")


def test_non_object_document_is_one_error():
    assert validate([1, 2]) == ["$: expected a JSON object, got list"]


# ---------------------------------------------------------------- metric rules


@pytest.mark.parametrize(
    "path,value,expected",
    [
        (
            "models.base.tool_call_f1.value",
            1.4,
            "models.base.tool_call_f1.value: expected a number in [0, 1], got 1.4",
        ),
        (
            "models.base.tool_call_f1.value",
            -0.1,
            "models.base.tool_call_f1.value: expected a number in [0, 1]",
        ),
        (
            "models.base.tool_call_f1.value",
            "0.5",
            "models.base.tool_call_f1.value: expected a number in [0, 1]",
        ),
        (
            "models.base.tool_call_f1.ci_low",
            0.9,
            "models.base.tool_call_f1: expected ci_low <= value <= ci_high",
        ),
        (
            "models.base.tool_call_f1.ci_high",
            0.2,
            "models.base.tool_call_f1: expected ci_low <= value <= ci_high",
        ),
        (
            "models.base.tool_call_f1.ci_high",
            1.2,
            "models.base.tool_call_f1.ci_high: expected a number in [0, 1]",
        ),
        ("models.base.tool_call_f1.n", 0, "models.base.tool_call_f1.n: expected an integer >= 1"),
        (
            "models.base.tool_call_f1.n",
            2.5,
            "models.base.tool_call_f1.n: expected an integer >= 1",
        ),
        (
            "models.base.tool_call_f1.n",
            True,
            "models.base.tool_call_f1.n: expected an integer >= 1",
        ),
        ("models.base.tool_call_f1.value", DELETE, "models.base.tool_call_f1.value: missing"),
        ("models.base.tool_call_f1", "high", "models.base.tool_call_f1: expected an object"),
        ("models.base.prior_trap_rate", DELETE, "models.base.prior_trap_rate: missing"),
        (
            "models.base.extra_metric",
            metric(0.5, 0.4, 0.6),
            "models.base: unexpected key 'extra_metric'",
        ),
        ("models.base", DELETE, "models.base: missing"),
        ("models.prompt_only", DELETE, "models.prompt_only: missing"),
    ],
)
def test_metric_rules(path, value, expected):
    errors = validate(mutated("eval_baselines", path, value))
    assert any(e.startswith(expected) for e in errors), errors


@pytest.mark.parametrize("bad", [float("nan"), None])
def test_nan_or_null_metric_names_zero_eligible_examples(bad):
    errors = validate(mutated("eval_baselines", "models.base.prior_trap_rate.value", bad))
    assert any(
        e.startswith("models.base.prior_trap_rate.value:") and "zero eligible examples" in e
        and "E-5" in e
        for e in errors
    ), errors


def test_nan_in_json_file_is_reported_not_crashed(tmp_path):
    path = tmp_path / "eval.json"
    text = json.dumps(eval_baselines()).replace('"value": 0.6,', '"value": NaN,', 1)
    assert "NaN" in text
    path.write_text(text)
    errors = validate_file(path)
    assert any("zero eligible examples" in e for e in errors), errors


# ---------------------------------------------------------------- base_probe


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("n_examples", 99, "n_examples: expected 100, got 99"),
        ("split", "heldout", "split: expected 'dev'"),
        ("model", "", "model: expected a non-empty string"),
        ("model_revision", DELETE, "model_revision: missing"),
        (
            "prior_trap_rate",
            0.55,
            "prior_trap_rate: expected 0.6 (metrics.prior_trap_rate.value), got 0.55",
        ),
        ("headroom", "low", "headroom: expected 'ok'"),
        ("headroom", "maybe", "headroom: expected 'ok'"),
        ("trap_example.task_id", "", "trap_example.task_id: expected a non-empty string"),
        ("trap_example.request", DELETE, "trap_example.request: missing"),
        (
            "trap_example.model_output",
            "  ",
            "trap_example.model_output: expected a non-empty string",
        ),
        (
            "trap_example.habit_call",
            {},
            "trap_example.habit_call: expected a non-empty string or object",
        ),
        (
            "trap_example.oracle_call",
            None,
            "trap_example.oracle_call: expected a non-empty string or object",
        ),
        ("trap_example", DELETE, "trap_example: missing"),
        ("metrics", DELETE, "metrics: missing"),
    ],
)
def test_base_probe_rules(path, value, expected):
    errors = validate(mutated("base_probe", path, value))
    assert any(e.startswith(expected) for e in errors), errors


def test_base_probe_headroom_ok_is_wrong_below_threshold():
    doc = base_probe()
    doc["metrics"]["prior_trap_rate"] = metric(0.09, 0.04, 0.14, 140)
    doc["prior_trap_rate"] = 0.09
    errors = validate(doc)
    assert any(e.startswith("headroom: expected 'low'") for e in errors), errors


# ---------------------------------------------------------------- eval_baselines


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("n_examples", 100, "n_examples: expected 200, got 100"),
        ("split", "dev", "split: expected 'heldout'"),
        (
            "decoding.max_new_tokens",
            512,
            "decoding.max_new_tokens: expected an integer <= 256, got 512",
        ),
        ("decoding.max_new_tokens", DELETE, "decoding.max_new_tokens: missing"),
        ("decoding.temperature", "cold", "decoding.temperature: expected a number"),
        ("decoding", DELETE, "decoding: missing"),
        ("gpu_seconds", 11000.0, "within_budget: expected false (gpu_seconds 11000.0 >= 10800)"),
        ("gpu_seconds", 10800.0, "within_budget: expected false (gpu_seconds 10800.0 >= 10800)"),
        ("gpu_seconds", -1.0, "gpu_seconds: expected a number >= 0"),
        ("gpu_seconds", "fast", "gpu_seconds: expected a number >= 0"),
        ("within_budget", "yes", "within_budget: expected a boolean"),
    ],
)
def test_eval_baselines_rules(path, value, expected):
    errors = validate(mutated("eval_baselines", path, value))
    assert any(e.startswith(expected) for e in errors), errors


def test_over_budget_run_is_valid_when_it_says_so():
    doc = eval_baselines()
    doc["gpu_seconds"] = 12000.0
    doc["within_budget"] = False
    assert validate(doc) == []


def test_eval_baselines_claiming_budget_met_when_not_is_caught():
    doc = eval_baselines()
    doc["gpu_seconds"] = 12000.0
    errors = validate(doc)
    assert any(e.startswith("within_budget: expected false") for e in errors), errors


# ---------------------------------------------------------------- sft


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("loss_last", 2.31, "loss_last: expected a number < loss_first (2.31), got 2.31"),
        ("loss_last", 3.0, "loss_last: expected a number < loss_first (2.31), got 3.0"),
        ("loss_first", "high", "loss_first: expected a number"),
        ("steps", 0, "steps: expected an integer >= 1"),
        ("checkpoint.hub_repo", "", "checkpoint.hub_repo: expected a non-empty string"),
        ("checkpoint.revision", 12, "checkpoint.revision: expected a string or null"),
        ("checkpoint", DELETE, "checkpoint: missing"),
        ("model_revision", "", "model_revision: expected a non-empty string"),
        ("models.sft", DELETE, "models.sft: missing"),
    ],
)
def test_sft_rules(path, value, expected):
    errors = validate(mutated("sft", path, value))
    assert any(e.startswith(expected) for e in errors), errors


# ---------------------------------------------------------------- grpo and rft


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("path", "ppo", "path: expected one of 'grpo', 'rft'"),
        (
            "generations_per_prompt",
            2,
            "generations_per_prompt: expected an integer >= 4 for path grpo (FR-003), got 2",
        ),
        ("generations_per_prompt", "4", "generations_per_prompt: expected an integer"),
        (
            "reward_signal.groups_with_variance",
            0,
            "reward_signal.groups_with_variance: expected >= 1 for path grpo",
        ),
        (
            "reward_signal.groups_with_variance",
            65,
            "reward_signal.groups_with_variance: expected an integer between 0 and "
            "groups_total (64)",
        ),
        ("reward_signal.groups_total", -1, "reward_signal.groups_total: expected an integer >= 0"),
        ("reward_signal", DELETE, "reward_signal: missing"),
        ("checkpoint.hub_repo", "", "checkpoint.hub_repo: expected a non-empty string"),
        ("checkpoint.hub_repo", None, "checkpoint.hub_repo: expected a non-empty string"),
        ("models.tuned", DELETE, "models.tuned: missing"),
        ("models.sft", DELETE, "models.sft: missing"),
    ],
)
def test_grpo_rules(path, value, expected):
    errors = validate(mutated("grpo", path, value))
    assert any(e.startswith(expected) for e in errors), errors


def test_grpo_zero_variance_error_points_at_the_rft_fallback():
    errors = validate(mutated("grpo", "reward_signal.groups_with_variance", 0))
    assert any("path rft" in e and "E-2" in e for e in errors), errors


def test_rft_kind_must_record_path_rft():
    errors = validate(mutated("rft", "path", "grpo"))
    assert any(e.startswith("path: expected 'rft' for a results/rft.json") for e in errors), errors


def test_rft_does_not_need_four_generations_or_reward_variance():
    doc = rft()
    doc["generations_per_prompt"] = 1
    assert validate(doc) == []


# ---------------------------------------------------------------- spike


@pytest.mark.parametrize(
    "path,value,expected",
    [
        (
            "fp16_nan_loss",
            True,
            "fp16_nan_loss: spike failed: NaN loss in fp16, see AS-2 fallback",
        ),
        ("grpo_step_ok", False, "grpo_step_ok: spike failed: the GRPO step did not run"),
        ("install_ok", "yes", "install_ok: expected a boolean"),
        ("fp16_nan_loss", DELETE, "fp16_nan_loss: missing"),
        ("grpo_step_ok", DELETE, "grpo_step_ok: missing"),
        ("reward_sees_full_conversation", DELETE, "reward_sees_full_conversation: missing"),
        ("reward_sees_full_conversation", 1, "reward_sees_full_conversation: expected a boolean"),
        ("sft_steps", "5", "sft_steps: expected an integer"),
        ("completions_0", "", "completions_0: expected a non-empty string"),
        ("completions_0", ["x"], "completions_0: expected a non-empty string"),
        ("kwargs_keys", [], "kwargs_keys: expected a non-empty list of strings"),
        ("kwargs_keys", ["prompts", 3], "kwargs_keys[1]: expected a non-empty string"),
        ("kwargs_keys", "prompts", "kwargs_keys: expected a non-empty list of strings"),
        ("python_version", "", "python_version: expected a non-empty string"),
        ("package_versions", {"torch": 2.11}, "package_versions.torch: expected a string"),
        ("package_versions", [], "package_versions: expected an object"),
    ],
)
def test_spike_rules(path, value, expected):
    errors = validate(mutated("spike", path, value))
    assert any(e.startswith(expected) for e in errors), errors


def test_spike_reporting_no_full_conversation_is_a_valid_finding():
    doc = spike()
    doc["reward_sees_full_conversation"] = False
    assert validate(doc) == []


# ---------------------------------------------------------------- final


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("models.tuned", DELETE, "models.tuned: missing"),
        ("models.prompt_only", DELETE, "models.prompt_only: missing"),
        ("tuned_path", "ppo", "tuned_path: expected one of 'grpo', 'rft', 'sft'"),
        ("report_path", "", "report_path: expected a non-empty string"),
        ("report_path", DELETE, "report_path: missing"),
        ("sc001.verdict", "maybe", "sc001.verdict: expected 'met' or 'not_met'"),
        ("sc001.gaps.vs_base.tool_call_f1", DELETE, "sc001.gaps.vs_base.tool_call_f1: missing"),
        ("sc001.gaps.vs_prompt_only", DELETE, "sc001.gaps.vs_prompt_only: missing"),
        (
            "sc001.gaps.vs_base.prior_trap_rate.gap",
            "big",
            "sc001.gaps.vs_base.prior_trap_rate.gap: expected a number in [-1, 1]",
        ),
        (
            "sc001.gaps.vs_base.prior_trap_rate.ci_high",
            0.1,
            "sc001.gaps.vs_base.prior_trap_rate: expected ci_low <= gap <= ci_high",
        ),
        ("sc001", DELETE, "sc001: missing"),
    ],
)
def test_final_rules(path, value, expected):
    errors = validate(mutated("final", path, value))
    assert any(e.startswith(expected) for e in errors), errors


@pytest.mark.parametrize(
    "path,value,why",
    [
        ("sc001.gaps.vs_base.prior_trap_rate", gap(0.08, 0.02, 0.14), "vs_base.prior_trap_rate"),
        ("sc001.gaps.vs_base.tool_call_f1", gap(0.09, 0.02, 0.16), "vs_base.tool_call_f1"),
        (
            "sc001.gaps.vs_prompt_only.prior_trap_rate",
            gap(0.05, 0.01, 0.09),
            "vs_prompt_only.prior_trap_rate",
        ),
        (
            "sc001.gaps.vs_prompt_only.tool_call_f1",
            gap(0.02, -0.04, 0.08),
            "vs_prompt_only.tool_call_f1",
        ),
        ("sc001.gaps.vs_base.prior_trap_rate", gap(0.30, 0.0, 0.40), "vs_base.prior_trap_rate"),
        (
            "sc001.gaps.vs_prompt_only.tool_call_f1",
            gap(0.25, -0.01, 0.33),
            "vs_prompt_only.tool_call_f1",
        ),
    ],
)
def test_met_verdict_overstating_a_gap_is_caught(path, value, why):
    errors = validate(mutated("final", path, value))
    matching = [e for e in errors if e.startswith('sc001.verdict: expected "not_met"')]
    assert matching and why in matching[0], errors


def test_not_met_verdict_when_all_gaps_pass_is_caught():
    errors = validate(mutated("final", "sc001.verdict", "not_met"))
    assert any(e.startswith('sc001.verdict: expected "met"') for e in errors), errors


def test_gap_of_exactly_ten_points_with_positive_ci_counts_as_met():
    doc = final()
    doc["sc001"]["gaps"]["vs_base"]["prior_trap_rate"] = gap(0.10, 0.01, 0.19)
    assert validate(doc) == []


# ---------------------------------------------------------------- secrets (B-3)

FAKE_HF = "hf_" + "AbCdEfGhIjKlMnOpQrStUv"


def test_hf_token_anywhere_in_the_file_text_is_rejected_without_echo(tmp_path):
    doc = sft()
    doc["trap_note"] = f"pushed with {FAKE_HF} yesterday"
    path = tmp_path / "sft.json"
    path.write_text(json.dumps(doc))
    errors = validate_file(path)
    assert any("contains what looks like a secret (B-3)" in e for e in errors), errors
    assert all(FAKE_HF not in e and "AbCdEfGh" not in e for e in errors)


def test_short_hf_prefix_is_not_a_secret():
    doc = sft()
    doc["checkpoint"]["hub_repo"] = "someone/hf_short"
    assert validate(doc) == []


@pytest.mark.parametrize(
    "key", ["Authorization", "api_key", "hf_token", "access_token", "API-KEY"]
)
def test_secret_looking_key_with_a_value_is_rejected(key):
    doc = sft()
    doc["checkpoint"][key] = "Bearer not-a-real-credential"
    errors = validate(doc)
    assert any(
        e.startswith(f"checkpoint.{key}:") and "secret (B-3)" in e for e in errors
    ), errors
    assert all("not-a-real-credential" not in e for e in errors)


def test_empty_secret_key_and_token_counts_are_fine():
    doc = sft()
    doc["token"] = ""
    doc["decoding_max_new_tokens"] = 256
    doc["tokens"] = "many"
    assert validate(doc) == []


def test_secret_key_nested_in_a_list_is_found():
    doc = sft()
    doc["history"] = [{"api_key": "x1"}]
    errors = validate(doc)
    assert any(e.startswith("history[0].api_key:") and "secret (B-3)" in e for e in errors), errors


# ---------------------------------------------------------------- validate_file and CLI


def test_missing_file_is_one_named_error(tmp_path):
    path = tmp_path / "nope.json"
    errors = validate_file(path)
    assert len(errors) == 1 and errors[0].startswith(f"{path}: cannot read file")


def test_invalid_json_is_one_named_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"kind": "sft",')
    errors = validate_file(path)
    assert len(errors) == 1 and errors[0].startswith(f"{path}: invalid JSON")


def test_top_level_list_is_one_named_error(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]")
    assert validate_file(path) == [f"{path}: expected a JSON object at the top level, got list"]


def test_main_prints_ok_kind_and_returns_zero(tmp_path, capsys):
    path = tmp_path / "spike.json"
    path.write_text(json.dumps(spike()))
    assert check_results.main([str(path)]) == 0
    assert capsys.readouterr().out == "OK spike\n"


def test_main_prints_one_error_per_line_and_returns_one(tmp_path, capsys):
    doc = spike()
    doc["fp16_nan_loss"] = True
    doc["completions_0"] = ""
    path = tmp_path / "spike.json"
    path.write_text(json.dumps(doc))
    assert check_results.main([str(path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("completions_0:") or lines[1].startswith("completions_0:")
    assert any(line.startswith("fp16_nan_loss: spike failed") for line in lines)
    assert not any(line.startswith("OK") for line in lines)


def test_main_without_a_path_returns_usage_code(capsys):
    assert check_results.main([]) == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_script_runs_from_repo_root_with_plain_python(tmp_path):
    good = tmp_path / "grpo.json"
    good.write_text(json.dumps(grpo()))
    ok = subprocess.run(
        [sys.executable, "scripts/check_results.py", str(good)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert (ok.returncode, ok.stdout) == (0, "OK grpo\n")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(mutated("grpo", "generations_per_prompt", 2)))
    fail = subprocess.run(
        [sys.executable, "scripts/check_results.py", str(bad)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert fail.returncode == 1
    assert fail.stdout.startswith("generations_per_prompt: expected an integer >= 4")


# ---------------------------------------------------------------- QA F-7: any API key (B-3)

# Break caught: scanning only hf_[A-Za-z0-9]{20,}, so other key shapes could ship in a file.
@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "AbCdEfGhIjKlMnOpQrStUvWx",
        "sk-proj-" + "AbCdEf_GhIjKl-MnOpQrStUv",
        "Bearer " + "AbCdEfGhIjKlMnOpQrStUvWx",
        "Bearer " + "eyJhbGciOi.JIUzI1NiIsInR5cCI6",
        "hf_" + "AbCdEfGhIjKl_MnOpQrStUv",
    ],
)
def test_other_api_key_shapes_are_rejected_without_echo(secret, tmp_path):
    doc = sft()
    doc["trap_note"] = f"used {secret} for the upload"
    assert any("secret (B-3)" in e for e in validate(doc)), secret
    path = tmp_path / "sft.json"
    path.write_text(json.dumps(doc))
    errors = validate_file(path)
    assert any("contains what looks like a secret (B-3)" in e for e in errors), errors
    assert all(secret[-12:] not in e for e in errors)


@pytest.mark.parametrize(
    "text",
    [
        "CohereLabs/tiny-aya-global",
        "task-0001",
        "heldout-task-0001-with-a-long-identifier-name",
        "risk-free",
        "ask-for-the-risk-free-long-identifier-name",
        "sk-short",
        "Bearer short",
        "bearer of the news",
    ],
)
def test_ordinary_text_is_not_mistaken_for_an_api_key(text):
    doc = sft()
    doc["checkpoint"]["hub_repo"] = text
    assert validate(doc) == []


# ---------------------------------------------------------------- QA F-9: --help

# Break caught: --help / -h read as a results file path ("cannot read file", exit 1).
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_flag_prints_usage_and_returns_zero(flag, capsys):
    assert check_results.main([flag]) == 0
    captured = capsys.readouterr()
    assert captured.out == "usage: python scripts/check_results.py <results/X.json>\n"
    assert captured.err == ""


def test_script_help_flag_exits_zero_from_the_command_line():
    run = subprocess.run(
        [sys.executable, "scripts/check_results.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert (run.returncode, run.stdout) == (
        0, "usage: python scripts/check_results.py <results/X.json>\n")


# ---------------------------------------------------------------- FIX-E1: review findings

SECRET_VALUE = "hf_" + "A" * 30
SECRET_FIELDS = [
    ("sft", "kind"),
    ("sft", "created_at"),
    ("sft", "git_sha"),
    ("sft", "seed"),
    ("sft", "schema_version"),
    ("sft", "steps"),
    ("sft", "checkpoint.revision"),
    ("sft", "models.base.prior_trap_rate.value"),
    ("base_probe", "headroom"),
    ("grpo", "path"),
    ("final", "tuned_path"),
    ("final", "sc001.verdict"),
]


# Break caught: building "expected X, got <value>" errors before the secret scan, so the
# CLI printed a credential that sat in a field with the wrong shape (R1).
@pytest.mark.parametrize("kind,path", SECRET_FIELDS)
def test_secret_in_a_wrongly_typed_field_gives_only_the_path_error(kind, path, tmp_path, capsys):
    doc = mutated(kind, path, SECRET_VALUE)
    assert validate(doc) == [f"{path}: contains what looks like a secret (B-3)"]
    file = tmp_path / "doc.json"
    file.write_text(json.dumps(doc))
    assert check_results.main([str(file)]) == 1
    out = capsys.readouterr().out
    assert out == f"{path}: contains what looks like a secret (B-3)\n"
    assert "AAAA" not in out


# Break caught: scanning raw JSON text only, so a token spelled with a \u escape in
# checkpoint.revision passed and the CLI printed "OK sft" (R7).
def test_escaped_token_in_checkpoint_revision_is_rejected_by_file_and_cli(tmp_path, capsys):
    doc = mutated("sft", "checkpoint.revision", SECRET_VALUE)
    text = json.dumps(doc).replace('"hf_', '"\\u0068f_')
    assert SECRET_VALUE not in text
    file = tmp_path / "sft.json"
    file.write_text(text)
    expected = ["checkpoint.revision: contains what looks like a secret (B-3)"]
    assert validate_file(file) == expected
    assert check_results.main([str(file)]) == 1
    assert capsys.readouterr().out == expected[0] + "\n"


def test_secret_shaped_key_name_is_not_echoed():
    doc = sft()
    doc["trap_note"] = {SECRET_VALUE: 1}
    errors = validate(doc)
    assert errors == ["trap_note.<redacted>: contains what looks like a secret (B-3)"]


def test_show_redacts_a_secret_shaped_string():
    assert check_results._show(SECRET_VALUE) == "<redacted>"
    assert check_results._show("plain") == "'plain'"


# Break caught: `if group` treats {} as absent, so empty comparison objects skipped every
# metric check and a "met" verdict with no gaps passed (R5).
@pytest.mark.parametrize("baseline", ["vs_base", "vs_prompt_only"])
@pytest.mark.parametrize("verdict", ["met", "not_met"])
def test_empty_comparison_object_is_an_error_naming_the_missing_gaps(baseline, verdict):
    doc = mutated("final", f"sc001.gaps.{baseline}", {})
    doc["sc001"]["verdict"] = verdict
    assert validate(doc) == [
        f"sc001.gaps.{baseline}.prior_trap_rate: missing",
        f"sc001.gaps.{baseline}.tool_call_f1: missing",
    ]


def test_empty_gaps_object_names_every_missing_comparison():
    doc = mutated("final", "sc001.gaps", {})
    assert validate(doc) == ["sc001.gaps.vs_base: missing", "sc001.gaps.vs_prompt_only: missing"]


# Break caught: unhashable `kind`, huge ints and deep or odd JSON escaping validate() as
# TypeError / OverflowError / ValueError / RecursionError instead of error lists (R9).
def _leaf_paths(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, prefix + (key,))
    else:
        yield prefix


def _set_path(doc, path, value):
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _deep_list(depth=900):
    return json.loads("[" * depth + "]" * depth)


ODD_VALUES = [
    [], {}, None, True, "", "x", 0, -1, 10**400, 10**5000, -(10**400), 1e308, 2.5,
    float("nan"), float("inf"), [[]], {"a": []}, _deep_list(), "\ud800",
]


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_no_odd_value_in_any_field_makes_validate_raise(kind):
    paths = list(_leaf_paths(BUILDERS[kind]()))
    assert paths
    for path in paths:
        for odd in ODD_VALUES:
            doc = BUILDERS[kind]()
            _set_path(doc, path, odd)
            errors = validate(doc)
            assert isinstance(errors, list) and all(isinstance(e, str) for e in errors), path


@pytest.mark.parametrize("kind_value", [[], {}, [1], {"a": 1}, 10**400])
def test_unhashable_or_huge_kind_is_one_kind_error(kind_value):
    errors = validate({"kind": kind_value})
    assert errors[0].startswith("kind: expected one of base_probe, eval_baselines, sft")


def test_huge_loss_is_an_error_not_an_overflow():
    doc = mutated("sft", "loss_first", 10**400)
    assert validate(doc) == ["loss_first: expected a number, got 1" + "0" * 400]
    doc = mutated("sft", "loss_first", 10**5000)
    assert validate(doc) == ["loss_first: expected a number, got <value too large to show>"]


@pytest.mark.parametrize(
    "text",
    [
        '{"kind": "sft", "seed": ' + "9" * 5000 + "}",
        "[" * 100000,
        '{"kind": "sft", "x": ' + "[" * 100000,
        '{"kind": "sft", "\\ud800api_key": "v"}',
    ],
)
def test_odd_json_files_never_raise_through_the_cli(text, tmp_path, capsys):
    file = tmp_path / "odd.json"
    file.write_text(text)
    assert check_results.main([str(file)]) == 1
    assert capsys.readouterr().out.strip() != ""
