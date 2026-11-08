"""Tests for the Colab spike helpers and notebook (T-010; AS-1, AS-2, AS-3, B-1).

The notebook itself needs a GPU and cannot run here, so these tests check that it is
well-formed, leaks no secret, and that the helpers it calls produce a results file the
validator accepts.
"""

import ast
import json
import re
from pathlib import Path

import pytest

import check_results
import spike
from alien.generator import Task

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "spike.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_sources(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _good_first_step() -> dict:
    return {
        "completions_0": '[{"role": "assistant", "content": "{}"}]',
        "kwargs_keys": ["prompts", "task_id", "trainer_state"],
        "reward_sees_full_conversation": True,
    }


def _good_doc(losses=(2.5, 2.1, 1.8), **overrides) -> dict:
    args = dict(
        install_ok=True,
        sft_losses=list(losses),
        grpo_step_ok=True,
        first_step=_good_first_step(),
        python_version="3.12.4",
        package_versions={"torch": "2.11.0", "trl": "1.7.1"},
        seed=0,
        git_sha="abc1234",
    )
    args.update(overrides)
    return spike.build_spike_results(**args)


# ---- notebook ----


def test_notebook_is_nbformat_4_with_cleared_outputs(notebook):
    assert notebook["nbformat"] == 4
    assert 8 <= len(notebook["cells"]) <= 14
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_every_code_cell_parses_after_dropping_shell_lines(notebook):
    for source in _code_sources(notebook):
        kept = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(kept))


def test_notebook_contains_no_hf_secret(notebook):
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_never_prints_the_token(notebook):
    for source in _code_sources(notebook):
        for line in source.splitlines():
            assert not re.search(r"print\(.*token", line, re.IGNORECASE), line


def test_notebook_reads_token_from_colab_secrets_into_environ(notebook):
    text = "\n".join(_code_sources(notebook))
    assert 'userdata.get("HF_TOKEN")' in text
    assert 'os.environ["HF_TOKEN"]' in text


def test_notebook_prints_b1_evidence_and_pins_trl(notebook):
    text = "\n".join(_code_sources(notebook))
    assert "completions[0]" in text
    assert "kwargs.keys()" in text
    assert "trl==1.7.1" in text
    assert "num_generations=4" in text
    assert "max_steps=20" in text
    assert "scripts/check_results.py results/spike.json" in text


def test_notebook_documents_unverified_lines_and_fallbacks(notebook):
    markdown = "\n".join("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "markdown")
    text = "\n".join(_code_sources(notebook)) + markdown
    assert "VERIFY ON COLAB" in text
    assert "AS-1 fallback" in markdown or "If AS-1 fails" in markdown
    assert "AS-2 fallback" in markdown
    assert "QLoRA" in markdown


# ---- build_spike_results / write_results ----


def test_good_run_passes_the_validator():
    doc = _good_doc()
    assert check_results.validate(doc) == []
    assert doc["kind"] == "spike"
    assert doc["schema_version"] == 1
    assert doc["sft_steps"] == 3
    assert doc["fp16_nan_loss"] is False
    assert doc["git_sha"] == "abc1234"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["created_at"])


def test_nan_losses_give_the_validator_nan_loss_error():
    doc = _good_doc(losses=[2.5, float("nan"), 2.0])
    assert doc["fp16_nan_loss"] is True
    errors = check_results.validate(doc)
    assert any("NaN loss" in e for e in errors)


def test_loss_is_nan_flags_nan_and_inf_but_not_finite_values():
    assert spike.loss_is_nan([1.0, float("nan")]) is True
    assert spike.loss_is_nan([1.0, float("inf")]) is True
    assert spike.loss_is_nan([0.0, -3.5, 12.0]) is False
    assert spike.loss_is_nan([]) is False


def test_write_results_writes_valid_json_atomically(tmp_path):
    target = tmp_path / "results" / "spike.json"
    spike.write_results(target, _good_doc())
    assert check_results.validate_file(target) == []
    assert json.loads(target.read_text())["python_version"] == "3.12.4"
    assert list(target.parent.iterdir()) == [target]  # no temp file left behind


def test_write_results_refuses_an_invalid_doc_and_writes_nothing(tmp_path):
    target = tmp_path / "spike.json"
    with pytest.raises(ValueError, match="NaN loss"):
        spike.write_results(target, _good_doc(losses=[1.0, float("nan")]))
    assert list(tmp_path.iterdir()) == []


# ---- capture_first_step ----


def test_capture_string_completion_without_prompts_is_not_full_conversation():
    out = spike.capture_first_step(['{"name": "move"}'], {"task_id": ["t0"]})
    assert out == {
        "completions_0": '{"name": "move"}',
        "kwargs_keys": ["task_id"],
        "reward_sees_full_conversation": False,
    }


def test_capture_two_message_completion_is_full_conversation():
    completion = [
        {"role": "user", "content": "Move crate 3"},
        {"role": "assistant", "content": '{"name": "move"}'},
    ]
    out = spike.capture_first_step([completion], {})
    assert out["reward_sees_full_conversation"] is True
    assert '"role": "user"' in out["completions_0"]


def test_capture_single_message_completion_with_prompts_kwarg_is_full_conversation():
    completions = [[{"role": "assistant", "content": "{}"}]]
    prompts = [[{"role": "user", "content": "Move crate 3"}]]
    out = spike.capture_first_step(completions, {"trainer_state": object(), "prompts": prompts})
    assert out["reward_sees_full_conversation"] is True
    assert out["kwargs_keys"] == ["prompts", "trainer_state"]


def test_capture_single_message_completion_alone_is_not_full_conversation():
    out = spike.capture_first_step([[{"role": "assistant", "content": "{}"}]], {"x": 1})
    assert out["reward_sees_full_conversation"] is False


def test_capture_truncates_to_2000_chars_and_survives_odd_objects():
    out = spike.capture_first_step(["a" * 5000], {})
    assert len(out["completions_0"]) == 2000
    odd = spike.capture_first_step([[{"role": "assistant", "content": object()}]], {})
    assert "assistant" in odd["completions_0"]


def test_capture_output_is_never_empty_so_the_validator_accepts_it():
    out = spike.capture_first_step([""], {"k": 1})
    assert out["completions_0"] == "''"
    assert spike.capture_first_step([], {"k": 1})["completions_0"] == "<no completions>"


# ---- data builders ----


def test_sft_texts_are_chat_examples_ending_in_an_oracle_json_call():
    rows = spike.build_sft_texts(n=5, seed=0)
    assert len(rows) == 5
    for row in rows:
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        call = json.loads(row["messages"][1]["content"])
        assert set(call) == {"name", "arguments"}


def test_sft_texts_are_deterministic_per_seed():
    assert spike.build_sft_texts(n=4, seed=3) == spike.build_sft_texts(n=4, seed=3)
    assert spike.build_sft_texts(n=4, seed=3) != spike.build_sft_texts(n=4, seed=4)


def test_grpo_rows_have_conversational_prompt_and_task_id_matching_the_tasks():
    rows = spike.build_grpo_rows(n=8, seed=0)
    tasks = spike.tasks_by_id(n=8, seed=0)
    assert len(rows) == 8
    assert set(rows[0]) == {"prompt", "task_id"}
    assert rows[0]["prompt"][0]["role"] == "user"
    assert [r["task_id"] for r in rows] == list(tasks)
    assert all(r["task_id"].startswith("train-") for r in rows)


def test_builders_reject_sizes_outside_the_train_split():
    with pytest.raises(ValueError, match="n must be between 1 and 600"):
        spike.build_sft_texts(n=0)
    with pytest.raises(ValueError, match="n must be between 1 and 600"):
        spike.build_grpo_rows(n=601)


# ---- reward stand-in and versions ----


def test_fallback_exact_match_scores_only_the_exact_oracle_call():
    task = next(iter(spike.tasks_by_id(n=1, seed=0).values()))
    assert isinstance(task, Task)
    exact = json.dumps(task.oracle_call)
    assert spike.fallback_exact_match(task, exact) == 1.0
    assert spike.fallback_exact_match(task, [{"role": "assistant", "content": exact}]) == 1.0
    assert spike.fallback_exact_match(task, "not json") == 0.0
    trap = next(t for t in spike.tasks_by_id(n=8, seed=0).values() if t.is_trap)
    assert spike.fallback_exact_match(trap, json.dumps(trap.habit_call)) == 0.0


def test_collect_package_versions_reports_missing_packages():
    versions = spike.collect_package_versions(["pytest", "definitely-not-a-package-xyz"])
    assert re.fullmatch(r"\d+\.\d+.*", versions["pytest"])
    assert versions["definitely-not-a-package-xyz"] == "not installed"
