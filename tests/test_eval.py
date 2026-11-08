"""Tests for the three-way eval harness and its CLI (FR-002, FR-006, FR-007, E-4..E-6).

Everything runs on fake deterministic generators: no torch, no GPU, no download. Each test
group names the production change that would make it fail.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_results
import run_eval
from alien.eval import (
    DECODING,
    MODEL_ID,
    ResultsInvalidError,
    StaleProgressError,
    build_prompts,
    build_results,
    evaluate_variants,
    run_fingerprint,
    run_variant,
    write_results,
)
from alien.oracle import FOLLOW_DOCS_LINE, call_to_json, render_prompt
from alien.persist import ProgressStore
from alien.scoring import NoEligibleExamplesError
from alien.splits import make_splits

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def heldout():
    return make_splits(0).heldout


class Interrupted(Exception):
    """Stands in for a free Colab session dying mid-run."""


class FakeClock:
    """Time only moves when a generator says so, so seconds are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def lookup_generator(tasks, kind, *, clock=None, step=0.0, fail_on_call=None):
    """A fake model: finds the task behind a prompt and answers oracle or habit.

    Counts calls and prompts; after ``fail_on_call`` calls it raises Interrupted.
    """
    table = {}
    for task in tasks:
        call = task.oracle_call if kind == "oracle" else task.habit_call
        for follow_docs in (False, True):
            table[render_prompt(task, follow_docs)] = call_to_json(call)

    def generate(prompts):
        generate.calls += 1
        if fail_on_call is not None and generate.calls == fail_on_call:
            raise Interrupted
        generate.prompts.extend(prompts)
        if clock is not None:
            clock.now += step
        return [table[p] for p in prompts]

    generate.calls = 0
    generate.prompts = []
    return generate


# ---- prompts (FR-006): the only difference between base and prompt-only is one line -----


def prompts_differ_by_one_line(task) -> bool:
    base, prompt_only = build_prompts([task], "base")[0], build_prompts([task], "prompt_only")[0]
    base_lines, other_lines = base.split("\n"), prompt_only.split("\n")
    extra = [line for line in other_lines if line not in base_lines]
    removed = [line for line in base_lines if line not in other_lines]
    one_extra_line = len(other_lines) == len(base_lines) + 1
    return extra == [FOLLOW_DOCS_LINE] and removed == [] and one_extra_line


def test_base_and_prompt_only_differ_by_exactly_the_follow_docs_line(heldout):
    # Breaks if prompt_only rewords the docs, adds a second line, or drops a base line.
    assert FOLLOW_DOCS_LINE.startswith("Follow the tool documentation exactly")
    assert len(heldout) == 240
    assert all(prompts_differ_by_one_line(task) for task in heldout)


def test_tuned_prompt_is_the_base_prompt(heldout):
    # Breaks if the tuned model is handed the follow-the-docs hint.
    tasks = heldout[:10]
    assert build_prompts(tasks, "tuned") == build_prompts(tasks, "base")
    assert build_prompts(tasks, "tuned") != build_prompts(tasks, "prompt_only")


def test_unknown_variant_is_rejected(heldout):
    with pytest.raises(ValueError, match="sft"):
        build_prompts(heldout[:2], "sft")


def test_decoding_is_greedy_and_capped_at_256_tokens():
    # Breaks if a sampling default or a longer generation sneaks in (FR-002 limit).
    assert DECODING == {"max_new_tokens": 256, "temperature": 0.0, "do_sample": False}
    assert MODEL_ID == "CohereLabs/tiny-aya-global"


# ---- run_variant: batching, recording, resume (E-6) ----------------------------------------


def test_run_variant_records_each_batch_and_returns_outputs_in_task_order(heldout, tmp_path):
    tasks = heldout[:20]
    store = ProgressStore(tmp_path / "base.jsonl")
    gen = lookup_generator(tasks, "oracle")
    result = run_variant("base", tasks, gen, store, batch_size=8)
    assert gen.calls == 3  # 8 + 8 + 4
    assert result["generated"] == 20
    assert result["outputs"] == [call_to_json(t.oracle_call) for t in tasks]
    assert store.done_ids() == {t.id for t in tasks}


def test_run_variant_rejects_a_generator_that_returns_the_wrong_count(heldout, tmp_path):
    # Breaks if a short batch is zipped silently, which would lose examples (E-6).
    tasks = heldout[:8]
    store = ProgressStore(tmp_path / "base.jsonl")
    with pytest.raises(ValueError, match="returned 7 completions for 8 prompts"):
        run_variant("base", tasks, lambda prompts: ["{}"] * 7, store, batch_size=8)
    assert store.done_ids() == set()


def test_interrupted_run_resumes_without_regenerating_or_double_scoring(heldout, tmp_path):
    # Breaks if resume re-generates recorded ids, loses the batch before the crash, or
    # appends an id twice.
    tasks = heldout[:40]
    expected = [call_to_json(t.oracle_call) for t in tasks]
    progress = tmp_path / "progress"

    dying = lookup_generator(tasks, "oracle", fail_on_call=4)  # 3 batches survive
    with pytest.raises(Interrupted):
        evaluate_variants(tasks, {"base": dying}, progress, batch_size=8, n_resamples=10)
    assert dying.calls == 4
    assert len(ProgressStore(progress / "base.jsonl").done_ids()) == 24

    survivor = lookup_generator(tasks, "oracle")
    evaluated = evaluate_variants(tasks, {"base": survivor}, progress, batch_size=8,
                                  n_resamples=10)
    assert survivor.calls == 2
    done_before_crash = {render_prompt(t) for t in tasks[:24]}
    assert not done_before_crash & set(survivor.prompts)
    assert survivor.prompts == [render_prompt(t) for t in tasks[24:]]

    lines = (progress / "base.jsonl").read_text().splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == [t.id for t in tasks]  # 40 lines, each id exactly once, in task order
    stored = ProgressStore(progress / "base.jsonl").results()
    assert [stored[t.id]["output"] for t in tasks] == expected

    uninterrupted = evaluate_variants(tasks, {"base": lookup_generator(tasks, "oracle")},
                                      tmp_path / "fresh", batch_size=8, n_resamples=10)
    assert evaluated["models"] == uninterrupted["models"]
    assert evaluated["n_examples"] == 40


def test_a_finished_run_makes_no_generator_calls_on_rerun(heldout, tmp_path):
    tasks = heldout[:16]
    first = lookup_generator(tasks, "oracle")
    evaluate_variants(tasks, {"base": first}, tmp_path, batch_size=8, n_resamples=10)
    again = lookup_generator(tasks, "oracle")
    evaluate_variants(tasks, {"base": again}, tmp_path, batch_size=8, n_resamples=10)
    assert again.calls == 0


# ---- scoring through the harness -----------------------------------------------------------


def test_oracle_generator_scores_perfect_and_habit_generator_falls_in_every_trap(
    heldout, tmp_path
):
    # Breaks if outputs are paired with the wrong task, or variants share one progress file.
    tasks = heldout[:40]  # 24 trap tasks, 16 normal tasks
    evaluated = evaluate_variants(
        tasks,
        {"base": lookup_generator(tasks, "habit"),
         "prompt_only": lookup_generator(tasks, "oracle")},
        tmp_path,
        n_resamples=20,
    )
    habit, oracle = evaluated["models"]["base"], evaluated["models"]["prompt_only"]
    assert habit["prior_trap_rate"]["value"] == 1.0
    assert habit["prior_trap_rate"]["n"] == 24
    assert habit["tool_call_f1"]["value"] == pytest.approx(0.4)  # 2*16 / (2*16 + 24 + 24)
    assert habit["normal_tool_f1"]["value"] == 1.0
    assert habit["normal_tool_f1"]["n"] == 16
    assert habit["tool_call_validity"]["value"] == 1.0
    assert oracle["prior_trap_rate"]["value"] == 0.0
    assert oracle["tool_call_f1"]["value"] == 1.0
    assert oracle["tool_call_validity"]["n"] == 40
    assert evaluated["n_examples"] == 40


def test_missing_or_malformed_output_scores_zero_not_an_error(heldout, tmp_path):
    # E-4: a model that answers "I cannot say" on every example is scored, not crashed on.
    tasks = heldout[:40]
    evaluated = evaluate_variants(tasks, {"base": lambda prompts: ["I cannot say"] * len(prompts)},
                                  tmp_path, n_resamples=10)
    model = evaluated["models"]["base"]
    assert model["tool_call_f1"]["value"] == 0.0
    assert model["tool_call_validity"]["value"] == 0.0
    assert model["prior_trap_rate"]["value"] == 0.0


def test_no_trap_tasks_aborts_before_any_generation_and_names_the_metric(heldout, tmp_path):
    # E-5: breaks if zero eligible examples become a printed 0/NaN, or only fail after the
    # GPU hours were spent.
    normal_only = [t for t in heldout if not t.is_trap][:20]
    gen = lookup_generator(normal_only, "oracle")
    with pytest.raises(NoEligibleExamplesError, match="prior_trap_rate"):
        evaluate_variants(normal_only, {"base": gen}, tmp_path, n_resamples=10)
    assert gen.calls == 0


def test_no_normal_tasks_aborts_naming_normal_tool_f1(heldout, tmp_path):
    trap_only = [t for t in heldout if t.is_trap][:20]
    with pytest.raises(NoEligibleExamplesError, match="normal_tool_f1"):
        evaluate_variants(trap_only, {"base": lookup_generator(trap_only, "oracle")},
                          tmp_path, n_resamples=10)


# ---- timing (FR-002) -----------------------------------------------------------------------


def test_gpu_seconds_add_up_across_sessions_via_timing_json(heldout, tmp_path):
    # Breaks if a resumed session overwrites the earlier session's time instead of adding.
    tasks = heldout[:40]
    clock = FakeClock()
    first = lookup_generator(tasks, "oracle", clock=clock, step=10.0, fail_on_call=4)
    with pytest.raises(Interrupted):
        evaluate_variants(tasks, {"base": first}, tmp_path, batch_size=8, n_resamples=10,
                          clock=clock)
    assert json.loads((tmp_path / "timing.json").read_text()) == {"seconds": {"base": 30.0}}

    second = lookup_generator(tasks, "oracle", clock=clock, step=10.0)
    evaluated = evaluate_variants(tasks, {"base": second}, tmp_path, batch_size=8,
                                  n_resamples=10, clock=clock)
    assert second.calls == 2
    assert evaluated["gpu_seconds"] == 50.0
    assert json.loads((tmp_path / "timing.json").read_text()) == {"seconds": {"base": 50.0}}


def test_gpu_seconds_only_count_the_requested_variants(heldout, tmp_path):
    tasks = heldout[:16]
    clock = FakeClock()
    (tmp_path / "timing.json").write_text(json.dumps({"seconds": {"tuned": 999.0}}))
    gens = {name: lookup_generator(tasks, "oracle", clock=clock, step=5.0)
            for name in ("base", "prompt_only")}
    evaluated = evaluate_variants(tasks, gens, tmp_path, batch_size=8, n_resamples=10,
                                  clock=clock)
    assert evaluated["gpu_seconds"] == 20.0  # 2 variants x 2 batches x 5 s


def make_results(tasks, tmp_path, seconds_per_batch):
    clock = FakeClock()
    gens = {name: lookup_generator(tasks, "oracle", clock=clock, step=seconds_per_batch)
            for name in ("base", "prompt_only")}
    evaluated = evaluate_variants(tasks, gens, tmp_path, batch_size=8, n_resamples=10,
                                  clock=clock)
    return build_results(evaluated, seed=0, model_revision="abc123")


def test_within_budget_flips_at_exactly_10800_seconds(heldout, tmp_path):
    # Breaks if the budget comparison is <= instead of < (FR-002: under 3 hours).
    tasks = heldout[:16]  # 2 variants x 2 batches = 4 batches
    assert make_results(tasks, tmp_path / "a", 2699.0)["within_budget"] is True  # 10796 s
    assert make_results(tasks, tmp_path / "b", 2700.0)["within_budget"] is False  # 10800 s
    assert make_results(tasks, tmp_path / "c", 2700.0)["gpu_seconds"] == 10800.0


# ---- results document ----------------------------------------------------------------------


def test_results_document_has_the_contract_fields_and_passes_check_results(
    heldout, tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_SHA", "abc1234")
    tasks = heldout[:200]
    clock = FakeClock()
    gens = {"base": lookup_generator(tasks, "habit", clock=clock, step=1.5),
            "prompt_only": lookup_generator(tasks, "oracle", clock=clock, step=1.5)}
    evaluated = evaluate_variants(tasks, gens, tmp_path / "p", n_resamples=20, clock=clock)
    doc = build_results(evaluated, seed=7, model_revision="deadbeef",
                        created_at="2026-11-08T10:00:00Z")
    assert check_results.validate(doc) == []
    assert set(doc) == {"kind", "schema_version", "created_at", "git_sha", "seed", "n_examples",
                        "split", "model_revision", "decoding", "models", "gpu_seconds",
                        "within_budget"}
    assert (doc["kind"], doc["schema_version"], doc["seed"]) == ("eval_baselines", 1, 7)
    assert (doc["split"], doc["n_examples"], doc["git_sha"]) == ("heldout", 200, "abc1234")
    assert doc["model_revision"] == "deadbeef"
    assert doc["decoding"] == {"max_new_tokens": 256, "temperature": 0.0, "do_sample": False}
    assert list(doc["models"]) == ["base", "prompt_only"]
    assert doc["models"]["base"]["prior_trap_rate"]["value"] == 1.0
    assert doc["models"]["prompt_only"]["tool_call_f1"]["value"] == 1.0
    assert doc["gpu_seconds"] == 75.0  # 2 variants x 25 batches x 1.5 s
    assert doc["within_budget"] is True


def test_git_sha_comes_from_the_environment_and_defaults_to_null(heldout, tmp_path, monkeypatch):
    tasks = heldout[:16]
    evaluated = evaluate_variants(
        tasks,
        {n: lookup_generator(tasks, "oracle") for n in ("base", "prompt_only")},
        tmp_path, n_resamples=10,
    )
    monkeypatch.delenv("GIT_SHA", raising=False)
    assert build_results(evaluated, seed=0, model_revision="r")["git_sha"] is None
    monkeypatch.setenv("GIT_SHA", "0123abc")
    assert build_results(evaluated, seed=0, model_revision="r")["git_sha"] == "0123abc"


def test_build_results_leaves_the_tuned_model_out_of_eval_baselines(heldout, tmp_path):
    tasks = heldout[:16]
    evaluated = evaluate_variants(
        tasks,
        {n: lookup_generator(tasks, "oracle") for n in ("base", "prompt_only", "tuned")},
        tmp_path, n_resamples=10,
    )
    assert list(build_results(evaluated, seed=0, model_revision="r")["models"]) == [
        "base", "prompt_only"]


def test_build_results_needs_both_baselines(heldout, tmp_path):
    tasks = heldout[:16]
    evaluated = evaluate_variants(tasks, {"base": lookup_generator(tasks, "oracle")},
                                  tmp_path, n_resamples=10)
    with pytest.raises(ValueError, match="prompt_only"):
        build_results(evaluated, seed=0, model_revision="r")


def test_write_results_refuses_an_invalid_document_and_writes_nothing(heldout, tmp_path):
    # 40 examples is not the 200 the contract demands, so validation must stop the write.
    doc = make_results(heldout[:16], tmp_path / "p", 1.0)
    out = tmp_path / "results" / "eval_baselines.json"
    with pytest.raises(ResultsInvalidError, match="n_examples: expected 200, got 16"):
        write_results(out, doc, check_results.validate)
    assert not out.exists()
    assert not out.parent.exists() or list(out.parent.iterdir()) == []


def test_write_results_writes_valid_json_atomically(heldout, tmp_path):
    tasks = heldout[:200]
    evaluated = evaluate_variants(
        tasks,
        {n: lookup_generator(tasks, "oracle") for n in ("base", "prompt_only")},
        tmp_path / "p", n_resamples=10,
    )
    doc = build_results(evaluated, seed=0, model_revision="r")
    out = tmp_path / "results" / "eval_baselines.json"
    write_results(out, doc, check_results.validate)
    assert json.loads(out.read_text()) == doc
    assert [p.name for p in out.parent.iterdir()] == ["eval_baselines.json"]  # no temp left
    assert check_results.validate_file(out) == []


# ---- CLI -----------------------------------------------------------------------------------


def test_help_exits_zero_without_torch_or_transformers():
    # Breaks if the script imports torch/transformers/peft at module level.
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "run_eval.py"), "--help"],
                          capture_output=True, text=True, timeout=60, cwd=REPO)
    assert done.returncode == 0
    for flag in ("--splits", "--which", "--fake", "--progress-dir", "--adapter", "--n-resamples"):
        assert flag in done.stdout


def fake_run(tmp_path, *extra):
    out = tmp_path / "eval_baselines.json"
    argv = ["--fake", "oracle", "--out", str(out), "--progress-dir", str(tmp_path / "progress"),
            "--n-resamples", "20", *extra]
    return out, run_eval.main(argv)


def test_fake_dry_run_writes_a_results_file_that_check_results_accepts(tmp_path, capsys):
    out, code = fake_run(tmp_path)
    assert code == 0
    assert check_results.validate_file(out) == []
    doc = json.loads(out.read_text())
    assert (doc["n_examples"], doc["split"], doc["models"]["base"]["tool_call_f1"]["value"]) == (
        200, "heldout", 1.0)
    assert doc["model_revision"] == "fake"
    printed = capsys.readouterr().out
    for fragment in ("model: fake", "examples: 200", "variants: base, prompt_only",
                     "generation budget"):
        assert fragment in printed


def test_fake_habit_dry_run_reports_every_trap_as_a_prior_trap(tmp_path):
    out = tmp_path / "eval_baselines.json"
    code = run_eval.main(["--fake", "habit", "--out", str(out), "--progress-dir",
                          str(tmp_path / "progress"), "--n-resamples", "20"])
    assert code == 0
    models = json.loads(out.read_text())["models"]
    assert models["base"]["prior_trap_rate"]["value"] == 1.0
    assert models["prompt_only"]["prior_trap_rate"]["value"] == 1.0
    assert models["base"]["prior_trap_rate"]["n"] == 115


def test_cli_exits_nonzero_and_writes_nothing_when_validation_fails(tmp_path, capsys):
    # --n 40 cannot satisfy the contract's n_examples == 200.
    out, code = fake_run(tmp_path, "--n", "40")
    assert code == 1
    assert not out.exists()
    assert "n_examples: expected 200, got 40" in capsys.readouterr().err


def test_cli_rerun_after_completion_does_not_regenerate(tmp_path):
    out, code = fake_run(tmp_path)
    assert code == 0
    before = (tmp_path / "progress" / "base.jsonl").read_text()
    first_seconds = json.loads(out.read_text())["gpu_seconds"]
    out2, code2 = fake_run(tmp_path)
    assert code2 == 0
    assert (tmp_path / "progress" / "base.jsonl").read_text() == before
    assert json.loads(out2.read_text())["gpu_seconds"] == pytest.approx(first_seconds, abs=1e-6)


def test_cli_rejects_tuned_without_adapter_and_mixed_runs(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_eval.main(["--fake", "oracle", "--which", "tuned", "--out", str(tmp_path / "x.json"),
                       "--progress-dir", str(tmp_path / "p")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        run_eval.main(["--fake", "oracle", "--which", "base,tuned", "--adapter", "a/b",
                       "--out", str(tmp_path / "x.json"), "--progress-dir", str(tmp_path / "p")])
    assert "tuned" in capsys.readouterr().err


def test_cli_tuned_run_writes_minimal_tuned_data_not_an_eval_baselines_file(tmp_path):
    out = tmp_path / "eval_tuned.json"
    code = run_eval.main(["--fake", "oracle", "--which", "tuned", "--adapter", "me/lora",
                          "--out", str(out), "--progress-dir", str(tmp_path / "p"),
                          "--n-resamples", "20"])
    assert code == 0
    doc = json.loads(out.read_text())
    assert list(doc["models"]) == ["tuned"]
    assert doc["models"]["tuned"]["tool_call_f1"]["value"] == 1.0
    assert (doc["n_examples"], doc["split"], doc["adapter"]) == (200, "heldout", "me/lora")
    assert "kind" not in doc


# ---- stale progress (F-5): outputs of another run are never reused --------------------------


def test_seed_zero_progress_is_refused_for_seed_one_not_reused(tmp_path):
    # Breaks if progress is deduped by task id alone: heldout-0000.. repeat in every seed, so
    # seed 1 would get "0 generations" and seed 0's outputs.
    tasks0, tasks1 = make_splits(0).heldout[:16], make_splits(1).heldout[:16]
    assert [t.id for t in tasks0] == [t.id for t in tasks1]
    assert build_prompts(tasks0, "base") != build_prompts(tasks1, "base")
    evaluate_variants(tasks0, {"base": lookup_generator(tasks0, "habit")}, tmp_path,
                      n_resamples=10)
    before = (tmp_path / "base.jsonl").read_text()
    gen = lookup_generator(tasks1, "oracle")
    with pytest.raises(StaleProgressError, match="fresh --progress-dir") as info:
        evaluate_variants(tasks1, {"base": gen}, tmp_path, n_resamples=10)
    assert str(tmp_path / "base.jsonl") in str(info.value)
    assert gen.calls == 0
    assert (tmp_path / "base.jsonl").read_text() == before


def test_a_second_adapter_in_the_same_progress_dir_is_refused(heldout, tmp_path):
    # Breaks if run_key is ignored: adapter B would be scored on adapter A's recorded outputs.
    tasks = heldout[:16]
    evaluate_variants(tasks, {"tuned": lookup_generator(tasks, "oracle")}, tmp_path,
                      n_resamples=10, run_key="adapter=A")
    gen = lookup_generator(tasks, "habit")
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        evaluate_variants(tasks, {"tuned": gen}, tmp_path, n_resamples=10, run_key="adapter=B")
    assert gen.calls == 0


def test_matching_run_key_resumes_and_generates_only_missing_ids(heldout, tmp_path):
    # Breaks if the fingerprint check rejects a legitimate resume or regenerates recorded ids.
    tasks = heldout[:40]
    dying = lookup_generator(tasks, "oracle", fail_on_call=4)
    with pytest.raises(Interrupted):
        evaluate_variants(tasks, {"base": dying}, tmp_path, batch_size=8, n_resamples=10,
                          run_key="rev=abc")
    survivor = lookup_generator(tasks, "oracle")
    evaluated = evaluate_variants(tasks, {"base": survivor}, tmp_path, batch_size=8,
                                  n_resamples=10, run_key="rev=abc")
    assert survivor.calls == 2
    assert survivor.prompts == [render_prompt(t) for t in tasks[24:]]
    assert evaluated["models"]["base"]["tool_call_f1"]["value"] == 1.0
    finished = lookup_generator(tasks, "oracle")
    evaluate_variants(tasks, {"base": finished}, tmp_path, batch_size=8, n_resamples=10,
                      run_key="rev=abc")
    assert finished.calls == 0


def test_progress_file_without_a_fingerprint_is_refused(heldout, tmp_path):
    # Breaks if a file written by older code (no fingerprint) is trusted: its run is unknown.
    tasks = heldout[:16]
    legacy = ProgressStore(tmp_path / "base.jsonl")
    for task in tasks:
        legacy.record(task.id, {"output": call_to_json(task.habit_call)})
    gen = lookup_generator(tasks, "oracle")
    with pytest.raises(StaleProgressError, match="no run fingerprint"):
        evaluate_variants(tasks, {"base": gen}, tmp_path, n_resamples=10)
    assert gen.calls == 0


def test_a_stale_second_variant_is_found_before_the_first_one_generates(heldout, tmp_path):
    # Breaks if stores are opened lazily: base would burn GPU time before prompt_only fails.
    tasks = heldout[:16]
    evaluate_variants(tasks, {"prompt_only": lookup_generator(tasks, "oracle")}, tmp_path,
                      n_resamples=10, run_key="rev=old")
    base = lookup_generator(tasks, "oracle")
    with pytest.raises(StaleProgressError):
        evaluate_variants(tasks, {"base": base, "prompt_only": lookup_generator(tasks, "oracle")},
                          tmp_path, n_resamples=10, run_key="rev=new")
    assert base.calls == 0
    assert not (tmp_path / "base.jsonl").exists()


def test_run_fingerprint_is_a_pinned_sha256_of_key_variant_and_prompts():
    # Breaks if the fingerprint format changes silently, orphaning every saved progress file.
    assert run_fingerprint("base", ["p1", "p2"], "k") == (
        "fba30417cd33b90cc7eb75e5d42f8f9df6903a5fda42998508c73a4cc2560171")
    base = run_fingerprint("base", ["p1", "p2"], "k")
    assert base != run_fingerprint("base", ["p2", "p1"], "k")
    assert base != run_fingerprint("tuned", ["p1", "p2"], "k")
    assert base != run_fingerprint("base", ["p1", "p2"], "j")


def test_timing_of_a_deleted_progress_file_is_not_added_to_the_new_run(heldout, tmp_path):
    # Breaks if timing.json seconds survive deleting the stale progress file: the new run's
    # gpu_seconds would include the other run's time (FR-002 budget would be overstated).
    tasks = heldout[:16]
    clock = FakeClock()
    evaluate_variants(tasks, {"base": lookup_generator(tasks, "oracle", clock=clock, step=10.0)},
                      tmp_path, batch_size=8, n_resamples=10, clock=clock, run_key="adapter=A")
    assert json.loads((tmp_path / "timing.json").read_text()) == {"seconds": {"base": 20.0}}
    (tmp_path / "base.jsonl").unlink()  # what the error message tells the user to do
    (tmp_path / "base.jsonl.key").unlink()
    evaluated = evaluate_variants(
        tasks, {"base": lookup_generator(tasks, "oracle", clock=clock, step=3.0)}, tmp_path,
        batch_size=8, n_resamples=10, clock=clock, run_key="adapter=B")
    assert evaluated["gpu_seconds"] == 6.0
    assert json.loads((tmp_path / "timing.json").read_text()) == {"seconds": {"base": 6.0}}


def test_a_refused_run_leaves_timing_json_untouched(heldout, tmp_path):
    # Breaks if the stale-progress check happens after timing.json was rewritten.
    tasks = heldout[:16]
    clock = FakeClock()
    evaluate_variants(tasks, {"base": lookup_generator(tasks, "oracle", clock=clock, step=10.0)},
                      tmp_path, batch_size=8, n_resamples=10, clock=clock, run_key="A")
    before = (tmp_path / "timing.json").read_text()
    with pytest.raises(StaleProgressError):
        evaluate_variants(tasks, {"base": lookup_generator(tasks, "oracle")}, tmp_path,
                          batch_size=8, n_resamples=10, clock=clock, run_key="B")
    assert (tmp_path / "timing.json").read_text() == before


def test_cli_refuses_progress_of_another_seed_with_exit_2_and_no_traceback(tmp_path, capsys):
    # Breaks if `run_eval --seed 1` over a seed-0 progress dir exits 0 with "0 generations".
    progress = tmp_path / "pe_h"
    code = run_eval.main(["--fake", "habit", "--progress-dir", str(progress), "--out",
                          str(tmp_path / "s0.json"), "--n-resamples", "20"])
    assert code == 0
    capsys.readouterr()
    out = tmp_path / "s1.json"
    code = run_eval.main(["--fake", "oracle", "--seed", "1", "--progress-dir", str(progress),
                          "--out", str(out), "--n-resamples", "20"])
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "fresh --progress-dir" in err
    assert str(progress / "base.jsonl") in err
    assert "Traceback" not in err


def test_cli_refuses_a_second_adapter_in_one_progress_dir(tmp_path, capsys):
    # Breaks if run_eval does not pass the adapter into the run fingerprint.
    argv = ["--fake", "oracle", "--which", "tuned", "--progress-dir", str(tmp_path / "pe_t"),
            "--n-resamples", "20"]
    assert run_eval.main([*argv, "--adapter", "A", "--out", str(tmp_path / "a.json")]) == 0
    capsys.readouterr()
    code = run_eval.main([*argv, "--adapter", "B", "--out", str(tmp_path / "b.json")])
    assert code == 2
    assert not (tmp_path / "b.json").exists()
    assert "fresh --progress-dir" in capsys.readouterr().err
    assert run_eval.main([*argv, "--adapter", "A", "--out", str(tmp_path / "a2.json")]) == 0


# ---- --hardness through the CLI (E-1) -------------------------------------------------------

# First held-out task of make_splits(0, 1): hardness 1 rewrites the parameter descriptions.
H1_AMOUNT_DESC = "Signed change to the range; positive values raise it, negative values lower it."
H0_AMOUNT_DESC = "Signed change to the range."


def spy_on_eval_tasks(monkeypatch):
    seen = {}
    real = run_eval.evaluate_variants

    def spy(tasks, *args, **kwargs):
        seen["tasks"] = list(tasks)
        return real(tasks, *args, **kwargs)

    monkeypatch.setattr(run_eval, "evaluate_variants", spy)
    return seen


def test_cli_hardness_1_evaluates_on_the_hardness_1_split(tmp_path, monkeypatch):
    # Breaks if run_eval ignores --hardness and always builds make_splits(seed) at hardness 0.
    seen = spy_on_eval_tasks(monkeypatch)
    _, code = fake_run(tmp_path, "--hardness", "1")
    assert code == 0
    assert seen["tasks"][0].id == "heldout-0000"
    assert seen["tasks"][0].tool.doc.endswith(f"- amount (integer): {H1_AMOUNT_DESC}")


def test_cli_default_hardness_is_0(tmp_path, monkeypatch):
    seen = spy_on_eval_tasks(monkeypatch)
    _, code = fake_run(tmp_path)
    assert code == 0
    assert seen["tasks"][0].tool.doc.endswith(f"- amount (integer): {H0_AMOUNT_DESC}")


@pytest.mark.parametrize("value", ["-1", "99"])
def test_cli_rejects_a_hardness_outside_0_to_3_with_exit_2(tmp_path, capsys, value):
    # Breaks if an out-of-range hardness is passed on to make_splits instead of a parser error.
    with pytest.raises(SystemExit) as raised:
        fake_run(tmp_path, "--hardness", value)
    assert raised.value.code == 2
    assert "--hardness must be between 0 and 3" in capsys.readouterr().err


def write_splits_file(path, seed, hardness):
    from alien.splits import make_splits, splits_to_json

    path.write_text(splits_to_json(make_splits(seed, hardness)))
    return path


def test_cli_accepts_a_splits_file_made_at_the_same_hardness(tmp_path, monkeypatch):
    # Breaks if the --splits path stops matching the file's recorded hardness to --hardness.
    splits = write_splits_file(tmp_path / "s.json", 0, 1)
    seen = spy_on_eval_tasks(monkeypatch)
    _, code = fake_run(tmp_path, "--splits", str(splits), "--hardness", "1")
    assert code == 0
    assert seen["tasks"][0].tool.doc.endswith(f"- amount (integer): {H1_AMOUNT_DESC}")


def test_cli_rejects_a_splits_file_of_another_hardness_naming_both_values(tmp_path, capsys):
    # Breaks if a hardness-1 splits file is silently evaluated as a hardness-0 run.
    splits = write_splits_file(tmp_path / "s.json", 0, 1)
    with pytest.raises(SystemExit) as raised:
        fake_run(tmp_path, "--splits", str(splits))
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "was made with seed 0 and hardness 1 but this run uses seed 0 and hardness 0" in err
    assert "Traceback" not in err
    assert not (tmp_path / "eval_baselines.json").exists()


def test_cli_rejects_a_splits_file_of_another_seed(tmp_path, capsys):
    splits = write_splits_file(tmp_path / "s.json", 1, 0)
    with pytest.raises(SystemExit) as raised:
        fake_run(tmp_path, "--splits", str(splits), "--seed", "0")
    assert raised.value.code == 2
    assert "was made with seed 1 and hardness 0 but this run uses seed 0" in (
        capsys.readouterr().err)


def test_cli_refuses_a_hardness_1_run_over_hardness_0_progress_with_exit_2(tmp_path, capsys):
    # Breaks if hardness is missing from the run fingerprint (run_key) AND the prompts happen
    # to match: the hardness-0 outputs would then be replayed as hardness-1 results.
    progress = tmp_path / "pe_hard"
    assert run_eval.main(["--fake", "habit", "--progress-dir", str(progress), "--out",
                          str(tmp_path / "h0.json"), "--n-resamples", "20"]) == 0
    capsys.readouterr()
    out = tmp_path / "h1.json"
    code = run_eval.main(["--fake", "habit", "--hardness", "1", "--progress-dir", str(progress),
                          "--out", str(out), "--n-resamples", "20"])
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "fresh --progress-dir" in err
    assert "Traceback" not in err


def test_run_key_carries_the_hardness(tmp_path, monkeypatch):
    # The prompts of a hardness-0 and hardness-1 doc differ, so only the run_key itself can
    # show that hardness was put into the fingerprint: read it from the call.
    keys = []
    real = run_eval.evaluate_variants

    def spy(tasks, generators, progress_dir, **kwargs):
        keys.append(json.loads(kwargs["run_key"]))
        return real(tasks, generators, progress_dir, **kwargs)

    monkeypatch.setattr(run_eval, "evaluate_variants", spy)
    fake_run(tmp_path, "--hardness", "2")
    assert keys[0]["hardness"] == 2


# ---- the run_key helper and the leak check on imported splits (review findings 2 and 3) -----


def test_make_run_key_is_the_sorted_json_run_eval_always_used():
    # Breaks if the helper changes shape: every progress dir written by run_eval would then
    # be refused as stale.
    assert run_eval.make_run_key(
        model_id="m/x", revision="main", resolved_revision="abc", adapter="u/ad",
        hardness=2) == (
        '{"adapter": "u/ad", "decoding": {"do_sample": false, "max_new_tokens": 256, '
        '"temperature": 0.0}, "fake": null, "hardness": 2, "model_id": "m/x", '
        '"resolved_revision": "abc", "revision": "main"}')


def write_leaky_splits_file(path):
    from alien.splits import Splits, splits_to_json

    real = make_splits(0)
    leaky = Splits([*real.train, real.heldout[0]], real.dev, real.heldout, real.heldout_twist,
                   0, 0)  # one held-out task appended to train: 601 train tasks
    path.write_text(splits_to_json(leaky))
    return path


def test_cli_rejects_a_splits_file_that_fails_the_leak_check_with_one_plain_line(
        tmp_path, capsys):
    # Breaks if load_splits_checked stops calling assert_no_leak on an imported file: a run
    # would train or evaluate with a held-out task in train and exit 0.
    splits = write_leaky_splits_file(tmp_path / "leaky.json")
    with pytest.raises(SystemExit) as raised:
        fake_run(tmp_path, "--splits", str(splits))
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith(f"error: {splits} fails the leak check: held-out tools in train: ")
    assert err.count("\n") == 1 and "Traceback" not in err
    assert not (tmp_path / "eval_baselines.json").exists()
