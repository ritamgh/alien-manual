"""Tests for the results-report builder (PRD S5, FR-006, SC-001, E-3, R-4).

Everything runs on fixtures built in the test from the real splits, oracle and scorer: no
model, no GPU, no real results file. Each group names the production change that would make
it fail.
"""

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import build_report
import check_results
from alien.eval import BASELINE_MODELS, ResultsInvalidError, build_results, write_results
from alien.oracle import call_to_json
from alien.persist import ProgressStore
from alien.scoring import ExampleScore, evaluate
from alien.splits import make_splits
from train_sft import build_sft_results

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_report.py"
RESAMPLES = 50


@pytest.fixture(scope="module")
def splits():
    return make_splits(0)


@pytest.fixture(scope="module")
def tasks(splits):
    return splits.heldout[:200]  # 162 trap tasks and 38 normal tasks for seed 0


def oracle_outputs(tasks):
    return [call_to_json(t.oracle_call) for t in tasks]


def habit_outputs(tasks):
    return [call_to_json(t.habit_call) for t in tasks]


def scores_for(tasks, outputs):
    return build_report.score_all(tasks, dict(zip([t.id for t in tasks], outputs)))


def metric(value, low=None, high=None, n=200):
    return {"value": value, "ci_low": value if low is None else low,
            "ci_high": value if high is None else high, "n": n}


def metrics(prior, f1, normal=0.9):
    return {"prior_trap_rate": metric(prior), "tool_call_f1": metric(f1),
            "tool_call_validity": metric(1.0), "normal_tool_f1": metric(normal)}


def rl_doc(path, f1, prior=0.1, *, groups=(100, 40)):
    return {
        "kind": path, "schema_version": 1, "created_at": "2026-11-07T18:30:00Z",
        "git_sha": None, "seed": 0, "path": path,
        "checkpoint": {"hub_repo": "user/alien-rl", "revision": "r2"},
        "generations_per_prompt": 4,
        "reward_signal": {"groups_total": groups[0], "groups_with_variance": groups[1]},
        "models": {"base": metrics(0.8, 0.4), "prompt_only": metrics(0.7, 0.5),
                   "sft": metrics(0.3, 0.7), "tuned": metrics(prior, f1)},
    }


def sft_doc_literal():
    return {"models": {"base": metrics(0.8, 0.4), "prompt_only": metrics(0.7, 0.5),
                       "sft": metrics(0.3, 0.7)}}


# ---- load_outputs and score_all -------------------------------------------------------------
# Break caught: reading the wrong key, or not reading ProgressStore's real file format.


def test_load_outputs_reads_a_real_progress_store_file(tmp_path):
    path = tmp_path / "progress" / "base.jsonl"
    store = ProgressStore(path)
    store.record("heldout-0000", {"output": '{"name": "a", "arguments": {}}'})
    store.record("heldout-0001", {"output": "no call here"})
    assert build_report.load_outputs(path) == {
        "heldout-0000": '{"name": "a", "arguments": {}}',
        "heldout-0001": "no call here",
    }


def test_load_outputs_rejects_a_record_without_an_output_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    ProgressStore(path).record("heldout-0000", {"text": "x"})
    with pytest.raises(build_report.ReportInputError, match="heldout-0000"):
        build_report.load_outputs(path)


def test_score_all_scores_in_task_order_with_literal_counts(tasks):
    scores = scores_for(tasks, oracle_outputs(tasks))
    assert [s.task_id for s in scores] == [t.id for t in tasks]
    assert sum(s.tp for s in scores) == 200
    assert sum(s.prior_trap for s in scores) == 0
    habit = scores_for(tasks, habit_outputs(tasks))
    assert sum(s.prior_trap for s in habit) == 162
    assert sum(s.tp for s in habit) == 38


def test_score_all_names_the_missing_example_instead_of_scoring_zero(tasks):
    outputs = dict(zip([t.id for t in tasks], oracle_outputs(tasks)))
    del outputs["heldout-0007"]
    with pytest.raises(build_report.ReportInputError, match="heldout-0007"):
        build_report.score_all(tasks, outputs)


# ---- compute_sc001 --------------------------------------------------------------------------
# Break caught: wrong gap sign for prior_trap_rate, wrong baseline pairing, a loosened rule.


def by_model(tasks, tuned, base, prompt_only=None):
    return {"base": scores_for(tasks, base),
            "prompt_only": scores_for(tasks, prompt_only or base),
            "tuned": scores_for(tasks, tuned)}


def test_oracle_tuned_against_habit_base_is_met_with_literal_gaps(tasks):
    sc001 = build_report.compute_sc001(
        by_model(tasks, oracle_outputs(tasks), habit_outputs(tasks)), n_resamples=200, seed=0)
    assert sc001["verdict"] == "met"
    for baseline in ("vs_base", "vs_prompt_only"):
        trap = sc001["gaps"][baseline]["prior_trap_rate"]
        # baseline 162/162 minus tuned 0/162: an improvement of 1.0, not -1.0
        assert trap == {"gap": 1.0, "ci_low": 1.0, "ci_high": 1.0}
        f1 = sc001["gaps"][baseline]["tool_call_f1"]
        # base f1 = 2*38 / (2*38 + 162 + 162) = 0.19, tuned f1 = 1.0
        assert f1["gap"] == pytest.approx(0.81, abs=1e-9)
        assert 0.7 < f1["ci_low"] <= 0.81 <= f1["ci_high"] < 0.9


def test_vs_prompt_only_uses_the_prompt_only_scores_not_the_base_scores(tasks):
    scores = by_model(tasks, oracle_outputs(tasks), habit_outputs(tasks),
                      prompt_only=oracle_outputs(tasks))
    sc001 = build_report.compute_sc001(scores, n_resamples=100, seed=0)
    assert sc001["gaps"]["vs_base"]["prior_trap_rate"]["gap"] == 1.0
    assert sc001["gaps"]["vs_prompt_only"]["prior_trap_rate"]["gap"] == 0.0
    assert sc001["verdict"] == "not_met"


def test_identical_tuned_and_base_is_not_met_with_zero_gaps(tasks):
    outputs = habit_outputs(tasks)
    sc001 = build_report.compute_sc001(by_model(tasks, outputs, outputs), n_resamples=100)
    assert sc001["verdict"] == "not_met"
    assert sc001["gaps"]["vs_base"]["tool_call_f1"] == {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def synthetic(prior_trap_count, tp):
    """20 trap examples; the first ``prior_trap_count`` follow the habit."""
    return [ExampleScore(task_id=f"t{i}", is_trap=True, valid=True, tp=tp, fp=1 - tp,
                         fn=1 - tp, prior_trap=i < prior_trap_count) for i in range(20)]


def test_gap_of_exactly_ten_points_with_ci_touching_zero_is_not_met():
    base = synthetic(2, tp=0)  # prior-trap 2/20 = 0.10 against tuned 0/20
    tuned = synthetic(0, tp=1)
    sc001 = build_report.compute_sc001(
        {"base": base, "prompt_only": base, "tuned": tuned}, n_resamples=1000, seed=0)
    trap = sc001["gaps"]["vs_base"]["prior_trap_rate"]
    assert trap["gap"] == pytest.approx(0.10, abs=1e-12)
    assert trap["ci_low"] <= 0
    assert sc001["verdict"] == "not_met"


def test_same_shape_with_a_clear_gap_is_met():
    base = synthetic(8, tp=0)  # 0.40 against 0.0: 20 examples cannot erase it by resampling
    tuned = synthetic(0, tp=1)
    sc001 = build_report.compute_sc001(
        {"base": base, "prompt_only": base, "tuned": tuned}, n_resamples=1000, seed=0)
    assert sc001["gaps"]["vs_base"]["prior_trap_rate"]["gap"] == pytest.approx(0.40)
    assert sc001["gaps"]["vs_base"]["prior_trap_rate"]["ci_low"] > 0
    assert sc001["verdict"] == "met"


def test_zero_prior_trap_gap_is_not_met_even_with_a_huge_f1_gap():
    base = synthetic(0, tp=0)
    tuned = synthetic(0, tp=1)
    # prior-trap gap 0 (both 0/20) fails the 10-point rule even though f1 gaps are huge
    sc001 = build_report.compute_sc001(
        {"base": base, "prompt_only": base, "tuned": tuned}, n_resamples=100, seed=0)
    assert sc001["gaps"]["vs_base"]["tool_call_f1"]["ci_low"] > 0
    assert sc001["verdict"] == "not_met"


# ---- pick_tuned -----------------------------------------------------------------------------
# Break caught: choosing the wrong doc, or trusting the file name instead of the recorded path.


def test_pick_tuned_prefers_grpo_when_only_grpo_exists():
    grpo = rl_doc("grpo", 0.8)
    assert build_report.pick_tuned(grpo, None, sft_doc_literal()) == ("grpo", grpo)


def test_pick_tuned_uses_rft_when_only_rft_exists():
    rft = rl_doc("rft", 0.8)
    assert build_report.pick_tuned(None, rft, sft_doc_literal()) == ("rft", rft)


def test_pick_tuned_falls_back_to_the_sft_model_when_neither_exists():
    sft = sft_doc_literal()
    assert build_report.pick_tuned(None, None, sft) == ("sft", sft)


def test_pick_tuned_with_both_takes_the_higher_f1():
    grpo, rft = rl_doc("grpo", 0.80), rl_doc("rft", 0.85)
    assert build_report.pick_tuned(grpo, rft, sft_doc_literal())[0] == "rft"
    grpo, rft = rl_doc("grpo", 0.90), rl_doc("rft", 0.85)
    assert build_report.pick_tuned(grpo, rft, sft_doc_literal())[0] == "grpo"


def test_pick_tuned_with_equal_f1_takes_the_lower_trap_rate_then_grpo():
    grpo, rft = rl_doc("grpo", 0.8, prior=0.2), rl_doc("rft", 0.8, prior=0.1)
    assert build_report.pick_tuned(grpo, rft, sft_doc_literal())[0] == "rft"
    grpo, rft = rl_doc("grpo", 0.8, prior=0.1), rl_doc("rft", 0.8, prior=0.1)
    assert build_report.pick_tuned(grpo, rft, sft_doc_literal())[0] == "grpo"


def test_a_grpo_file_that_fell_back_is_reported_as_rft():
    fell_back = rl_doc("grpo", 0.8)
    fell_back["path"] = "rft"
    assert build_report.pick_tuned(fell_back, None, sft_doc_literal())[0] == "rft"


def test_choice_note_exists_only_when_both_files_exist():
    grpo, rft = rl_doc("grpo", 0.80), rl_doc("rft", 0.85)
    note = build_report.choice_note(grpo, rft)
    assert "rft" in note and "higher" in note
    assert build_report.choice_note(grpo, None) is None
    assert build_report.choice_note(None, rft) is None


# ---- build_final ----------------------------------------------------------------------------
# Break caught: a final doc that check_results rejects, or a verdict the gaps contradict.


def fixture_docs(tasks, *, tuned_outputs=None, sft_outputs=None, with_rl=True):
    """Valid eval_baselines, sft and grpo documents plus the four output lists."""
    base_out, prompt_out = habit_outputs(tasks), habit_outputs(tasks)
    sft_out = oracle_outputs(tasks) if sft_outputs is None else sft_outputs
    tuned_out = sft_out if tuned_outputs is None else tuned_outputs
    ev = {name: evaluate(tasks, out, n_resamples=RESAMPLES, seed=0)
          for name, out in (("base", base_out), ("prompt_only", prompt_out),
                            ("sft", sft_out), ("tuned", tuned_out))}
    baselines = build_results(
        {"models": {n: ev[n] for n in BASELINE_MODELS}, "gpu_seconds": 100.0,
         "n_examples": 200}, seed=0, model_revision="rev1", created_at="2026-11-07T18:00:00Z")
    sft = build_sft_results(
        evaluated={"models": {n: ev[n] for n in ("base", "prompt_only", "sft")}},
        loss_first=2.0, loss_last=0.5, steps=100, hub_repo="user/alien-sft", revision="r1",
        model_revision="rev1", seed=0, created_at="2026-11-07T18:10:00Z")
    grpo = rl_doc("grpo", 0.0)
    grpo["models"] = {**{n: ev[n] for n in ("base", "prompt_only", "sft")}, "tuned": ev["tuned"]}
    outputs = {"base": base_out, "prompt_only": prompt_out, "sft": sft_out, "tuned": tuned_out}
    assert check_results.validate(baselines) == []
    assert check_results.validate(sft) == []
    assert check_results.validate(grpo) == []
    return baselines, sft, (grpo if with_rl else None), outputs


def final_for(tasks, **kwargs):
    baselines, sft, grpo, outputs = fixture_docs(tasks, **kwargs)
    scores = {name: scores_for(tasks, out) for name, out in outputs.items()}
    return build_report.build_final(
        baselines, sft, grpo, None, scores, seed=0, report_path="report.md",
        n_resamples=100, created_at="2026-11-07T19:00:00Z", git_sha=None)


def test_build_final_is_a_valid_final_document_with_the_contract_fields(tasks):
    doc = final_for(tasks)
    assert check_results.validate(doc) == []
    assert doc["kind"] == "final"
    assert doc["tuned_path"] == "grpo"
    assert doc["report_path"] == "report.md"
    assert doc["seed"] == 0
    assert doc["sc001"]["verdict"] == "met"
    assert doc["models"]["base"]["tool_call_f1"]["value"] == pytest.approx(0.19)
    assert doc["models"]["tuned"]["tool_call_f1"]["value"] == 1.0


def test_build_final_without_grpo_or_rft_uses_the_sft_model_as_tuned(tasks):
    doc = final_for(tasks, with_rl=False)
    assert doc["tuned_path"] == "sft"
    assert doc["models"]["tuned"] == doc["models"]["sft"]
    assert check_results.validate(doc) == []


def test_flipping_the_verdict_makes_the_final_document_invalid(tasks):
    doc = final_for(tasks)
    flipped = copy.deepcopy(doc)
    flipped["sc001"]["verdict"] = "not_met"
    errors = check_results.validate(flipped)
    assert any(e.startswith("sc001.verdict") for e in errors)


def test_write_results_refuses_a_flipped_final_document_and_writes_nothing(tasks, tmp_path):
    doc = final_for(tasks)
    doc["sc001"]["verdict"] = "not_met"
    out = tmp_path / "final.json"
    with pytest.raises(ResultsInvalidError, match="sc001.verdict"):
        write_results(out, doc, check_results.validate)
    assert not out.exists()


def test_build_final_rejects_outputs_that_do_not_produce_the_recorded_metrics(tasks):
    baselines, sft, grpo, outputs = fixture_docs(tasks)
    scores = {name: scores_for(tasks, out) for name, out in outputs.items()}
    scores["tuned"] = scores_for(tasks, habit_outputs(tasks))  # not the model that was scored
    with pytest.raises(build_report.ReportInputError, match="tuned.*prior_trap_rate"):
        build_report.build_final(baselines, sft, grpo, None, scores, seed=0,
                                 report_path="report.md", n_resamples=50)


# ---- render_report --------------------------------------------------------------------------
# Break caught: a softened or missing verdict, a missing table, trap quote or E-3 line.


def set_normal(doc, base, tuned):
    doc = copy.deepcopy(doc)
    doc["models"]["base"]["normal_tool_f1"] = metric(base)
    doc["models"]["tuned"]["normal_tool_f1"] = metric(tuned)
    return doc


PROBE = {
    "trap_example": {
        "task_id": "dev-0001",
        "request": "Look up the record for pallet P-83.",
        "model_output": '{"name": "read_runru", "arguments": {"item": "P-83"}}',
        "habit_call": {"name": "read_runru", "arguments": {"item": "P-83"}},
        "oracle_call": {"name": "grab_runru", "arguments": {"item": "P-83"}},
    },
}
DOC_TEXT = ("Returns the stored record for a pallet. Note: this tool is registered as "
            "`grab_runru`; the name `read_runru` is retired and rejected.")


def test_met_report_starts_with_the_verdict_and_the_smallest_gap(tasks):
    text = build_report.render_report(final_for(tasks), PROBE, trap_doc=DOC_TEXT)
    first = text.splitlines()[0]
    assert first.startswith("SC-001 met")
    assert "81.0 points" in first  # the smallest gap: tool_call_f1, 1.0 - 0.19


def test_not_met_report_says_so_on_the_first_line_without_soft_words(tasks):
    doc = final_for(tasks, sft_outputs=habit_outputs(tasks), with_rl=False)  # tuned == base
    assert doc["sc001"]["verdict"] == "not_met"
    text = build_report.render_report(doc, None)
    assert text.splitlines()[0].startswith("SC-001 NOT met")
    assert "0.0 points" in text.splitlines()[0]
    lowered = text.lower()
    assert "promising" not in lowered and "trend" not in lowered


def test_report_has_the_four_metric_table_with_intervals_for_four_models(tasks):
    text = build_report.render_report(final_for(tasks), None)
    assert "| Metric | Base | Prompt-only | SFT | Tuned |" in text
    # base tool_call_f1 is exactly 0.19 with a CI around it; tuned is 100.0%
    assert re.search(r"\| Tool-call F1.*\| 19\.0% \[\d+\.\d%, \d+\.\d%\] \|", text)
    assert "100.0% [100.0%, 100.0%]" in text
    for name in ("Prior-trap rate", "Tool-call validity", "Normal-tool F1"):
        assert name in text


def test_report_has_the_gap_table_with_the_ci_in_points(tasks):
    text = build_report.render_report(final_for(tasks), None)
    assert "| Prior-trap rate | 100.0 | 100.0 to 100.0 | yes |" in text
    assert text.count("| yes |") == 4


def test_report_says_which_path_ran_and_why(tasks):
    text = build_report.render_report(final_for(tasks), None, source_doc=rl_doc("grpo", 0.9))
    assert "Path that ran: grpo" in text
    assert "40 of 100" in text
    rft = final_for(tasks)
    rft["tuned_path"] = "rft"
    text = build_report.render_report(rft, None, source_doc=rl_doc("rft", 0.9))
    assert "Path that ran: rft" in text and "rejection-sampling" in text
    sft = final_for(tasks, with_rl=False)
    text = build_report.render_report(sft, None, source_doc=None)
    assert "Path that ran: sft-only" in text


def test_report_quotes_the_trap_example_when_the_probe_is_given(tasks):
    text = build_report.render_report(final_for(tasks), PROBE, trap_doc=DOC_TEXT)
    assert "Look up the record for pallet P-83." in text
    assert '{"name": "read_runru", "arguments": {"item": "P-83"}}' in text
    assert "the name `read_runru` is retired and rejected" in text
    assert '{"name": "grab_runru", "arguments": {"item": "P-83"}}' in text


def test_report_says_the_probe_file_was_not_found_when_it_is_absent(tasks):
    text = build_report.render_report(final_for(tasks), None)
    assert "probe file was not found" in text
    assert "pallet P-83" not in text


def test_e3_line_reports_a_drop_of_more_than_five_points(tasks):
    doc = set_normal(final_for(tasks), base=0.90, tuned=0.80)
    text = build_report.render_report(doc, None)
    assert "E-3" in text and "dropped by 10.0 points" in text


def test_e3_line_reports_no_drop_beyond_the_limit_at_exactly_five_points(tasks):
    doc = set_normal(final_for(tasks), base=0.90, tuned=0.85)
    text = build_report.render_report(doc, None)
    assert "did not drop by more than 5 points" in text
    assert "dropped by" not in text


def test_report_lists_the_honest_caveats(tasks):
    text = build_report.render_report(final_for(tasks), None).lower()
    assert "200 held-out examples" in text
    assert "single seed" in text and "seed 0" in text
    assert "exact-match" in text and "different argument forms score 0" in text
    assert "invented tools" in text


# ---- command line ---------------------------------------------------------------------------
# Break caught: a CLI that writes a final.json check_results rejects, or hides a missing input.


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def write_outputs(path, tasks, outputs):
    store = ProgressStore(path)
    for task, text in zip(tasks, outputs):
        store.record(task.id, {"output": text})


def lay_out(tmp_path, tasks, *, with_rl=True, with_probe=True):
    baselines, sft, grpo, outputs = fixture_docs(tasks, with_rl=with_rl)
    results = tmp_path / "results"
    write_json(results / "eval_baselines.json", baselines)
    write_json(results / "sft.json", sft)
    if grpo is not None:
        write_json(results / "grpo.json", grpo)
    if with_probe:
        probe = {"seed": 0, **PROBE}
        write_json(results / "base_probe.json", probe)
    for name in ("base", "prompt_only"):
        write_outputs(results / "progress" / f"{name}.jsonl", tasks, outputs[name])
    write_outputs(tmp_path / "work" / "progress_sft" / "tuned.jsonl", tasks, outputs["sft"])
    write_outputs(tmp_path / "work" / "progress_grpo" / "tuned.jsonl", tasks, outputs["tuned"])
    return results


def cli_args(tmp_path, *extra, tuned=True):
    results = tmp_path / "results"
    args = [
        "--baselines", str(results / "eval_baselines.json"),
        "--sft", str(results / "sft.json"),
        "--grpo", str(results / "grpo.json"),
        "--rft", str(results / "rft.json"),
        "--probe", str(results / "base_probe.json"),
        "--out", str(results / "final.json"),
        "--report", str(tmp_path / "report.md"),
        "--seed", "0", "--n-resamples", "100",
        "--outputs", f"base={results / 'progress' / 'base.jsonl'}",
        "--outputs", f"prompt_only={results / 'progress' / 'prompt_only.jsonl'}",
        "--outputs", f"sft={tmp_path / 'work' / 'progress_sft' / 'tuned.jsonl'}",
    ]
    if tuned:
        args += ["--outputs", f"tuned={tmp_path / 'work' / 'progress_grpo' / 'tuned.jsonl'}"]
    return args + list(extra)


def test_cli_end_to_end_writes_a_final_json_that_check_results_accepts(tmp_path, tasks, capsys):
    lay_out(tmp_path, tasks)
    assert build_report.main(cli_args(tmp_path)) == 0
    out = tmp_path / "results" / "final.json"
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "check_results.py"), str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "OK final"
    doc = json.loads(out.read_text())
    assert doc["tuned_path"] == "grpo" and doc["sc001"]["verdict"] == "met"
    report = (tmp_path / "report.md").read_text()
    assert report.splitlines()[0].startswith("SC-001 met")
    assert "Path that ran: grpo" in report
    assert "pallet P-83" in report
    assert "Always call `grab_runru`." in report  # the doc text, looked up from the dev split
    assert "SC-001 met" in capsys.readouterr().out


def test_cli_without_grpo_or_rft_reports_the_sft_only_path(tmp_path, tasks):
    results = lay_out(tmp_path, tasks, with_rl=False)
    assert build_report.main(cli_args(tmp_path, tuned=False)) == 0
    doc = json.loads((results / "final.json").read_text())
    assert doc["tuned_path"] == "sft"
    assert doc["models"]["tuned"] == doc["models"]["sft"]
    assert "Path that ran: sft-only" in (tmp_path / "report.md").read_text()


def test_cli_without_a_probe_file_still_builds_and_says_it_was_not_found(tmp_path, tasks):
    lay_out(tmp_path, tasks, with_probe=False)
    assert build_report.main(cli_args(tmp_path)) == 0
    assert "probe file was not found" in (tmp_path / "report.md").read_text()


def test_cli_missing_sft_json_exits_2_naming_the_file_and_the_colab_step(
        tmp_path, tasks, capsys):
    results = lay_out(tmp_path, tasks)
    (results / "sft.json").unlink()
    assert build_report.main(cli_args(tmp_path)) == 2
    err = capsys.readouterr().err
    assert str(results / "sft.json") in err and "train_sft.py" in err
    assert not (results / "final.json").exists()


def test_cli_missing_baselines_exits_2_naming_run_eval(tmp_path, tasks, capsys):
    results = lay_out(tmp_path, tasks)
    (results / "eval_baselines.json").unlink()
    assert build_report.main(cli_args(tmp_path)) == 2
    err = capsys.readouterr().err
    assert str(results / "eval_baselines.json") in err and "run_eval.py" in err


def test_cli_missing_output_file_exits_2_naming_the_file(tmp_path, tasks, capsys):
    lay_out(tmp_path, tasks)
    missing = tmp_path / "results" / "progress" / "base.jsonl"
    missing.unlink()
    assert build_report.main(cli_args(tmp_path)) == 2
    assert str(missing) in capsys.readouterr().err


def test_cli_missing_tuned_output_when_grpo_ran_exits_2(tmp_path, tasks, capsys):
    lay_out(tmp_path, tasks)
    assert build_report.main(cli_args(tmp_path, tuned=False)) == 2
    assert "tuned" in capsys.readouterr().err


def test_cli_mismatched_output_file_exits_1_and_writes_no_final_json(tmp_path, tasks, capsys):
    results = lay_out(tmp_path, tasks)
    (results / "progress" / "base.jsonl").unlink()
    write_outputs(results / "progress" / "base.jsonl", tasks, oracle_outputs(tasks))
    assert build_report.main(cli_args(tmp_path)) == 1
    assert "base" in capsys.readouterr().err
    assert not (results / "final.json").exists()


def test_cli_rejects_an_unknown_output_name(tmp_path, tasks):
    lay_out(tmp_path, tasks)
    with pytest.raises(SystemExit) as exc:
        build_report.main(cli_args(tmp_path, "--outputs", "nope=x.jsonl"))
    assert exc.value.code == 2


def test_help_works_and_imports_nothing_heavy():
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert done.returncode == 0
    assert "--outputs" in done.stdout and "progress_sft" in done.stdout
    probe = ("import sys; sys.path[:0] = ['scripts', '.']; import build_report; "
             "assert not {'torch', 'transformers', 'peft', 'trl'} & set(sys.modules)")
    assert subprocess.run([sys.executable, "-c", probe], cwd=REPO).returncode == 0


def test_script_text_has_no_secret():
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", SCRIPT.read_text(encoding="utf-8"))


# ---- --hardness (E-1) -----------------------------------------------------------------------


def test_cli_with_hardness_1_quotes_the_hardness_1_tool_doc(tmp_path, tasks):
    # Breaks if --hardness stops reaching the dev-split lookup of the quoted tool doc.
    lay_out(tmp_path, tasks)
    assert build_report.main(cli_args(tmp_path, "--hardness", "1")) == 0
    report = (tmp_path / "report.md").read_text()
    assert "Registered as `grab_runru`; the older name `read_runru` is retired" in report
    assert "Always call `grab_runru`." not in report


def test_cli_with_hardness_1_scores_the_hardness_1_heldout_split(tmp_path, tasks, monkeypatch):
    # Breaks if the held-out tasks are still built at hardness 0 while the eval ran at 1.
    lay_out(tmp_path, tasks)
    built = []
    real = build_report.make_splits

    def spy(seed, hardness=0, *rest):
        built.append((seed, hardness))
        return real(seed, hardness, *rest)

    monkeypatch.setattr(build_report, "make_splits", spy)
    assert build_report.main(cli_args(tmp_path, "--hardness", "1")) == 0
    assert built and set(built) == {(0, 1)}


@pytest.mark.parametrize("value", ["-1", "99"])
def test_cli_rejects_a_hardness_outside_0_to_3_with_exit_2(tmp_path, tasks, capsys, value):
    # Breaks if an out-of-range --hardness silently drops the quoted tool doc (the old path).
    lay_out(tmp_path, tasks)
    with pytest.raises(SystemExit) as exc:
        build_report.main(cli_args(tmp_path, "--hardness", value))
    assert exc.value.code == 2
    assert "--hardness must be between 0 and 3" in capsys.readouterr().err
    assert not (tmp_path / "results" / "final.json").exists()


def test_cli_exits_2_when_the_probe_trap_example_is_not_in_the_dev_split(
        tmp_path, tasks, capsys):
    # Breaks if a probe whose trap example cannot be found in the rebuilt dev split goes on to
    # print "(the tool documentation is not stored in the probe file)" and exit 0.
    # The limit: the generator's ids and requests do not depend on hardness, so this cannot
    # tell a hardness-0 rebuild from a hardness-1 probe; that needs base_probe.json to record
    # its hardness (see the FIX-C handoff).
    results = lay_out(tmp_path, tasks)
    probe = json.loads((results / "base_probe.json").read_text())
    probe["trap_example"]["request"] = "Look up the record for pallet P-99."
    write_json(results / "base_probe.json", probe)
    assert build_report.main(cli_args(tmp_path)) == 2
    err = capsys.readouterr().err
    assert str(results / "base_probe.json") in err
    assert "pass the same --hardness as the probe run" in err
    assert not (results / "final.json").exists()
    assert not (tmp_path / "report.md").exists()
