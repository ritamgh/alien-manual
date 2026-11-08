"""Tests for the rejection-sampling fallback script (DEC-1, R-3, E-2, E-6, B-2, B-3).

No torch, no GPU, no network. Sampling is replaced by a fake generate function that answers from
the tasks themselves, and the trainer by a fake with the same small interface; selection, row
building, the reward signal, resume, upload and the results file all run for real. Each group
names the production change that would make it fail.
"""

import argparse
import functools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import check_results
import train_grpo
import train_rft
import train_sft
from alien.eval import build_results
from alien.oracle import render_prompt
from alien.persist import ProgressStore, latest_checkpoint, write_checkpoint
from alien.splits import Splits, make_splits, splits_to_json

SCRIPT = Path(train_rft.__file__)
HEAVY = ("torch", "transformers", "peft", "trl", "huggingface_hub", "datasets")
TASKS = make_splits(0).train
NOPE = '{"name": "nope", "arguments": {}}'


def metric(value: float) -> dict:
    return {"value": value, "ci_low": max(0.0, value - 0.05), "ci_high": min(1.0, value + 0.05),
            "n": 200}


def metrics(trap: float, f1: float, normal: float = 0.8) -> dict:
    return {"prior_trap_rate": metric(trap), "tool_call_f1": metric(f1),
            "tool_call_validity": metric(0.9), "normal_tool_f1": metric(normal)}


def write_prior_files(tmp_path: Path, baselines_seed: int = 0) -> None:
    evaluated = {"models": {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4)},
                 "gpu_seconds": 1200.0, "n_examples": 200}
    (tmp_path / "eval_baselines.json").write_text(json.dumps(build_results(
        evaluated, seed=baselines_seed, model_revision="rev9",
        created_at="2026-11-07T18:30:00Z")))
    sft_models = dict(evaluated["models"], sft=metrics(0.2, 0.8))
    (tmp_path / "sft.json").write_text(json.dumps(train_sft.build_sft_results(
        evaluated={"models": sft_models}, loss_first=2.5, loss_last=0.4, steps=76,
        hub_repo="someone/alien-sft", revision="abc123", model_revision="rev9",
        seed=0, git_sha="deadbeef", created_at="2026-11-07T18:30:00Z")))


def spread(call: dict, n: int) -> list[str]:
    """``n`` distinct texts that all parse to the same call (different JSON indentation)."""
    return [json.dumps(call, indent=i) for i in range(1, n + 1)]


# ---- hyper-parameters -----------------------------------------------------------------------


def test_rft_hparams_are_a_one_epoch_low_learning_rate_pass():
    hp = train_rft.RFT_HPARAMS
    assert hp["learning_rate"] == 1e-4
    assert hp["num_train_epochs"] == 1
    assert hp["samples_per_task"] == 8
    assert hp["max_per_task"] == 2
    assert hp["fp16"] is True and hp["save_steps"] == 50
    assert not any(key.startswith("lora_") for key in hp)  # the adapter is continued, not new
    smoke = train_rft.plan_rft_hparams(seed=4, smoke=True)
    assert (smoke["seed"], smoke["max_steps"], smoke["samples_per_task"]) == (4, 4, 4)
    assert train_rft.RFT_HPARAMS["samples_per_task"] == 8  # the constant is not mutated


# ---- select_by_reward -----------------------------------------------------------------------


def test_select_keeps_every_distinct_correct_completion_up_to_the_cap():
    task = TASKS[0]
    plain, spaced = json.dumps(task.oracle_call), json.dumps(task.oracle_call, indent=1)
    fenced = "```json\n" + plain + "\n```"
    kept = train_rft.select_by_reward(
        [task], [[plain, plain + "\n", spaced, fenced]], max_per_task=3)
    assert kept == [(task, plain), (task, spaced), (task, fenced)]


def test_select_with_no_correct_completion_keeps_nothing():
    trap = next(t for t in TASKS if t.is_trap)
    two_calls = NOPE + NOPE
    habit = json.dumps(trap.habit_call)  # the prior-trap answer is wrong, so reward 0
    assert train_rft.select_by_reward(
        [trap, TASKS[1]], [[habit, "garbage"], [NOPE, two_calls, ""]]) == []


def test_select_keeps_at_most_m_per_task_and_the_first_ones_sampled():
    task = TASKS[0]
    texts = spread(task.oracle_call, 5)
    assert [t for _, t in train_rft.select_by_reward([task], [texts], max_per_task=2)] \
        == texts[:2]
    assert [t for _, t in train_rft.select_by_reward([task], [texts], max_per_task=1)] \
        == texts[:1]


def test_select_skips_wrong_completions_between_correct_ones_and_keeps_task_order():
    a, b = TASKS[0], TASKS[1]
    right_a, right_b = json.dumps(a.oracle_call), json.dumps(b.oracle_call)
    kept = train_rft.select_by_reward([a, b], [["bad", right_a], [right_b, "bad"]])
    assert kept == [(a, right_a), (b, right_b)]


def test_select_with_a_wrong_number_of_completion_lists_names_the_mismatch():
    with pytest.raises(ValueError, match="2 tasks but 1 lists of completions"):
        train_rft.select_by_reward(TASKS[:2], [["x"]])


def test_build_rft_rows_pair_the_eval_prompt_with_the_kept_completion():
    task = TASKS[0]
    text = json.dumps(task.oracle_call)
    assert train_rft.build_rft_rows([(task, text)]) == [{"messages": [
        {"role": "user", "content": render_prompt(task)},
        {"role": "assistant", "content": text}]}]


# ---- sampling and the reward signal ---------------------------------------------------------


def test_sample_candidates_returns_k_completions_per_task_in_task_order():
    prompts_seen: list[list[str]] = []

    def generate(prompts):
        prompts_seen.append(list(prompts))
        return [f"out{len(prompts_seen)}-{i}" for i in range(len(prompts))]

    got = train_rft.sample_candidates(TASKS[:2], generate, 3, batch_size=4)
    assert got == [["out1-0", "out1-1", "out1-2"], ["out1-3", "out2-0", "out2-1"]]
    assert [len(batch) for batch in prompts_seen] == [4, 2]
    assert prompts_seen[0][:3] == [render_prompt(TASKS[0])] * 3


def test_an_interrupted_sampling_run_resumes_without_regenerating_or_losing_samples(tmp_path):
    path = tmp_path / "samples.jsonl"
    calls = {"n": 0}

    def dies_in_second_batch(prompts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("session lost")
        return ["first-batch"] * len(prompts)

    with pytest.raises(RuntimeError):
        train_rft.sample_candidates(TASKS[:2], dies_in_second_batch, 3, store=ProgressStore(path),
                                    batch_size=4)
    regenerated: list[int] = []

    def after_restart(prompts):
        regenerated.append(len(prompts))
        return ["second-batch"] * len(prompts)

    got = train_rft.sample_candidates(TASKS[:2], after_restart, 3, store=ProgressStore(path),
                                      batch_size=4)
    assert regenerated == [2]  # only the two missing samples were generated again
    assert got == [["first-batch"] * 3, ["first-batch", "second-batch", "second-batch"]]


def test_reward_signal_counts_groups_and_the_ones_that_vary():
    mixed, never = TASKS[0], TASKS[1]
    right = json.dumps(mixed.oracle_call)
    assert train_rft.reward_signal(
        [mixed, never], [[right, "x", "x", "x"], ["x", "x", "x", "x"]], 4) \
        == {"groups_total": 2, "groups_with_variance": 1}


# ---- training, resume, upload ---------------------------------------------------------------


def make_args(**overrides) -> argparse.Namespace:
    values = dict(hub_repo="someone/alien-rft", seed=0, smoke=False, hardness=0,
                  revision="main", sft_adapter="/sft")
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeTrainer:
    def __init__(self, steps: int = 12):
        self.state = SimpleNamespace(global_step=steps)
        self.model = SimpleNamespace(save_pretrained=lambda d: (Path(d) / "adapter.bin")
                                     .write_text("weights"))
        self.resumed = "never called"
        self.trained = False

    def train(self, resume_from_checkpoint=None):
        self.resumed = resume_from_checkpoint
        self.trained = True


EXTRA = {"reward_signal": {"groups_total": 6, "groups_with_variance": 2}}


def fake_train(tmp_path, trainers, upload=lambda folder: "abc123", rows=None, **arg_overrides):
    def make_trainer(args, rows, ckpt_root, trainer_dir, hparams):
        trainers.append(FakeTrainer())
        return trainers[-1], "rev9"

    return train_rft.train(
        make_args(**arg_overrides), rows or [], dict(EXTRA), tmp_path / "ckpt",
        tmp_path / "trainer", train_rft.plan_rft_hparams(0, False), make_trainer=make_trainer,
        upload=upload)


def test_training_saves_then_uploads_and_stores_the_reward_signal(tmp_path):
    trainers: list[FakeTrainer] = []
    uploaded = []
    outcome = fake_train(tmp_path, trainers, upload=lambda f: uploaded.append(f) or "abc123")
    assert outcome["code"] == 0 and outcome["revision"] == "abc123"
    assert outcome["summary"] == {"steps": 12, "model_revision": "rev9", **EXTRA}
    assert uploaded == [outcome["final_dir"]] == [latest_checkpoint(tmp_path / "ckpt")]
    assert trainers[0].resumed is None


def test_training_resumes_from_the_newest_checkpoint(tmp_path):
    write_checkpoint(tmp_path / "ckpt", "step-50", lambda d: (d / "adapter.bin").write_text("w"))
    trainers: list[FakeTrainer] = []
    fake_train(tmp_path, trainers)
    assert trainers[0].resumed == str(tmp_path / "ckpt" / "step-50")


def test_upload_failure_keeps_the_adapter_and_returns_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train_rft, "finish_and_upload", functools.partial(
        train_sft.finish_and_upload, base_delay=0.0, sleep=lambda s: None))

    def failing_upload(folder):
        raise ConnectionError("down")

    outcome = fake_train(tmp_path, [], upload=failing_upload)
    assert outcome["code"] == 1
    assert (latest_checkpoint(tmp_path / "ckpt") / "adapter.bin").read_text() == "weights"
    assert "nothing was deleted" in capsys.readouterr().out


def finished_final(ckpt_root, with_config=True, **arg_overrides):
    """A manifest-listed ``final`` checkpoint as an earlier run with these args left it."""
    args = make_args(**arg_overrides)
    config = train_sft.build_run_config(
        "train_rft", args, train_rft.plan_rft_hparams(args.seed, False),
        sft_adapter=args.sft_adapter)

    def write_final(directory):
        (directory / "adapter.bin").write_text("weights")
        (directory / "training_summary.json").write_text(
            json.dumps({"steps": 12, "model_revision": "rev9", **EXTRA}))
        if with_config:
            train_sft.write_run_config(directory, config, "rev9")

    write_checkpoint(ckpt_root, "final", write_final)


def test_a_rerun_with_a_complete_final_checkpoint_skips_training(tmp_path):
    finished_final(tmp_path / "ckpt")
    trainers: list[FakeTrainer] = []
    outcome = fake_train(tmp_path, trainers, upload=lambda f: "r2")
    assert trainers == []  # no trainer was even built
    assert outcome["code"] == 0 and outcome["revision"] == "r2"
    assert outcome["summary"]["reward_signal"] == EXTRA["reward_signal"]


# ---- the whole command, with the GPU steps replaced -----------------------------------------


@pytest.fixture
def small_splits(tmp_path):
    """Six train tasks so a full (non-smoke) run samples 6 x 8 completions, not 4800."""
    real = make_splits(0)
    path = tmp_path / "splits.json"
    path.write_text(splits_to_json(Splits(real.train[:6], [], real.heldout[:2],
                                          real.heldout_twist, 0, 0)))
    return path


def oracle_sampler(tasks, correct_ids):
    """A fake ``make_hf_sampler``: varied correct answers for some tasks, junk for the rest."""
    by_prompt = {render_prompt(t): t for t in tasks}
    counter = {"n": 0}

    def make_sampler(model_id, revision, adapter, hparams):
        def generate(prompts):
            out = []
            for prompt in prompts:
                task = by_prompt[prompt]
                counter["n"] += 1
                if task.id in correct_ids:
                    out.append(json.dumps(task.oracle_call, indent=counter["n"] % 3 + 1))
                else:
                    out.append("garbage")
            return out
        return generate

    return make_sampler


def run_main(tmp_path, small_splits, *extra):
    return train_rft.main(["--hub-repo", "someone/alien-rft", "--seed", "0",
                           "--work-dir", str(tmp_path / "w"), "--splits", str(small_splits),
                           "--baselines", str(tmp_path / "eval_baselines.json"),
                           "--sft-results", str(tmp_path / "sft.json"),
                           "--sft-adapter", str(tmp_path), *extra])


def test_a_full_run_writes_a_valid_rft_json_that_says_rft_ran(
        tmp_path, small_splits, capsys, monkeypatch):
    write_prior_files(tmp_path)
    tasks = make_splits(0).train[:6]
    correct = {t.id for t in tasks[:4]}
    monkeypatch.setattr(train_rft, "make_hf_sampler", oracle_sampler(tasks, correct))
    trained_rows = []

    def make_trainer(args, rows, ckpt_root, trainer_dir, hparams):
        trained_rows.extend(rows)
        return FakeTrainer(steps=9), "rev9"

    monkeypatch.setattr(train_rft, "build_rft_trainer", make_trainer)
    monkeypatch.setattr(train_rft, "hub_upload", lambda folder, repo, message: "def456")
    monkeypatch.setattr(train_grpo, "evaluate_tuned", lambda *a, **k: {
        "models": {"tuned": metrics(0.1, 0.9)}, "gpu_seconds": 300.0})
    out = tmp_path / "rft.json"
    assert run_main(tmp_path, small_splits, "--out", str(out)) == 0

    printed = capsys.readouterr().out
    assert "kept 8 completions from 4 tasks; skipped 2 tasks with no correct sample" in printed
    assert len(trained_rows) == 8
    assert all(json.loads(r["messages"][1]["content"]) for r in trained_rows)
    doc = json.loads(out.read_text())
    assert check_results.validate(doc) == []
    assert (doc["kind"], doc["path"], doc["generations_per_prompt"]) == ("rft", "rft", 8)
    assert doc["checkpoint"] == {"hub_repo": "someone/alien-rft", "revision": "def456"}
    # 6 groups of 8 samples; the 2 tasks with only junk answers and the 4 with all answers
    # correct are all single-valued, so nothing varied.
    assert doc["reward_signal"] == {"groups_total": 6, "groups_with_variance": 0}
    assert sorted(doc["models"]) == ["base", "prompt_only", "sft", "tuned"]


def test_zero_surviving_rows_exit_4_without_training_or_a_results_file(
        tmp_path, small_splits, capsys, monkeypatch):
    write_prior_files(tmp_path)
    tasks = make_splits(0).train[:6]
    monkeypatch.setattr(train_rft, "make_hf_sampler", oracle_sampler(tasks, set()))
    monkeypatch.setattr(train_rft, "train", lambda *a, **k: pytest.fail("nothing to train on"))
    out = tmp_path / "rft.json"
    assert run_main(tmp_path, small_splits, "--out", str(out)) == 4
    captured = capsys.readouterr()
    assert "kept 0 completions from 0 tasks; skipped 6 tasks" in captured.out
    assert "nothing to train on" in captured.err and "tuned_path 'sft'" in captured.err
    assert not out.exists()


def test_a_rerun_after_a_complete_final_checkpoint_does_not_sample_again(
        tmp_path, small_splits, monkeypatch):
    write_prior_files(tmp_path)
    finished_final(tmp_path / "w" / "rft" / "ckpt", sft_adapter=str(tmp_path))
    monkeypatch.setattr(train_rft, "make_hf_sampler",
                        lambda *a: pytest.fail("a finished run must not sample again"))
    monkeypatch.setattr(train_rft, "hub_upload", lambda folder, repo, message: "r2")
    monkeypatch.setattr(train_grpo, "evaluate_tuned", lambda *a, **k: {
        "models": {"tuned": metrics(0.1, 0.9)}, "gpu_seconds": 300.0})
    out = tmp_path / "rft.json"
    assert run_main(tmp_path, small_splits, "--out", str(out)) == 0
    doc = json.loads(out.read_text())
    assert doc["reward_signal"] == {"groups_total": 6, "groups_with_variance": 2}
    assert doc["checkpoint"]["revision"] == "r2"


def test_baselines_made_with_another_seed_stop_the_run_before_sampling(
        tmp_path, small_splits, capsys, monkeypatch):
    write_prior_files(tmp_path, baselines_seed=1)
    monkeypatch.setattr(train_rft, "make_hf_sampler",
                        lambda *a: pytest.fail("sampling started"))
    for name in HEAVY:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert run_main(tmp_path, small_splits) == 2
    assert "was made with seed 1 but this run uses seed 0" in capsys.readouterr().err
    assert not any(name in sys.modules for name in HEAVY)


# ---- dry run, secrets, imports --------------------------------------------------------------


def test_dry_run_prints_the_plan_exits_zero_and_imports_no_heavy_library(
        monkeypatch, capsys, tmp_path):
    for name in HEAVY:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert train_rft.main(["--dry-run", "--work-dir", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "torch" not in sys.modules
    assert "train tasks to sample: 600" in out
    assert "samples per task: 8, keep at most 2 correct per task" in out
    assert '"learning_rate": 0.0001' in out
    assert not (tmp_path / "w").exists()
    assert not any(name in sys.modules for name in HEAVY)


def test_the_script_contains_no_token_shaped_text():
    assert re.search(r"hf_[A-Za-z0-9]{20,}", SCRIPT.read_text(encoding="utf-8")) is None


def test_heavy_libraries_are_imported_only_inside_functions():
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        if re.match(r"(import|from)\s+(torch|transformers|peft|trl|huggingface_hub|datasets)\b",
                    line):
            pytest.fail(f"module-level heavy import: {line}")


# ---- --hardness (E-1) -----------------------------------------------------------------------

H1_TRAIN_NOTE = ("Registered as `query_ponthem`; the older name `read_ponthem` is retired and "
                 "rejected.")
H0_TRAIN_NOTE = ("Note: this tool is registered as `query_ponthem`; the name `read_ponthem` is "
                 "retired and rejected. Always call `query_ponthem`.")


def spy_on_rft_splits(monkeypatch):
    seen = {}
    real = train_rft.train_tasks

    def spy(splits, n=None):
        seen["splits"] = splits
        return real(splits, n)

    monkeypatch.setattr(train_rft, "train_tasks", spy)
    return seen


def test_hardness_1_samples_tasks_of_the_hardness_1_split(monkeypatch, tmp_path):
    # Breaks if train_rft ignores --hardness and calls make_splits(seed) at hardness 0.
    seen = spy_on_rft_splits(monkeypatch)
    assert train_rft.main(["--dry-run", "--hardness", "1", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 1
    assert seen["splits"].train[0].id == "train-0000"
    assert H1_TRAIN_NOTE in seen["splits"].train[0].tool.doc
    assert H0_TRAIN_NOTE not in seen["splits"].train[0].tool.doc


def test_default_hardness_builds_the_hardness_0_split(monkeypatch, tmp_path):
    seen = spy_on_rft_splits(monkeypatch)
    assert train_rft.main(["--dry-run", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 0
    assert H0_TRAIN_NOTE in seen["splits"].train[0].tool.doc


@pytest.mark.parametrize("value", ["-1", "99"])
def test_hardness_outside_0_to_3_is_a_parser_error_with_exit_2(capsys, tmp_path, value):
    with pytest.raises(SystemExit) as raised:
        train_rft.main(["--dry-run", "--hardness", value, "--work-dir", str(tmp_path)])
    assert raised.value.code == 2
    assert "--hardness must be between 0 and 3" in capsys.readouterr().err


def test_a_splits_file_is_accepted_at_its_own_hardness_and_rejected_at_another(
        tmp_path, capsys, monkeypatch):
    # Breaks if --splits stops checking the file's recorded hardness against --hardness.
    path = tmp_path / "splits.json"
    path.write_text(splits_to_json(make_splits(0, 1)))
    seen = spy_on_rft_splits(monkeypatch)
    argv = ["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")]
    assert train_rft.main([*argv, "--hardness", "1"]) == 0
    assert seen["splits"].hardness == 1
    with pytest.raises(SystemExit) as raised:
        train_rft.main(argv)
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "was made with seed 0 and hardness 1 but this run uses seed 0 and hardness 0" in err
    assert "Traceback" not in err


def test_rft_dry_run_rejects_a_splits_file_that_fails_the_leak_check(tmp_path, capsys):
    # Breaks if train_rft accepts an imported splits file with a held-out task in train
    # (reviewer saw dry runs exit 0 with 601 train tasks).
    real = make_splits(0)
    path = tmp_path / "leaky.json"
    path.write_text(splits_to_json(Splits(
        [*real.train, real.heldout[0]], real.dev, real.heldout, real.heldout_twist, 0, 0)))
    with pytest.raises(SystemExit) as raised:
        train_rft.main(["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert f"error: {path} fails the leak check: held-out tools in train: " in err
    assert "Traceback" not in err


# ---- a finished checkpoint is reused only by the same run (review finding 4) ---------------


def test_a_final_checkpoint_of_another_seed_hardness_or_base_revision_is_refused(
        tmp_path, capsys):
    # Breaks if train_rft reuses a manifest-listed final checkpoint without comparing the run.
    finished_final(tmp_path / "ckpt", seed=0, hardness=0, revision="OLD_BASE")
    trainers: list[FakeTrainer] = []
    outcome = fake_train(tmp_path, trainers, seed=1, hardness=3, revision="NEW_BASE",
                         upload=lambda f: pytest.fail("a refused run must not upload"))
    assert outcome == {"code": 2, "revision": None, "summary": None, "final_dir": None}
    assert trainers == []
    assert capsys.readouterr().err == (
        f"error: {tmp_path / 'ckpt' / 'final'} was made by a different run (differs in: "
        "hardness, revision, seed); use a fresh --work-dir\n")


def test_a_final_checkpoint_with_no_recorded_config_is_refused(tmp_path, capsys):
    # Breaks if an older checkpoint (no run_config.json) is trusted instead of refused.
    finished_final(tmp_path / "ckpt", with_config=False)
    outcome = fake_train(tmp_path, [], upload=lambda f: pytest.fail("must not upload"))
    assert outcome["code"] == 2
    assert "has no run_config.json (written by older code); use a fresh --work-dir" in (
        capsys.readouterr().err)


def test_a_trained_final_checkpoint_records_the_rft_config_and_the_same_run_reuses_it(tmp_path):
    # Breaks if train_rft does not write run_config.json beside the adapter.
    outcome = fake_train(tmp_path, [])
    recorded = json.loads((outcome["final_dir"] / "run_config.json").read_text())
    assert recorded["script"] == "train_rft"
    assert (recorded["seed"], recorded["hardness"], recorded["revision"]) == (0, 0, "main")
    assert (recorded["resolved_revision"], recorded["sft_adapter"]) == ("rev9", "/sft")
    assert (recorded["hparams.samples_per_task"], recorded["hparams.learning_rate"]) == (8, 1e-4)
    trainers: list[FakeTrainer] = []
    again = fake_train(tmp_path, trainers, upload=lambda f: "r2")
    assert trainers == [] and again["code"] == 0 and again["revision"] == "r2"


def test_main_refuses_a_finished_rft_run_when_the_hardness_flag_changed(
        tmp_path, small_splits, capsys, monkeypatch):
    # Breaks if main reuses the finished checkpoint after --hardness changed (the splits file
    # is rebuilt at hardness 1 here, as a user rerunning with the new flag would do).
    write_prior_files(tmp_path)
    finished_final(tmp_path / "w" / "rft" / "ckpt", sft_adapter=str(tmp_path))
    monkeypatch.setattr(train_rft, "make_hf_sampler",
                        lambda *a: pytest.fail("a finished run must not sample again"))
    monkeypatch.setattr(train_rft, "hub_upload", lambda *a: pytest.fail("must not upload"))
    splits = tmp_path / "h1.json"
    real = make_splits(0, 1)
    splits.write_text(splits_to_json(Splits(real.train[:6], [], real.heldout[:2],
                                            real.heldout_twist, 0, 1)))
    assert run_main(tmp_path, splits, "--hardness", "1") == 2
    assert "differs in: hardness); use a fresh --work-dir" in capsys.readouterr().err
