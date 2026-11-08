"""Tests for the GRPO training script's CPU-testable parts (FR-003, S4, E-2, E-3, E-6, B-1..B-3).

No torch, no GPU, no network. The GPU path (``build_grpo_trainer``, ``hub_upload``,
``evaluate_tuned``) is Colab-gated; what is tested here is everything around it, with a fake
trainer that calls the real reward function and the real monitor. Each group names the
production change that would make it fail.
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
import train_sft
from alien.eval import build_results
from alien.oracle import render_prompt
from alien.persist import latest_checkpoint, load_checkpoints, write_checkpoint
from alien.splits import Splits, make_splits, splits_to_json

SCRIPT = Path(train_grpo.__file__)
HEAVY = ("torch", "transformers", "peft", "trl", "huggingface_hub", "datasets")


def metric(value: float) -> dict:
    return {"value": value, "ci_low": max(0.0, value - 0.05), "ci_high": min(1.0, value + 0.05),
            "n": 200}


def metrics(trap: float, f1: float, normal: float = 0.8) -> dict:
    return {"prior_trap_rate": metric(trap), "tool_call_f1": metric(f1),
            "tool_call_validity": metric(0.9), "normal_tool_f1": metric(normal)}


def models(tuned_normal: float = 0.8) -> dict:
    return {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4),
            "sft": metrics(0.2, 0.8), "tuned": metrics(0.1, 0.9, tuned_normal)}


class Forbidden(list):
    """A split that must never be read: any access fails the test."""

    def __iter__(self):
        raise AssertionError("this split must not be read (B-2)")

    def __len__(self):
        raise AssertionError("this split must not be read (B-2)")

    def __getitem__(self, index):
        raise AssertionError("this split must not be read (B-2)")


def write_prior_files(tmp_path: Path, baselines_seed: int = 0, sft_seed: int = 0) -> tuple:
    evaluated = {"models": {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4)},
                 "gpu_seconds": 1200.0, "n_examples": 200}
    baselines = tmp_path / "eval_baselines.json"
    baselines.write_text(json.dumps(build_results(
        evaluated, seed=baselines_seed, model_revision="rev9",
        created_at="2026-11-07T18:30:00Z")))
    sft_models = dict(evaluated["models"], sft=metrics(0.2, 0.8))
    sft = tmp_path / "sft.json"
    sft.write_text(json.dumps(train_sft.build_sft_results(
        evaluated={"models": sft_models}, loss_first=2.5, loss_last=0.4, steps=76,
        hub_repo="someone/alien-sft", revision="abc123", model_revision="rev9",
        seed=sft_seed, git_sha="deadbeef", created_at="2026-11-07T18:30:00Z")))
    return baselines, sft


def rl_doc(path: str = "grpo", **overrides) -> dict:
    kwargs = dict(path=path, evaluated={"models": models()}, hub_repo="someone/alien-grpo",
                  revision="def456", generations_per_prompt=4,
                  reward_summary={"groups_total": 80, "groups_with_variance": 12}, seed=0,
                  git_sha="deadbeef", created_at="2026-11-07T18:30:00Z")
    kwargs.update(overrides)
    return train_grpo.build_grpo_results(**kwargs)


# ---- hyper-parameters (PRD FR-003, AS-2): a changed constant fails a literal ---------------


def test_grpo_hparams_are_the_prd_values():
    hp = train_grpo.GRPO_HPARAMS
    assert hp["num_generations"] == 4
    assert hp["learning_rate"] == 5e-6
    assert hp["max_completion_length"] == 256
    assert hp["max_prompt_length"] == 512
    assert hp["fp16"] is True
    assert hp["save_steps"] == 50
    assert hp["max_steps"] == 100
    assert hp["per_device_train_batch_size"] == 4
    assert hp["per_device_train_batch_size"] % hp["num_generations"] == 0


def test_plan_grpo_hparams_smoke_shrinks_the_run_and_never_mutates_the_constant():
    normal = train_grpo.plan_grpo_hparams(seed=3, smoke=False)
    assert normal["seed"] == 3 and normal["max_steps"] == 100
    smoke = train_grpo.plan_grpo_hparams(seed=3, smoke=True)
    assert (smoke["max_steps"], smoke["save_steps"], smoke["num_generations"]) == (3, 2, 4)
    assert train_grpo.GRPO_HPARAMS["max_steps"] == 100


# ---- rows (B-2): train split only ----------------------------------------------------------


def test_build_grpo_rows_reads_only_the_train_split():
    real = make_splits(0)
    guarded = Splits(real.train, Forbidden(), Forbidden(), real.heldout_twist, 0, 0)
    rows = train_grpo.build_grpo_rows(guarded)
    assert len(rows) == 600


def test_build_grpo_rows_have_the_trl_conversational_shape():
    splits = make_splits(0)
    rows = train_grpo.build_grpo_rows(splits, n=5)
    assert len(rows) == 5
    first, task = rows[0], splits.train[0]
    assert sorted(first) == ["prompt", "task_id"]
    assert first["task_id"] == task.id
    assert [m["role"] for m in first["prompt"]] == ["user"]
    assert f"Request: {task.request}" in first["prompt"][0]["content"]


# ---- results builder (shared with train_rft) ------------------------------------------------


def test_build_grpo_results_passes_the_validator_with_the_contract_fields():
    doc = rl_doc()
    assert check_results.validate(doc) == []
    assert doc["kind"] == "grpo" and doc["path"] == "grpo"
    assert doc["generations_per_prompt"] == 4
    assert doc["reward_signal"] == {"groups_total": 80, "groups_with_variance": 12}
    assert doc["checkpoint"] == {"hub_repo": "someone/alien-grpo", "revision": "def456"}
    assert sorted(doc["models"]) == ["base", "prompt_only", "sft", "tuned"]
    assert doc["seed"] == 0 and doc["git_sha"] == "deadbeef"


def test_a_grpo_result_with_no_reward_variance_is_rejected_by_name():
    doc = rl_doc(reward_summary={"groups_total": 80, "groups_with_variance": 0})
    errors = check_results.validate(doc)
    assert any(e.startswith("reward_signal.groups_with_variance: expected >= 1 for path grpo")
               and "path rft" in e for e in errors), errors


def test_fewer_than_four_generations_under_path_grpo_is_rejected_by_name():
    errors = check_results.validate(rl_doc(generations_per_prompt=2))
    assert any(e.startswith("generations_per_prompt: expected an integer >= 4") for e in errors)


def test_the_same_builder_makes_a_valid_rft_document_even_without_variance():
    doc = rl_doc(path="rft", generations_per_prompt=8, hub_repo="someone/alien-rft",
                 reward_summary={"groups_total": 600, "groups_with_variance": 0})
    assert check_results.validate(doc) == []
    assert doc["kind"] == "rft" and doc["path"] == "rft"


def test_build_grpo_results_refuses_an_unknown_path_and_names_a_missing_model():
    with pytest.raises(ValueError, match="path must be one of grpo, rft, got 'sft'"):
        rl_doc(path="sft")
    partial = models()
    del partial["tuned"]
    with pytest.raises(ValueError, match="missing tuned"):
        rl_doc(evaluated={"models": partial})


# ---- E-3: the forgetting warning ------------------------------------------------------------


def test_warning_line_is_printed_when_normal_tool_f1_drops_more_than_005(capsys):
    train_grpo.print_summary(models(tuned_normal=0.7))
    out = capsys.readouterr().out
    assert "warning: normal_tool_f1 fell from 0.800 (base) to 0.700 (tuned)" in out
    assert out.count("\n") == 5  # four model lines and the warning


@pytest.mark.parametrize("tuned_normal", [0.9, 0.8, 0.77, 0.75])
def test_no_warning_when_normal_tool_f1_held_or_dropped_by_at_most_005(capsys, tuned_normal):
    train_grpo.print_summary(models(tuned_normal=tuned_normal))
    out = capsys.readouterr().out
    assert "warning" not in out
    assert out.count("\n") == 4


# ---- B-1: first reward call diagnostics -----------------------------------------------------


def test_first_reward_call_prints_completion_zero_and_kwargs_keys_once():
    seen: list[str] = []
    calls = []

    def reward(prompts, completions, **kwargs):
        calls.append(kwargs)
        return [1.0] * len(completions)

    wrapped = train_grpo.announce_first_call(reward, say=seen.append)
    completion = [{"role": "assistant", "content": '{"name": "x"}'}]
    assert wrapped(["p"], [completion], task_id=["t1"], trainer_state=None) == [1.0]
    wrapped(["p"], [completion], task_id=["t1"], trainer_state=None)
    assert seen == [
        "first reward call: completions[0] = "
        "[{'role': 'assistant', 'content': '{\"name\": \"x\"}'}]",
        "first reward call: kwargs.keys() = ['task_id', 'trainer_state']",
    ]
    assert len(calls) == 2  # the wrapper still delegates every call


# ---- the training loop around a fake trainer ------------------------------------------------


def make_args(**overrides) -> argparse.Namespace:
    values = dict(hub_repo="someone/alien-grpo", seed=0, smoke=False, hardness=0,
                  revision="main", sft_adapter="/sft")
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeModel:
    def save_pretrained(self, directory: str) -> None:
        (Path(directory) / "adapter_model.safetensors").write_text("weights")


class FakeTrainer:
    """Stands in for GRPOTrainer: ``body`` plays the training loop against the real reward."""

    def __init__(self, body, steps: int = 7):
        self.body = body
        self.state = SimpleNamespace(global_step=steps)
        self.model = FakeModel()
        self.resumed = "never called"

    def train(self, resume_from_checkpoint=None):
        self.resumed = resume_from_checkpoint
        self.body()


def factory(body, record=None):
    """A ``make_trainer`` whose fake loop is ``body(reward_fn, monitor)``."""

    def make_trainer(args, rows, reward_fn, monitor, ckpt_root, trainer_dir, hparams):
        trainer = FakeTrainer(lambda: body(reward_fn, monitor))
        if record is not None:
            record.append(trainer)
        return trainer, "rev9"

    return make_trainer


def score_group(reward_fn, task, texts):
    """One GRPO reward call: TRL repeats the prompt once per generation."""
    prompts = [[{"role": "user", "content": render_prompt(task)}] for _ in texts]
    completions = [[{"role": "assistant", "content": text}] for text in texts]
    return reward_fn(prompts, completions, task_id=[task.id] * len(texts))


@pytest.fixture
def train_tasks():
    return make_splits(0).train[:60]


def run_train(tasks, ckpt_root, body, **kwargs):
    hparams = train_grpo.plan_grpo_hparams(0, False)
    args = kwargs.pop("args", make_args())
    kwargs.setdefault("upload", lambda folder: "abc123")
    return train_grpo.train(
        args, train_grpo.build_grpo_rows(Splits(tasks, [], [], "x", 0, 0)), tasks, ckpt_root,
        ckpt_root.parent / "trainer", hparams, make_trainer=factory(body), **kwargs)


def test_all_zero_groups_stop_the_run_with_exit_3_and_keep_the_checkpoint(
        tmp_path, capsys, train_tasks):
    ckpt_root = tmp_path / "ckpt"
    write_checkpoint(ckpt_root, "step-50", lambda d: (d / "adapter.bin").write_text("w"))

    def body(reward_fn, monitor):
        for task in train_tasks:  # 60 groups of 4 wrong answers: past the patience of 50
            assert score_group(reward_fn, task, ['{"name": "nope", "arguments": {}}'] * 4) \
                == [0.0] * 4
            monitor.raise_if_dead()  # what the Trainer callback does after every step

    outcome = run_train(train_tasks, ckpt_root, body)
    assert outcome["code"] == 3
    assert outcome["final_dir"] is None
    out = capsys.readouterr().out
    assert "reward is dead: switching to rejection-sampling fine-tuning, " \
           "see scripts/train_rft.py" in out
    assert [p.name for p in load_checkpoints(ckpt_root)] == ["step-50"]
    assert (ckpt_root / "step-50" / "adapter.bin").read_text() == "w"  # nothing deleted


def test_a_run_that_ends_with_no_reward_variance_is_also_dead_and_never_uploads(
        tmp_path, train_tasks):
    def body(reward_fn, monitor):
        for task in train_tasks[:10]:  # too few groups for the monitor to raise on its own
            score_group(reward_fn, task, ["garbage"] * 4)

    def must_not_upload(folder):
        raise AssertionError("a dead run must not upload")

    outcome = run_train(train_tasks, tmp_path / "ckpt", body, upload=must_not_upload)
    assert outcome["code"] == 3
    assert load_checkpoints(tmp_path / "ckpt") == []


def test_a_run_with_reward_variance_uploads_and_reports_the_real_signal(tmp_path, train_tasks):
    task = train_tasks[0]
    right = json.dumps(task.oracle_call)

    def body(reward_fn, monitor):
        assert score_group(reward_fn, task, [right, "garbage", "garbage", "garbage"]) \
            == [1.0, 0.0, 0.0, 0.0]

    uploaded = []
    outcome = run_train(train_tasks, tmp_path / "ckpt", body,
                        upload=lambda folder: uploaded.append(folder) or "abc123")
    assert outcome["code"] == 0 and outcome["revision"] == "abc123"
    assert outcome["summary"] == {
        "steps": 7, "model_revision": "rev9",
        "reward_signal": {"groups_total": 1, "groups_with_variance": 1}}
    final = outcome["final_dir"]
    assert uploaded == [final] and final == latest_checkpoint(tmp_path / "ckpt")
    assert (final / "adapter_model.safetensors").read_text() == "weights"
    assert json.loads((final / "training_summary.json").read_text())["steps"] == 7


def test_upload_failure_keeps_the_trained_checkpoint_and_returns_1(
        tmp_path, capsys, monkeypatch, train_tasks):
    monkeypatch.setattr(train_grpo, "finish_and_upload", functools.partial(
        train_sft.finish_and_upload, base_delay=0.0, sleep=lambda s: None))
    task = train_tasks[0]
    right = json.dumps(task.oracle_call)

    def failing_upload(folder):
        raise ConnectionError("down")

    outcome = run_train(
        train_tasks, tmp_path / "ckpt",
        lambda reward_fn, monitor: score_group(reward_fn, task, [right, "x", "x", "x"]),
        upload=failing_upload)
    assert outcome["code"] == 1 and outcome["revision"] is None
    final = latest_checkpoint(tmp_path / "ckpt")
    assert final == outcome["final_dir"]
    assert (final / "adapter_model.safetensors").exists()
    assert "nothing was deleted" in capsys.readouterr().out


def finished_final(ckpt_root, with_config=True, **arg_overrides):
    """A manifest-listed ``final`` checkpoint as an earlier run with these args left it."""
    args = make_args(**arg_overrides)
    config = train_sft.build_run_config(
        "train_grpo", args, train_grpo.plan_grpo_hparams(args.seed, False),
        sft_adapter=args.sft_adapter)

    def write_final(directory):
        (directory / "adapter_model.safetensors").write_text("weights")
        (directory / "training_summary.json").write_text(json.dumps({
            "steps": 7, "model_revision": "rev9",
            "reward_signal": {"groups_total": 5, "groups_with_variance": 2}}))
        if with_config:
            train_sft.write_run_config(directory, config, "rev9")

    write_checkpoint(ckpt_root, "final", write_final)


def test_rerun_after_a_complete_final_checkpoint_only_retries_the_upload(tmp_path, train_tasks):
    ckpt_root = tmp_path / "ckpt"
    finished_final(ckpt_root)

    def must_not_train(reward_fn, monitor):
        raise AssertionError("training must be skipped")

    outcome = run_train(train_tasks, ckpt_root, must_not_train,
                        upload=lambda folder: "r2")
    assert outcome["code"] == 0 and outcome["revision"] == "r2"
    assert outcome["summary"]["reward_signal"] == {"groups_total": 5, "groups_with_variance": 2}


def test_resume_continues_from_the_newest_checkpoint_and_keeps_its_reward_counters(
        tmp_path, train_tasks):
    ckpt_root = tmp_path / "ckpt"

    def write_step(directory):
        (directory / "adapter.bin").write_text("w")
        (directory / "reward_monitor.json").write_text(
            json.dumps({"groups_total": 40, "groups_with_variance": 3}))

    write_checkpoint(ckpt_root, "step-50", write_step)
    task = train_tasks[0]
    trainers: list[FakeTrainer] = []
    hparams = train_grpo.plan_grpo_hparams(0, False)
    outcome = train_grpo.train(
        make_args(), [], train_tasks, ckpt_root, tmp_path / "trainer", hparams,
        make_trainer=factory(
            lambda reward_fn, monitor: score_group(reward_fn, task, ["x", "x", "x", "x"]),
            record=trainers),
        upload=lambda folder: "abc")
    assert trainers[0].resumed == str(ckpt_root / "step-50")
    # 40 groups before the interruption (3 varied) + 1 all-zero group after it
    assert outcome["summary"]["reward_signal"] == {"groups_total": 41, "groups_with_variance": 3}
    assert outcome["code"] == 0


# ---- inputs (exit 2 before any training) ----------------------------------------------------


def run_main(tmp_path, *extra, seed="0"):
    return train_grpo.main(["--hub-repo", "someone/alien-grpo", "--seed", seed,
                            "--work-dir", str(tmp_path / "w"),
                            "--baselines", str(tmp_path / "eval_baselines.json"),
                            "--sft-results", str(tmp_path / "sft.json"),
                            "--sft-adapter", str(tmp_path), *extra])


@pytest.fixture
def no_heavy_imports(monkeypatch):
    for name in HEAVY:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(train_grpo, "train", lambda *a, **k: pytest.fail("training started"))


def test_baselines_made_with_another_seed_stop_the_run_before_training(
        tmp_path, capsys, no_heavy_imports):
    write_prior_files(tmp_path, baselines_seed=1)
    assert run_main(tmp_path) == 2
    assert "eval_baselines.json was made with seed 1 but this run uses seed 0" \
        in capsys.readouterr().err
    assert not any(name in sys.modules for name in HEAVY)


def test_sft_results_made_with_another_seed_stop_the_run_before_training(
        tmp_path, capsys, no_heavy_imports):
    write_prior_files(tmp_path, sft_seed=5)
    assert run_main(tmp_path) == 2
    assert "sft.json was made with seed 5" in capsys.readouterr().err


def test_missing_sft_results_stop_the_run_with_a_plain_message(
        tmp_path, capsys, no_heavy_imports):
    write_prior_files(tmp_path)
    (tmp_path / "sft.json").unlink()
    assert run_main(tmp_path) == 2
    assert "sft.json not found: run scripts/train_sft.py first" in capsys.readouterr().err


def test_missing_baselines_stop_the_run_with_a_plain_message(
        tmp_path, capsys, no_heavy_imports):
    write_prior_files(tmp_path)
    (tmp_path / "eval_baselines.json").unlink()
    assert run_main(tmp_path) == 2
    assert "eval_baselines.json not found: run scripts/run_eval.py first" \
        in capsys.readouterr().err


def test_a_missing_default_sft_adapter_stops_the_run_with_a_plain_message(
        tmp_path, capsys, no_heavy_imports):
    write_prior_files(tmp_path)
    code = train_grpo.main(["--hub-repo", "someone/alien-grpo", "--work-dir", str(tmp_path / "w"),
                            "--baselines", str(tmp_path / "eval_baselines.json"),
                            "--sft-results", str(tmp_path / "sft.json")])
    assert code == 2
    assert "no SFT adapter at" in capsys.readouterr().err


def test_smoke_without_an_sft_adapter_is_a_usage_error(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        train_grpo.main(["--smoke", "--work-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "--sft-adapter" in capsys.readouterr().err


def test_a_real_run_without_hub_repo_is_a_usage_error(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        train_grpo.main(["--work-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "--hub-repo" in capsys.readouterr().err


# ---- the whole command, with the GPU steps replaced -----------------------------------------


def ok_outcome(tmp_path):
    return {"code": 0, "revision": "def456", "final_dir": tmp_path / "final",
            "summary": {"steps": 100, "model_revision": "rev9",
                        "reward_signal": {"groups_total": 400, "groups_with_variance": 37}}}


def test_a_successful_run_writes_a_valid_grpo_json_with_the_four_models(
        tmp_path, capsys, monkeypatch):
    write_prior_files(tmp_path)
    monkeypatch.setattr(train_grpo, "train", lambda *a, **k: ok_outcome(tmp_path))
    monkeypatch.setattr(train_grpo, "evaluate_tuned", lambda *a, **k: {
        "models": {"tuned": metrics(0.1, 0.9, normal=0.6)}, "gpu_seconds": 300.0})
    out = tmp_path / "grpo.json"
    assert run_main(tmp_path, "--out", str(out)) == 0
    doc = json.loads(out.read_text())
    assert check_results.validate(doc) == []
    assert doc["path"] == "grpo" and doc["generations_per_prompt"] == 4
    assert doc["reward_signal"] == {"groups_total": 400, "groups_with_variance": 37}
    assert doc["checkpoint"] == {"hub_repo": "someone/alien-grpo", "revision": "def456"}
    assert doc["models"]["base"]["prior_trap_rate"]["value"] == 0.7  # from eval_baselines
    assert doc["models"]["sft"]["tool_call_f1"]["value"] == 0.8  # from sft.json
    assert doc["models"]["tuned"]["tool_call_f1"]["value"] == 0.9
    assert "warning: normal_tool_f1 fell from 0.800 (base) to 0.600 (tuned)" \
        in capsys.readouterr().out


def test_an_upload_failure_returns_1_and_writes_no_results_file(tmp_path, monkeypatch):
    write_prior_files(tmp_path)
    failed = dict(ok_outcome(tmp_path), code=1, revision=None)
    monkeypatch.setattr(train_grpo, "train", lambda *a, **k: failed)
    monkeypatch.setattr(train_grpo, "evaluate_tuned",
                        lambda *a, **k: pytest.fail("no evaluation after a failed upload"))
    out = tmp_path / "grpo.json"
    assert run_main(tmp_path, "--out", str(out)) == 1
    assert not out.exists()


def test_a_dead_reward_returns_3_writes_no_results_and_runs_no_fallback_by_default(
        tmp_path, monkeypatch):
    write_prior_files(tmp_path)
    dead = {"code": 3, "revision": None, "summary": None, "final_dir": None}
    monkeypatch.setattr(train_grpo, "train", lambda *a, **k: dead)
    monkeypatch.setattr(train_grpo, "run_fallback",
                        lambda args: pytest.fail("fallback must be opt-in"))
    out = tmp_path / "grpo.json"
    assert run_main(tmp_path, "--out", str(out)) == 3
    assert not out.exists()


def test_auto_fallback_runs_train_rft_in_the_same_process_after_a_dead_reward(
        tmp_path, monkeypatch):
    import train_rft

    write_prior_files(tmp_path)
    dead = {"code": 3, "revision": None, "summary": None, "final_dir": None}
    monkeypatch.setattr(train_grpo, "train", lambda *a, **k: dead)
    seen = []
    monkeypatch.setattr(train_rft, "main", lambda argv: seen.append(argv) or 0)
    assert run_main(tmp_path, "--auto-fallback") == 0
    argv = seen[0]
    assert argv[argv.index("--hub-repo") + 1] == "someone/alien-grpo"
    assert argv[argv.index("--seed") + 1] == "0"
    assert argv[argv.index("--sft-adapter") + 1] == str(tmp_path)
    assert argv[argv.index("--work-dir") + 1] == str(tmp_path / "w")


# ---- dry run, secrets, imports --------------------------------------------------------------


def test_dry_run_prints_the_plan_exits_zero_and_imports_no_heavy_library(
        monkeypatch, capsys, tmp_path):
    for name in HEAVY:
        monkeypatch.delitem(sys.modules, name, raising=False)
    code = train_grpo.main(["--dry-run", "--work-dir", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch" not in sys.modules
    assert "train rows: 600" in out
    assert "generations per prompt: 4" in out
    assert '"learning_rate": 5e-06' in out
    assert "resume: fresh" in out
    assert "auto fallback to rejection sampling: off" in out
    assert not (tmp_path / "w").exists()  # a dry run creates nothing
    assert not any(name in sys.modules for name in HEAVY)


def test_dry_run_smoke_uses_16_rows_and_no_hub_upload(capsys, tmp_path):
    assert train_grpo.main(["--dry-run", "--smoke", "--work-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "train rows: 16" in out
    assert "hub upload: none (smoke)" in out
    assert '"max_steps": 3' in out


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


def spy_on_grpo_splits(monkeypatch):
    seen = {}
    real = train_grpo.train_tasks

    def spy(splits, n=None):
        seen["splits"] = splits
        return real(splits, n)

    monkeypatch.setattr(train_grpo, "train_tasks", spy)
    return seen


def test_hardness_1_builds_the_grpo_tasks_from_the_hardness_1_split(monkeypatch, tmp_path):
    # Breaks if train_grpo ignores --hardness and calls make_splits(seed) at hardness 0.
    seen = spy_on_grpo_splits(monkeypatch)
    assert train_grpo.main(["--dry-run", "--hardness", "1", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 1
    assert seen["splits"].train[0].id == "train-0000"
    assert H1_TRAIN_NOTE in seen["splits"].train[0].tool.doc
    assert H0_TRAIN_NOTE not in seen["splits"].train[0].tool.doc


def test_default_hardness_builds_the_hardness_0_split(monkeypatch, tmp_path):
    seen = spy_on_grpo_splits(monkeypatch)
    assert train_grpo.main(["--dry-run", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 0
    assert H0_TRAIN_NOTE in seen["splits"].train[0].tool.doc


@pytest.mark.parametrize("value", ["-1", "99"])
def test_hardness_outside_0_to_3_is_a_parser_error_with_exit_2(capsys, tmp_path, value):
    with pytest.raises(SystemExit) as raised:
        train_grpo.main(["--dry-run", "--hardness", value, "--work-dir", str(tmp_path)])
    assert raised.value.code == 2
    assert "--hardness must be between 0 and 3" in capsys.readouterr().err


def test_a_splits_file_is_accepted_at_its_own_hardness_and_rejected_at_another(
        tmp_path, capsys, monkeypatch):
    # Breaks if --splits stops checking the file's recorded hardness against --hardness.
    path = tmp_path / "splits.json"
    path.write_text(splits_to_json(make_splits(0, 1)))
    seen = spy_on_grpo_splits(monkeypatch)
    argv = ["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")]
    assert train_grpo.main([*argv, "--hardness", "1"]) == 0
    assert seen["splits"].hardness == 1
    with pytest.raises(SystemExit) as raised:
        train_grpo.main(argv)
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "was made with seed 0 and hardness 1 but this run uses seed 0 and hardness 0" in err
    assert "Traceback" not in err


def test_auto_fallback_passes_the_hardness_on_to_train_rft(monkeypatch, tmp_path):
    # Breaks if run_fallback drops --hardness: rft would then train on the hardness-0 split
    # right after grpo trained on the hardened one.
    import train_rft

    seen = {}
    monkeypatch.setattr(train_rft, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    args = argparse.Namespace(
        hub_repo="u/r", seed=0, hardness=2, revision="main", baselines="b.json",
        sft_results="s.json", eval_batch_size=8, sft_adapter="ad", work_dir=None, splits=None,
        smoke=False)
    assert train_grpo.run_fallback(args) == 0
    argv = seen["argv"]
    assert argv[argv.index("--hardness") + 1] == "2"


# ---- the tuned evaluation is fingerprinted by adapter and revision (review finding 2) -------


def fake_hf_loader(monkeypatch, splits, kind_by_adapter):
    """Replace run_eval.make_hf_generate: the adapter decides oracle or habit answers, and every
    prompt generated is counted."""
    import run_eval

    prompts = []
    tasks = splits.heldout[:200]

    def make(model_id, revision, adapter, decoding):
        inner = run_eval.make_fake_generate(tasks, kind_by_adapter[adapter])

        def generate(batch):
            prompts.extend(batch)
            return inner(batch)

        return generate

    monkeypatch.setattr(run_eval, "make_hf_generate", make)
    return prompts


def eval_args(**overrides) -> argparse.Namespace:
    values = dict(seed=0, eval_batch_size=8, revision="main", hardness=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluate_tuned_refuses_a_second_adapter_over_the_same_progress_dir(
        tmp_path, monkeypatch):
    # Breaks if evaluate_tuned stops passing a run_key: adapter B would be scored on adapter
    # A's recorded outputs (reviewer saw F1 1.0 for B, real 0.425).
    from alien.persist import StaleProgressError

    splits = make_splits(0)
    prompts = fake_hf_loader(monkeypatch, splits, {"/ckpt/A": "oracle", "/ckpt/B": "habit"})
    first = train_grpo.evaluate_tuned(
        eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev1")
    assert first["models"]["tuned"]["tool_call_f1"]["value"] == 1.0
    assert len(prompts) == 200
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        train_grpo.evaluate_tuned(eval_args(), splits, tmp_path, Path("/ckpt/B"), "rev1")
    assert len(prompts) == 200  # nothing was generated for B


def test_evaluate_tuned_refuses_a_new_base_revision_but_resumes_the_same_run(
        tmp_path, monkeypatch):
    # Breaks if the resolved base revision is left out of the run_key.
    from alien.persist import StaleProgressError

    splits = make_splits(0)
    prompts = fake_hf_loader(monkeypatch, splits, {"/ckpt/A": "oracle"})
    train_grpo.evaluate_tuned(eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev1")
    train_grpo.evaluate_tuned(eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev1")
    assert len(prompts) == 200  # the identical second run generated nothing
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        train_grpo.evaluate_tuned(eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev2")


def test_finalize_turns_a_stale_tuned_progress_dir_into_exit_2_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    # Breaks if finalize lets StaleProgressError escape as a traceback instead of exit 2.
    from alien.persist import StaleProgressError

    def stale(*args, **kwargs):
        raise StaleProgressError("p/tuned.jsonl belongs to a different run; "
                                 "use a fresh --progress-dir or delete this file")

    monkeypatch.setattr(train_grpo, "evaluate_tuned", stale)
    outcome = {"summary": {"model_revision": "rev1", "reward_signal": {}}, "final_dir": tmp_path,
               "revision": "abc"}
    code = train_grpo.finalize(
        eval_args(out=str(tmp_path / "o.json"), hub_repo="u/r"), path="grpo",
        splits=make_splits(0), run=tmp_path, outcome=outcome, prior={}, generations_per_prompt=4)
    assert code == 2
    err = capsys.readouterr().err
    assert "fresh --progress-dir" in err and "Traceback" not in err


def test_grpo_dry_run_rejects_a_splits_file_that_fails_the_leak_check(tmp_path, capsys):
    # Breaks if train_grpo accepts an imported splits file with a held-out task in train
    # (reviewer saw dry runs exit 0 with 601 train tasks).
    real = make_splits(0)
    path = tmp_path / "leaky.json"
    path.write_text(splits_to_json(Splits(
        [*real.train, real.heldout[0]], real.dev, real.heldout, real.heldout_twist, 0, 0)))
    with pytest.raises(SystemExit) as raised:
        train_grpo.main(["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert f"error: {path} fails the leak check: held-out tools in train: " in err
    assert "Traceback" not in err


# ---- a finished checkpoint is reused only by the same run (review finding 4) ---------------


def must_not_train(reward_fn, monitor):
    raise AssertionError("training must be skipped")


def must_not_upload(folder):
    raise AssertionError("a refused run must not upload")


def test_a_final_checkpoint_of_another_seed_hardness_or_base_revision_is_refused(
        tmp_path, capsys, train_tasks):
    # Breaks if train() reuses a manifest-listed final checkpoint without comparing the run:
    # reviewer saw seed 1, hardness 3, NEW_BASE return success with the OLD_BASE summary.
    ckpt_root = tmp_path / "ckpt"
    finished_final(ckpt_root, seed=0, hardness=0, revision="OLD_BASE")
    outcome = run_train(train_tasks, ckpt_root, must_not_train, upload=must_not_upload,
                        args=make_args(seed=1, hardness=3, revision="NEW_BASE"))
    assert outcome == {"code": 2, "revision": None, "summary": None, "final_dir": None}
    err = capsys.readouterr().err
    assert err == (f"error: {ckpt_root / 'final'} was made by a different run (differs in: "
                   "hardness, revision, seed); use a fresh --work-dir\n")


def test_a_final_checkpoint_from_another_sft_adapter_or_hyperparameter_is_refused(
        tmp_path, capsys, train_tasks):
    # Breaks if the starting adapter is left out of the recorded config.
    ckpt_root = tmp_path / "ckpt"
    finished_final(ckpt_root, sft_adapter="/sft-A")
    outcome = run_train(train_tasks, ckpt_root, must_not_train, upload=must_not_upload,
                        args=make_args(sft_adapter="/sft-B"))
    assert outcome["code"] == 2
    assert "differs in: sft_adapter); use a fresh --work-dir" in capsys.readouterr().err


def test_a_final_checkpoint_with_no_recorded_config_is_refused(tmp_path, capsys, train_tasks):
    # Breaks if an older checkpoint (no run_config.json) is trusted instead of refused.
    ckpt_root = tmp_path / "ckpt"
    finished_final(ckpt_root, with_config=False)
    outcome = run_train(train_tasks, ckpt_root, must_not_train, upload=must_not_upload)
    assert outcome["code"] == 2
    assert capsys.readouterr().err == (
        f"error: {ckpt_root / 'final'} has no run_config.json (written by older code); "
        "use a fresh --work-dir\n")


def test_a_trained_final_checkpoint_records_the_run_config_and_the_same_run_reuses_it(
        tmp_path, train_tasks):
    # Breaks if train() stops writing run_config.json next to the adapter (the reuse check
    # would then refuse every finished run) or writes different values.
    task = train_tasks[0]
    right = json.dumps(task.oracle_call)
    ckpt_root = tmp_path / "ckpt"
    outcome = run_train(train_tasks, ckpt_root, lambda reward_fn, monitor: score_group(
        reward_fn, task, [right, "x", "x", "x"]))
    recorded = json.loads((outcome["final_dir"] / "run_config.json").read_text())
    assert recorded["script"] == "train_grpo"
    assert (recorded["seed"], recorded["hardness"], recorded["revision"]) == (0, 0, "main")
    assert recorded["model_id"] == "CohereLabs/tiny-aya-global"
    assert (recorded["resolved_revision"], recorded["sft_adapter"]) == ("rev9", "/sft")
    assert (recorded["hparams.num_generations"], recorded["hparams.max_steps"]) == (4, 100)
    again = run_train(train_tasks, ckpt_root, must_not_train, upload=lambda folder: "r2")
    assert again["code"] == 0 and again["revision"] == "r2"
