"""Tests for the SFT training script's CPU-testable parts (FR-005, S3, E-6, R-4, B-2, B-3).

No torch, no GPU, no network. The GPU path (``train``) and the hub upload call are Colab-gated;
what is tested here is everything around them. Each group names the production change that
would make it fail.
"""

import json
import re
import sys
from pathlib import Path

import pytest

import check_results
import train_sft
from alien.eval import ResultsInvalidError, build_results
from alien.persist import latest_checkpoint, load_checkpoints
from alien.splits import Splits, make_splits, splits_to_json

SCRIPT = Path(train_sft.__file__)


def metric(value: float) -> dict:
    return {"value": value, "ci_low": max(0.0, value - 0.05), "ci_high": min(1.0, value + 0.05),
            "n": 200}


def metrics(trap: float, f1: float) -> dict:
    return {"prior_trap_rate": metric(trap), "tool_call_f1": metric(f1),
            "tool_call_validity": metric(0.9), "normal_tool_f1": metric(0.8)}


def fake_evaluated(with_sft: bool = True) -> dict:
    models = {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4)}
    if with_sft:
        models["sft"] = metrics(0.2, 0.8)
    return {"models": models, "gpu_seconds": 1200.0, "n_examples": 200}


def sft_doc(**overrides) -> dict:
    kwargs = dict(evaluated=fake_evaluated(), loss_first=2.5, loss_last=0.4, steps=76,
                  hub_repo="someone/alien-sft", revision="abc123", model_revision="rev9",
                  seed=0, git_sha="deadbeef", created_at="2026-11-07T18:30:00Z")
    kwargs.update(overrides)
    return train_sft.build_sft_results(**kwargs)


class Forbidden(list):
    """A split that must never be read: any access fails the test."""

    def __iter__(self):
        raise AssertionError("this split must not be read (B-2)")

    def __len__(self):
        raise AssertionError("this split must not be read (B-2)")

    def __getitem__(self, index):
        raise AssertionError("this split must not be read (B-2)")


# ---- hyper-parameters (PRD S3, AS-2): a changed constant fails a literal ----------------


def test_sft_hparams_are_the_prd_values():
    hp = train_sft.SFT_HPARAMS
    assert hp["learning_rate"] == 2e-4
    assert hp["lora_r"] == 16
    assert hp["lora_alpha"] == 32
    assert hp["lora_dropout"] == 0.05
    assert hp["target_modules"] == "all-linear"
    assert hp["num_train_epochs"] == 2
    assert hp["per_device_train_batch_size"] == 2
    assert hp["gradient_accumulation_steps"] == 8
    assert hp["max_length"] == 1024
    assert hp["warmup_ratio"] == 0.03
    assert hp["fp16"] is True
    assert hp["save_steps"] == 50
    assert hp["logging_steps"] == 5


def test_plan_hparams_smoke_shrinks_the_run_and_normal_run_keeps_the_constants():
    normal = train_sft.plan_hparams(seed=3, smoke=False)
    assert normal["seed"] == 3
    assert normal["max_length"] == 1024
    assert "max_steps" not in normal
    smoke = train_sft.plan_hparams(seed=3, smoke=True)
    assert (smoke["max_steps"], smoke["max_length"], smoke["save_steps"]) == (8, 256, 4)
    assert smoke["learning_rate"] == 2e-4
    assert train_sft.SFT_HPARAMS["max_length"] == 1024  # the constant itself is not mutated


# ---- rows (B-2): train split only, oracle JSON as the last message ----------------------


def test_build_sft_rows_has_one_row_per_train_task_ending_in_the_oracle_call():
    splits = make_splits(0)
    rows = train_sft.build_sft_rows(splits)
    assert len(rows) == len(splits.train) == 600
    for row, task in zip(rows, splits.train):
        last = row["messages"][-1]
        assert last["role"] == "assistant"
        assert last["content"].startswith('{"name": "')
        assert json.loads(last["content"]) == task.oracle_call
        assert row["messages"][0]["role"] == "user"


def test_build_sft_rows_never_reads_heldout_or_dev():
    real = make_splits(0)
    guarded = Splits(real.train, Forbidden(), Forbidden(), real.heldout_twist, 0, 0)
    assert len(train_sft.build_sft_rows(guarded)) == 600


def test_to_prompt_completion_splits_the_assistant_message_off():
    row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    assert train_sft.to_prompt_completion(row) == {
        "prompt": [{"role": "user", "content": "Q"}],
        "completion": [{"role": "assistant", "content": "A"}],
    }


# ---- losses ------------------------------------------------------------------------------


def test_extract_losses_ignores_entries_without_loss():
    log = [
        {"loss": 2.5, "step": 5, "epoch": 0.1},
        {"eval_loss": 9.9, "step": 5},
        {"loss": 1.75, "step": 10, "epoch": 0.2},
        {"loss": 0.5, "step": 15, "epoch": 0.3},
        {"train_runtime": 9.0, "train_loss": 1.6, "step": 16},
    ]
    assert train_sft.extract_losses(log) == (2.5, 0.5, 15)


def test_extract_losses_single_entry_is_first_and_last():
    assert train_sft.extract_losses([{"loss": 1.25, "step": 5}]) == (1.25, 1.25, 5)


@pytest.mark.parametrize("log", [[], [{"train_runtime": 3.0, "step": 4}]])
def test_extract_losses_without_any_loss_raises_a_named_error(log):
    with pytest.raises(train_sft.NoLossLoggedError, match="loss"):
        train_sft.extract_losses(log)


# ---- results document (FR-006, contract kind sft) ----------------------------------------


def test_build_sft_results_passes_the_validator_and_has_the_contract_fields():
    doc = sft_doc()
    assert check_results.validate(doc) == []
    assert doc["kind"] == "sft"
    assert doc["checkpoint"] == {"hub_repo": "someone/alien-sft", "revision": "abc123"}
    assert doc["model_revision"] == "rev9"
    assert (doc["loss_first"], doc["loss_last"], doc["steps"]) == (2.5, 0.4, 76)
    assert sorted(doc["models"]) == ["base", "prompt_only", "sft"]
    assert doc["models"]["sft"]["prior_trap_rate"]["value"] == 0.2
    assert doc["seed"] == 0 and doc["git_sha"] == "deadbeef"


def test_a_loss_that_did_not_fall_is_reported_by_the_validator_on_loss_last():
    errors = check_results.validate(sft_doc(loss_first=1.0, loss_last=1.0))
    assert len(errors) == 1
    assert errors[0].startswith("loss_last: expected a number < loss_first")


def test_build_sft_results_names_a_missing_model():
    with pytest.raises(ValueError, match="sft"):
        sft_doc(evaluated=fake_evaluated(with_sft=False))


def test_load_baselines_reads_a_valid_eval_baselines_file(tmp_path):
    doc = build_results(fake_evaluated(with_sft=False), seed=0, model_revision="rev9",
                        created_at="2026-11-07T18:30:00Z")
    path = tmp_path / "eval_baselines.json"
    path.write_text(json.dumps(doc))
    loaded = train_sft.load_baselines(path)
    assert sorted(loaded["models"]) == ["base", "prompt_only"]
    assert loaded["models"]["base"]["tool_call_f1"]["value"] == 0.3
    assert loaded["seed"] == 0


def test_load_baselines_rejects_an_invalid_file(tmp_path):
    path = tmp_path / "eval_baselines.json"
    path.write_text(json.dumps({"kind": "eval_baselines"}))
    with pytest.raises(ResultsInvalidError, match="schema_version"):
        train_sft.load_baselines(path)


# ---- checkpoints and upload (E-6, R-4) ----------------------------------------------------


def test_mirror_checkpoint_lists_the_copy_in_the_manifest_and_is_idempotent(tmp_path):
    src = tmp_path / "trainer" / "checkpoint-50"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_text("weights")
    root = tmp_path / "ckpt"
    train_sft.mirror_checkpoint(src, root, "step-50")
    train_sft.mirror_checkpoint(src, root, "step-50")
    assert [p.name for p in load_checkpoints(root)] == ["step-50"]
    assert (root / "step-50" / "adapter_model.safetensors").read_text() == "weights"


def test_resume_state_tells_fresh_resume_and_final_apart(tmp_path):
    root = tmp_path / "ckpt"
    assert train_sft.resume_state(root) == ("fresh", None)
    src = tmp_path / "src"
    src.mkdir()
    train_sft.mirror_checkpoint(src, root, "step-50")
    assert train_sft.resume_state(root) == ("resume", root / "step-50")
    train_sft.mirror_checkpoint(src, root, "final")
    assert train_sft.resume_state(root) == ("final", root / "final")


def write_adapter(directory: Path) -> None:
    (directory / "adapter_model.safetensors").write_text("weights")


def test_failed_upload_keeps_the_checkpoint_and_returns_a_nonzero_code(tmp_path):
    root = tmp_path / "ckpt"
    calls = []
    sleeps = []

    def failing_upload(folder: Path) -> str:
        calls.append(folder)
        raise ConnectionError("network down hf_" + "x" * 30)

    messages = []
    code, revision, final_dir = train_sft.finish_and_upload(
        root, write_adapter, failing_upload, sleep=sleeps.append, say=messages.append)
    assert code == 1
    assert revision is None
    assert len(calls) == 3 and sleeps == [2.0, 4.0]
    assert latest_checkpoint(root) == final_dir == root / "final"
    assert (final_dir / "adapter_model.safetensors").read_text() == "weights"
    text = "\n".join(messages)
    assert "3 attempts" in text and str(final_dir) in text
    assert "hf_" not in text


def test_successful_upload_returns_the_revision_after_the_checkpoint_is_listed(tmp_path):
    root = tmp_path / "ckpt"
    seen = []

    def upload(folder: Path) -> str:
        seen.append(latest_checkpoint(root))  # the manifest already lists the final adapter
        return "abc123"

    code, revision, final_dir = train_sft.finish_and_upload(
        root, write_adapter, upload, sleep=lambda s: None, say=lambda m: None)
    assert (code, revision) == (0, "abc123")
    assert seen == [final_dir]


def test_rerun_with_a_complete_final_checkpoint_retries_only_the_upload(tmp_path):
    root = tmp_path / "ckpt"

    def fail(folder: Path) -> str:
        raise OSError("down")

    train_sft.finish_and_upload(root, write_adapter, fail, sleep=lambda s: None,
                                say=lambda m: None)

    def must_not_write(directory: Path) -> None:
        raise AssertionError("the final adapter was already written")

    code, revision, _ = train_sft.finish_and_upload(
        root, must_not_write, lambda folder: "r2", sleep=lambda s: None, say=lambda m: None)
    assert (code, revision) == (0, "r2")


def test_default_work_dir_prefers_a_mounted_drive(tmp_path):
    drive = tmp_path / "drive"
    assert train_sft.default_work_dir(drive) == Path("work")
    drive.mkdir()
    assert train_sft.default_work_dir(drive) == drive / "alien_manual"


# ---- the command line ---------------------------------------------------------------------


def test_dry_run_prints_the_plan_exits_zero_and_imports_no_heavy_library(
        monkeypatch, capsys, tmp_path):
    heavy = ("torch", "transformers", "peft", "trl", "huggingface_hub", "datasets")
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, None)  # an import of it now raises ImportError
    code = train_sft.main(["--dry-run", "--work-dir", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert code == 0
    assert "train rows: 600" in out
    assert "held-out examples to evaluate: 200" in out
    assert '"learning_rate": 0.0002' in out
    assert "resume: fresh" in out
    assert not (tmp_path / "w").exists()  # a dry run creates nothing
    for name in heavy:
        assert sys.modules[name] is None


def test_dry_run_smoke_uses_16_rows_and_no_hub_upload(capsys, tmp_path):
    assert train_sft.main(["--dry-run", "--smoke", "--work-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "train rows: 16" in out
    assert "hub upload: none (smoke)" in out
    assert "results file: none (smoke)" in out
    assert '"max_steps": 8' in out


def test_a_real_run_without_hub_repo_is_a_usage_error(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        train_sft.main(["--work-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "--hub-repo" in capsys.readouterr().err


def test_baselines_made_with_another_seed_stop_the_run_before_any_training(
        capsys, tmp_path):
    doc = build_results(fake_evaluated(with_sft=False), seed=1, model_revision="rev9",
                        created_at="2026-11-07T18:30:00Z")
    path = tmp_path / "eval_baselines.json"
    path.write_text(json.dumps(doc))
    code = train_sft.main(["--hub-repo", "someone/alien-sft", "--seed", "0",
                           "--baselines", str(path), "--work-dir", str(tmp_path / "w")])
    assert code == 2
    assert "seed 1" in capsys.readouterr().err


def test_the_script_contains_no_token_shaped_text():
    assert re.search(r"hf_[A-Za-z0-9]{20,}", SCRIPT.read_text(encoding="utf-8")) is None


def test_heavy_libraries_are_imported_only_inside_functions():
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        if re.match(r"(import|from)\s+(torch|transformers|peft|trl|huggingface_hub|datasets)\b",
                    line):
            pytest.fail(f"module-level heavy import: {line}")


# ---- --hardness (E-1) -----------------------------------------------------------------------

# First train task of make_splits(0, 1) versus make_splits(0, 0): hardness 1 swaps the note.
H1_TRAIN_NOTE = "Note: at most 100 calls per minute are accepted."
H0_TRAIN_NOTE = ("Note: despite its name, a positive `change` lowers the water allowance and a "
                 "negative `change` raises it.")


def spy_on_sft_splits(monkeypatch):
    seen = {}
    real = train_sft.build_sft_rows

    def spy(splits):
        seen["splits"] = splits
        return real(splits)

    monkeypatch.setattr(train_sft, "build_sft_rows", spy)
    return seen


def test_hardness_1_builds_the_sft_rows_from_the_hardness_1_split(monkeypatch, tmp_path):
    # Breaks if train_sft ignores --hardness and calls make_splits(seed) at hardness 0.
    seen = spy_on_sft_splits(monkeypatch)
    assert train_sft.main(["--dry-run", "--hardness", "1", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 1
    assert seen["splits"].train[0].id == "train-0000"
    assert H1_TRAIN_NOTE in seen["splits"].train[0].tool.doc
    assert H0_TRAIN_NOTE not in seen["splits"].train[0].tool.doc


def test_default_hardness_builds_the_hardness_0_split(monkeypatch, tmp_path):
    seen = spy_on_sft_splits(monkeypatch)
    assert train_sft.main(["--dry-run", "--work-dir", str(tmp_path)]) == 0
    assert seen["splits"].hardness == 0
    assert H0_TRAIN_NOTE in seen["splits"].train[0].tool.doc


@pytest.mark.parametrize("value", ["-1", "99"])
def test_hardness_outside_0_to_3_is_a_parser_error_with_exit_2(capsys, tmp_path, value):
    with pytest.raises(SystemExit) as raised:
        train_sft.main(["--dry-run", "--hardness", value, "--work-dir", str(tmp_path)])
    assert raised.value.code == 2
    assert "--hardness must be between 0 and 3" in capsys.readouterr().err


def test_a_splits_file_is_accepted_at_its_own_hardness_and_rejected_at_another(
        tmp_path, capsys, monkeypatch):
    # Breaks if --splits stops checking the file's recorded hardness against --hardness.
    path = tmp_path / "splits.json"
    path.write_text(splits_to_json(make_splits(0, 1)))
    seen = spy_on_sft_splits(monkeypatch)
    argv = ["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")]
    assert train_sft.main([*argv, "--hardness", "1"]) == 0
    assert seen["splits"].hardness == 1
    with pytest.raises(SystemExit) as raised:
        train_sft.main(argv)
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "was made with seed 0 and hardness 1 but this run uses seed 0 and hardness 0" in err
    assert "Traceback" not in err


# ---- the SFT evaluation is fingerprinted by adapter and revision (review finding 2) --------


def sft_eval_args(**overrides):
    import argparse

    values = dict(seed=0, eval_batch_size=8, revision="main", hardness=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_hf_loader(monkeypatch, splits, kind_by_adapter):
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


def test_evaluate_models_refuses_a_second_adapter_over_the_same_progress_dir(
        tmp_path, monkeypatch):
    # Breaks if the SFT evaluation stops passing a run_key: adapter B would be scored on
    # adapter A's recorded outputs.
    from alien.persist import StaleProgressError

    splits = make_splits(0)
    baselines = {"models": {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4)}}
    prompts = fake_hf_loader(monkeypatch, splits, {"/ckpt/A": "oracle", "/ckpt/B": "habit"})
    first = train_sft.evaluate_models(
        sft_eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev1", baselines)
    assert first["models"]["sft"]["tool_call_f1"]["value"] == 1.0
    assert len(prompts) == 200
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        train_sft.evaluate_models(
            sft_eval_args(), splits, tmp_path, Path("/ckpt/B"), "rev1", baselines)
    assert len(prompts) == 200


def test_regenerated_baselines_are_fingerprinted_by_the_base_revision(tmp_path, monkeypatch):
    # Breaks if the regenerated base/prompt_only progress ignores the model revision.
    from alien.persist import StaleProgressError

    splits = make_splits(0)
    fake_hf_loader(monkeypatch, splits, {None: "habit", "/ckpt/A": "oracle"})
    train_sft.evaluate_models(sft_eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev1", None)
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        train_sft.evaluate_models(
            sft_eval_args(), splits, tmp_path, Path("/ckpt/A"), "rev2", None)


def test_main_exits_2_with_one_plain_line_when_the_sft_progress_dir_is_stale(
        tmp_path, monkeypatch, capsys):
    # Breaks if main lets StaleProgressError escape as a traceback instead of exit 2.
    splits = make_splits(0)
    baselines = {"models": {"base": metrics(0.7, 0.3), "prompt_only": metrics(0.6, 0.4)}}
    fake_hf_loader(monkeypatch, splits, {None: "habit", "/ckpt/A": "oracle", "/ckpt/B": "habit"})
    work = tmp_path / "w"
    train_sft.evaluate_models(sft_eval_args(), splits, work, Path("/ckpt/A"), "rev1", baselines)
    summary = {"loss_first": 2.0, "loss_last": 1.0, "steps": 5, "model_revision": "rev1"}
    monkeypatch.setattr(train_sft, "train", lambda *a, **k: {
        "code": 0, "revision": "abc", "summary": summary, "final_dir": Path("/ckpt/B")})
    code = train_sft.main(["--hub-repo", "u/r", "--work-dir", str(work),
                           "--baselines", str(tmp_path / "missing.json"),
                           "--out", str(tmp_path / "sft.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert "fresh --progress-dir" in err and "Traceback" not in err


def test_sft_dry_run_rejects_a_splits_file_that_fails_the_leak_check(tmp_path, capsys):
    # Breaks if train_sft accepts an imported splits file with a held-out task in train
    # (reviewer saw dry runs exit 0 with 601 train tasks).
    real = make_splits(0)
    path = tmp_path / "leaky.json"
    path.write_text(splits_to_json(Splits(
        [*real.train, real.heldout[0]], real.dev, real.heldout, real.heldout_twist, 0, 0)))
    with pytest.raises(SystemExit) as raised:
        train_sft.main(["--dry-run", "--splits", str(path), "--work-dir", str(tmp_path / "w")])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert f"error: {path} fails the leak check: held-out tools in train: " in err
    assert "Traceback" not in err


# ---- a finished checkpoint is reused only by the same run (review finding 4) ---------------


def cfg_args(**overrides):
    import argparse

    values = dict(seed=0, hardness=0, revision="main", full_sequence=False, hub_repo="u/r")
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_run_config_lists_the_run_identity_with_flat_hparam_keys():
    # Breaks if the recorded identity loses a key (the reuse check could not see that change).
    config = train_sft.build_run_config(
        "train_sft", cfg_args(), {"seed": 0, "learning_rate": 0.0002}, full_sequence=False)
    assert config == {
        "script": "train_sft", "seed": 0, "hardness": 0, "model_id": "CohereLabs/tiny-aya-global",
        "revision": "main", "hparams.seed": 0, "hparams.learning_rate": 0.0002,
        "full_sequence": False}


def finished_final(root, with_config=True, **arg_overrides):
    args = cfg_args(**arg_overrides)
    config = train_sft.build_run_config(
        "train_sft", args, train_sft.plan_hparams(args.seed, False),
        full_sequence=args.full_sequence)

    def write_final(directory):
        (directory / "adapter_model.safetensors").write_text("weights")
        (directory / "training_summary.json").write_text(json.dumps({
            "loss_first": 2.0, "loss_last": 1.0, "steps": 76, "model_revision": "rev9"}))
        if with_config:
            train_sft.write_run_config(directory, config, "rev9")

    from alien.persist import write_checkpoint

    write_checkpoint(root, "final", write_final)


def run_sft_train(root, **arg_overrides):
    args = cfg_args(**arg_overrides)
    return train_sft.train(args, [], root, root.parent / "trainer",
                           train_sft.plan_hparams(args.seed, False))


def test_the_same_run_reuses_its_final_checkpoint_and_only_retries_the_upload(
        tmp_path, monkeypatch):
    # Breaks if the reuse check refuses a final checkpoint that matches this run.
    root = tmp_path / "ckpt"
    finished_final(root)
    monkeypatch.setattr(train_sft, "hub_upload", lambda folder, repo: "r2")
    outcome = run_sft_train(root)
    assert outcome["code"] == 0 and outcome["revision"] == "r2"
    assert outcome["summary"]["steps"] == 76


@pytest.mark.parametrize("overrides, differing", [
    ({"seed": 1}, "hparams.seed, seed"),
    ({"hardness": 2}, "hardness"),
    ({"revision": "NEW_BASE"}, "revision"),
    ({"full_sequence": True}, "full_sequence"),
])
def test_a_final_checkpoint_of_another_run_is_refused_naming_the_differing_keys(
        tmp_path, capsys, monkeypatch, overrides, differing):
    # Breaks if train() trusts a manifest-listed final checkpoint without comparing the run.
    root = tmp_path / "ckpt"
    finished_final(root)
    monkeypatch.setattr(train_sft, "hub_upload", lambda *a: pytest.fail("must not upload"))
    outcome = run_sft_train(root, **overrides)
    assert outcome == {"code": 2, "revision": None, "summary": None, "final_dir": None}
    assert capsys.readouterr().err == (
        f"error: {root / 'final'} was made by a different run (differs in: {differing}); "
        "use a fresh --work-dir\n")


def test_a_final_checkpoint_with_no_recorded_config_is_refused(tmp_path, capsys, monkeypatch):
    # Breaks if a checkpoint written by older code (no run_config.json) is trusted.
    root = tmp_path / "ckpt"
    finished_final(root, with_config=False)
    monkeypatch.setattr(train_sft, "hub_upload", lambda *a: pytest.fail("must not upload"))
    assert run_sft_train(root)["code"] == 2
    assert capsys.readouterr().err == (
        f"error: {root / 'final'} has no run_config.json (written by older code); "
        "use a fresh --work-dir\n")


def test_a_recorded_config_that_is_not_a_json_object_is_refused(tmp_path, capsys):
    # Breaks if a torn or hand-edited run_config.json crashes with a traceback.
    root = tmp_path / "ckpt"
    finished_final(root)
    (root / "final" / "run_config.json").write_text("[1, 2")
    assert run_sft_train(root)["code"] == 2
    assert "has no readable run_config.json" in capsys.readouterr().err
