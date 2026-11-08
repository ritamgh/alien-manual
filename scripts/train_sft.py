"""Resumable fp16 LoRA SFT of Tiny Aya Global on oracle traces (PRD FR-005, S3, E-6, R-4).

Trains on the TRAIN split only (B-2), keeps every checkpoint on local persistent storage first
(with a completion manifest), then uploads the final adapter to the HF Hub with three bounded
attempts, then evaluates the SFT model on the held-out split next to base and prompt-only and
writes results/sft.json (contract kind ``sft``).

Colab usage (GPU needed; HF_TOKEN in the environment or a cached hub login, never in a file):

    python scripts/train_sft.py --hub-repo <user>/alien-sft --seed 0
    python scripts/check_results.py results/sft.json

Re-running the same command after an interrupted session resumes from the newest checkpoint in
``--work-dir`` (put it on Drive). If the upload fails, the local checkpoint stays untouched, the
script exits 1, and re-running retries only the upload and the evaluation. SFT is the guaranteed
result (R-4). Other switches: ``--dry-run`` prints the plan without importing any heavy library;
``--smoke`` is 8 steps on 16 rows at max_length 256 with no upload and no results file (for the
8 GB laptop GPU).

A finished ``ckpt/final`` records the run it came from in ``run_config.json`` (script, seed,
hardness, base model and revision, hyper-parameters); the train scripts reuse a final checkpoint
only for an identical run, else they exit 2 and ask for a fresh ``--work-dir``.

Exits 0 on success, 1 when the upload fails or sft.json fails validation, 2 on bad arguments or a
stale evaluation progress directory (another revision or adapter: use a fresh work dir).
torch, transformers, peft, trl, datasets and huggingface_hub are imported only inside functions.
Anything not checkable without a GPU is marked VERIFY ON COLAB.

Layout under ``--work-dir`` (``<work-dir>/smoke`` for ``--smoke``): ``ckpt/`` holds the
manifest-listed checkpoints (``step-N`` and ``final``), ``trainer/`` is the Trainer's own
output, ``progress_sft/`` and ``progress_baselines/`` hold the resumable eval progress.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_results  # noqa: E402
from alien.eval import (  # noqa: E402
    DECODING,
    MODEL_ID,
    ResultsInvalidError,
    _now,
    evaluate_variants,
    write_results,
)
from alien.generator import Task  # noqa: E402
from alien.oracle import to_chat_example  # noqa: E402
from alien.persist import (  # noqa: E402
    StaleProgressError,
    UploadError,
    latest_checkpoint,
    load_checkpoints,
    upload_with_retry,
    write_checkpoint,
)
from alien.scoring import NoEligibleExamplesError  # noqa: E402
from alien.splits import Splits  # noqa: E402
from run_eval import add_hardness_argument, load_splits_checked, make_run_key  # noqa: E402

SFT_HPARAMS: dict[str, Any] = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_length": 1024,
    "warmup_ratio": 0.03,
    "seed": 0,
    "fp16": True,
    "save_steps": 50,
    "logging_steps": 5,
}
SMOKE_OVERRIDES: dict[str, Any] = {"max_steps": 8, "max_length": 256, "save_steps": 4}
SMOKE_ROWS = 16
EVAL_N = 200  # same held-out slice as scripts/run_eval.py, so the three models are comparable
FINAL_NAME = "final"
SUMMARY_FILE = "training_summary.json"
RUN_CONFIG_FILE = "run_config.json"
UPLOAD_ATTEMPTS = 3
DRIVE_ROOT = Path("/content/drive/MyDrive")


class NoLossLoggedError(ValueError):
    """The Trainer log history holds no entry with a training loss."""


# ---- pure helpers ---------------------------------------------------------------------------


def default_work_dir(drive_root: Path = DRIVE_ROOT) -> Path:
    """Drive folder when Drive is mounted (Colab), else ./work."""
    return drive_root / "alien_manual" if drive_root.is_dir() else Path("work")


def plan_hparams(seed: int, smoke: bool) -> dict[str, Any]:
    hparams = dict(SFT_HPARAMS, seed=seed)
    if smoke:
        hparams.update(SMOKE_OVERRIDES)
    return hparams


def build_sft_rows(splits: Splits) -> list[dict]:
    """One ``{"messages": [user, assistant]}`` row per TRAIN task. Never touches dev/held-out."""
    return [to_chat_example(task) for task in splits.train]


def to_prompt_completion(row: dict) -> dict:
    """Split a chat row so TRL computes the loss on the assistant reply only."""
    *prompt, completion = row["messages"]
    return {"prompt": prompt, "completion": [completion]}


def extract_losses(log_history: Sequence[dict]) -> tuple[float, float, int]:
    """First and last logged training loss and the step of the last one.

    Entries without a ``loss`` key (eval results, the final train summary) are ignored.
    """
    logged = [entry for entry in log_history if "loss" in entry]
    if not logged:
        raise NoLossLoggedError(
            "no training loss was logged: log_history has no entry with a 'loss' key"
        )
    return float(logged[0]["loss"]), float(logged[-1]["loss"]), int(logged[-1]["step"])


def build_sft_results(
    *,
    evaluated: dict[str, Any],
    loss_first: float,
    loss_last: float,
    steps: int,
    hub_repo: str,
    revision: str | None,
    model_revision: str,
    seed: int,
    git_sha: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The ``sft`` results document; ``evaluated["models"]`` holds base, prompt_only and sft."""
    wanted = ("base", "prompt_only", "sft")
    missing = [name for name in wanted if name not in evaluated["models"]]
    if missing:
        raise ValueError(f"sft results need models {', '.join(wanted)}; "
                         f"missing {', '.join(missing)}")
    return {
        "kind": "sft",
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": git_sha or os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "model_revision": model_revision,
        "checkpoint": {"hub_repo": hub_repo, "revision": revision},
        "loss_first": float(loss_first),
        "loss_last": float(loss_last),
        "steps": steps,
        "models": {name: evaluated["models"][name] for name in wanted},
    }


def load_baselines(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate an ``eval_baselines`` file; its ``models`` are base and prompt_only."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = check_results.validate(doc)
    if errors:
        raise ResultsInvalidError("\n".join(errors))
    if doc["kind"] != "eval_baselines":
        raise ResultsInvalidError(f"kind: expected 'eval_baselines', got {doc['kind']!r}")
    return doc


def mirror_checkpoint(src: Path, root: Path, name: str) -> None:
    """Copy a Trainer checkpoint into the manifest-listed store, once (E-6)."""
    if any(path.name == name for path in load_checkpoints(root)):
        return
    write_checkpoint(root, name, lambda directory: shutil.copytree(
        src, directory, dirs_exist_ok=True))


def resume_state(root: Path) -> tuple[str, Path | None]:
    """``("fresh", None)``, ``("resume", newest step checkpoint)`` or ``("final", final dir)``."""
    newest = latest_checkpoint(root)
    if newest is None:
        return "fresh", None
    return ("final" if newest.name == FINAL_NAME else "resume"), newest


def build_run_config(script: str, args: argparse.Namespace, hparams: dict[str, Any],
                     **extra: Any) -> dict[str, Any]:
    """What makes a trained checkpoint this run's: the script, seed, hardness, base model and
    revision as requested, the hyper-parameters (keys ``hparams.<name>``) and ``extra``."""
    config = {"script": script, "seed": args.seed, "hardness": args.hardness,
              "model_id": MODEL_ID, "revision": args.revision,
              **{f"hparams.{key}": value for key, value in hparams.items()}, **extra}
    return json.loads(json.dumps(config))


def write_run_config(directory: Path, config: dict[str, Any], model_revision: str) -> None:
    """Record ``config`` (plus the resolved base commit) inside the final checkpoint, so the
    manifest never lists a final checkpoint without it."""
    (directory / RUN_CONFIG_FILE).write_text(
        json.dumps({**config, "resolved_revision": model_revision}, indent=1), encoding="utf-8")


def refuse_reused_final(final_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """None when ``final_dir`` was made by the run described by ``config``; else a ``train``
    outcome with exit code 2 after one plain line on stderr.

    Only the keys of ``config`` are compared (the recorded resolved revision is information, it
    cannot be known without loading the model). A final checkpoint with no readable recorded
    config, such as one from older code, is refused too.
    """
    path = final_dir / RUN_CONFIG_FILE
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(recorded, dict):
            raise ValueError("not a JSON object")
    except FileNotFoundError:
        problem = f"has no {RUN_CONFIG_FILE} (written by older code)"
    except ValueError:
        problem = f"has no readable {RUN_CONFIG_FILE}"
    else:
        differing = sorted(key for key, value in config.items() if recorded.get(key) != value)
        if not differing:
            return None
        problem = f"was made by a different run (differs in: {', '.join(differing)})"
    print(f"error: {final_dir} {problem}; use a fresh --work-dir", file=sys.stderr)
    return {"code": 2, "revision": None, "summary": None, "final_dir": None}


def finish_and_upload(
    root: Path,
    write_adapter: Callable[[Path], None] | None,
    upload: Callable[[Path], str | None],
    *,
    attempts: int = UPLOAD_ATTEMPTS,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = print,
) -> tuple[int, str | None, Path]:
    """Write the final adapter checkpoint locally, THEN upload it with bounded retries.

    Returns ``(exit_code, hub_revision, final_dir)``. When the upload still fails the
    checkpoint stays on disk, nothing is deleted, and the code is 1 (R-4). If the final
    checkpoint is already listed (a re-run), it is not written again, only uploaded.
    """
    final_dir = root / FINAL_NAME
    if not any(path.name == FINAL_NAME for path in load_checkpoints(root)):
        if write_adapter is None:
            raise ValueError("no final checkpoint exists and no adapter writer was given")
        final_dir = write_checkpoint(root, FINAL_NAME, write_adapter)
    try:
        revision = upload_with_retry(
            lambda: upload(final_dir), attempts=attempts, base_delay=base_delay, sleep=sleep)
    except UploadError as exc:
        say(f"The upload to the HF Hub failed ({exc}). The trained adapter is saved at "
            f"{final_dir} and nothing was deleted. Check HF_TOKEN and the repo name, then run "
            "the same command again: it skips training and retries the upload.")
        return 1, None, final_dir
    return 0, revision, final_dir


# ---- GPU path (Colab only) ------------------------------------------------------------------


def hub_upload(folder: Path, hub_repo: str) -> str | None:
    """Upload the adapter folder; return the commit hash. The token comes from the environment
    through the hub library and is never read, stored or printed here.

    With ``ALIEN_NO_UPLOAD=1`` nothing is published: the adapter stays on local disk and the
    revision is None (the checker accepts a null revision)."""
    if os.environ.get("ALIEN_NO_UPLOAD"):
        print("hub upload skipped (ALIEN_NO_UPLOAD is set): the adapter stays local")
        return None
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub_repo, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=hub_repo,
                             commit_message="SFT LoRA adapter")
    return getattr(info, "oid", None)  # VERIFY ON COLAB: CommitInfo.oid is the commit hash


def train(
    args: argparse.Namespace, rows: list[dict], ckpt_root: Path, trainer_dir: Path,
    hparams: dict[str, Any],
) -> dict[str, Any]:
    """Train (or resume), then finish_and_upload. Returns code, revision, summary, final_dir.

    A re-run whose ``final`` checkpoint is complete skips training and only retries the upload,
    but only if that checkpoint's ``run_config.json`` matches this run; else exit code 2.
    """
    state, newest = resume_state(ckpt_root)
    config = build_run_config("train_sft", args, hparams, full_sequence=args.full_sequence)

    def upload(folder: Path) -> str | None:
        return hub_upload(folder, args.hub_repo)

    if state == "final":
        refusal = refuse_reused_final(newest, config)
        if refusal is not None:
            return refusal
        summary = json.loads((newest / SUMMARY_FILE).read_text(encoding="utf-8"))
        print(f"final adapter already saved at {newest}; skipping training")
        code, revision, final_dir = finish_and_upload(ckpt_root, None, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    # VERIFY ON COLAB: transformers 5 renamed torch_dtype to dtype (as in the spike notebook).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, dtype=torch.float16, device_map={"": 0})
    model_revision = getattr(model.config, "_commit_hash", None) or args.revision

    if args.full_sequence:
        dataset = Dataset.from_list(rows)  # loss on every token
    else:
        # VERIFY ON COLAB: a prompt/completion dataset makes recent TRL compute the loss on the
        # completion only. If this TRL version ignores that, rerun with --full-sequence.
        dataset = Dataset.from_list([to_prompt_completion(row) for row in rows])
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(trainer_dir),
        "num_train_epochs": hparams["num_train_epochs"],
        "per_device_train_batch_size": hparams["per_device_train_batch_size"],
        "gradient_accumulation_steps": hparams["gradient_accumulation_steps"],
        "learning_rate": hparams["learning_rate"],
        "warmup_steps": hparams["warmup_ratio"],  # transformers 5: a float below 1 is a ratio of total steps
        "fp16": hparams["fp16"],
        "logging_steps": hparams["logging_steps"],
        "save_strategy": "steps",
        "save_steps": hparams["save_steps"],
        "save_total_limit": 1,
        "max_length": hparams["max_length"],  # VERIFY ON COLAB: older TRL: max_seq_length
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": hparams["seed"],
    }
    if "max_steps" in hparams:
        sft_kwargs["max_steps"] = hparams["max_steps"]
    lora = LoraConfig(
        r=hparams["lora_r"], lora_alpha=hparams["lora_alpha"],
        lora_dropout=hparams["lora_dropout"], target_modules=hparams["target_modules"],
        task_type="CAUSAL_LM",
    )

    class MirrorToManifest(TrainerCallback):
        """After each Trainer save, copy the checkpoint into the manifest-listed store."""

        def on_save(self, targs, tstate, control, **kwargs):
            src = Path(targs.output_dir) / f"checkpoint-{tstate.global_step}"
            mirror_checkpoint(src, ckpt_root, f"step-{tstate.global_step}")

    trainer = SFTTrainer(
        model=model, args=SFTConfig(**sft_kwargs), train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora, callbacks=[MirrorToManifest()],
    )
    # fp16 mixed precision needs fp32 trainable weights; peft usually does this already.
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
    try:
        if newest is not None:
            # VERIFY ON COLAB: resuming restores the LoRA weights, optimizer and step count.
            print(f"resuming from {newest}")
        trainer.train(resume_from_checkpoint=str(newest) if newest is not None else None)
        loss_first, loss_last, _ = extract_losses(trainer.state.log_history)
        summary = {"loss_first": loss_first, "loss_last": loss_last,
                   "steps": int(trainer.state.global_step), "model_revision": model_revision}
        print(f"loss: first {loss_first:.4f}, last {loss_last:.4f}, steps {summary['steps']}")
        if args.smoke:
            return {"code": 0, "revision": None, "summary": summary, "final_dir": None}

        def write_adapter(directory: Path) -> None:
            trainer.model.save_pretrained(str(directory))
            (directory / SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
            write_run_config(directory, config, model_revision)

        code, revision, final_dir = finish_and_upload(ckpt_root, write_adapter, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}
    finally:
        trainer = model = None  # free the GPU before the evaluation loads its own copy
        gc.collect()
        torch.cuda.empty_cache()


def _summary_line(name: str, metrics: dict[str, Any]) -> str:
    return f"{name}: " + "; ".join(
        f"{key} {metrics[key]['value']:.3f}" for key in check_results.METRIC_KEYS)


def evaluate_models(
    args: argparse.Namespace, splits: Splits, work: Path, final_dir: Path, model_revision: str,
    baselines: dict[str, Any] | None,
) -> dict[str, Any]:
    """Held-out metrics for base, prompt_only (from the file, or regenerated) and sft."""
    from run_eval import make_hf_generate  # heavy imports happen inside it

    tasks: list[Task] = splits.heldout[:EVAL_N]

    def run_key(adapter: str | None) -> str:
        return make_run_key(model_id=MODEL_ID, revision=args.revision,
                            resolved_revision=model_revision, adapter=adapter,
                            hardness=args.hardness)

    models: dict[str, Any] = {}
    if baselines is None:
        print("results/eval_baselines.json not found: regenerating base and prompt_only on the "
              "held-out split now (this costs GPU time; the file is not written)")
        base = make_hf_generate(MODEL_ID, model_revision, None, DECODING)
        models.update(evaluate_variants(
            tasks, {"base": base, "prompt_only": base}, work / "progress_baselines",
            seed=args.seed, batch_size=args.eval_batch_size, run_key=run_key(None))["models"])
        base = None
        gc.collect()
    else:
        models.update(baselines["models"])
    tuned = make_hf_generate(MODEL_ID, model_revision, str(final_dir), DECODING)
    # evaluate_variants knows the SFT model as the "tuned" variant (plain prompt, FR-006).
    sft = evaluate_variants(tasks, {"tuned": tuned}, work / "progress_sft", seed=args.seed,
                            batch_size=args.eval_batch_size, run_key=run_key(str(final_dir)))
    models["sft"] = sft["models"]["tuned"]
    return {"models": models, "gpu_seconds": sft["gpu_seconds"], "n_examples": len(tasks)}


# ---- command line ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable fp16 LoRA SFT on oracle traces, then held-out eval (S3).")
    parser.add_argument("--hub-repo", help="HF Hub repo for the adapter, e.g. user/alien-sft")
    parser.add_argument("--seed", type=int, default=0, help="splits, training and bootstrap seed")
    parser.add_argument("--work-dir", help="persistent storage (default Drive when mounted, "
                                           "else ./work)")
    parser.add_argument("--splits", help="JSON file from splits_to_json (default: make_splits)")
    add_hardness_argument(parser)
    parser.add_argument("--revision", default="main", help="base model hub revision")
    parser.add_argument("--baselines", default="results/eval_baselines.json")
    parser.add_argument("--out", default="results/sft.json")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--full-sequence", action="store_true",
                        help="loss on every token instead of the assistant reply only")
    parser.add_argument("--smoke", action="store_true",
                        help="8 steps, 16 rows, max_length 256, no upload, no results file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without importing any heavy library")
    return parser


def _print_plan(args: argparse.Namespace, rows: list[dict], hparams: dict[str, Any],
                ckpt_root: Path, work: Path, baselines_found: bool) -> None:
    state, newest = resume_state(ckpt_root)
    print(f"model: {MODEL_ID} (revision {args.revision}), fp16 LoRA")
    print(f"seed: {args.seed}")
    print(f"train rows: {len(rows)} (train split only)")
    print(f"held-out examples to evaluate: {EVAL_N}")
    print(f"loss on: {'every token' if args.full_sequence else 'assistant reply only'}")
    print(f"work dir: {work}")
    print(f"resume: {state}" + (f" ({newest})" if newest is not None else ""))
    if args.smoke:
        print("hub upload: none (smoke)")
        print("results file: none (smoke)")
    else:
        print(f"hub upload: {args.hub_repo or '(--hub-repo missing)'}, "
              f"{UPLOAD_ATTEMPTS} attempts")
        print(f"results file: {args.out}")
        print(f"baselines: {args.baselines} "
              f"({'found' if baselines_found else 'missing: would be regenerated'})")
    print("hyper-parameters: " + json.dumps(hparams, indent=1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be >= 1")
    if not (args.smoke or args.dry_run) and not args.hub_repo:
        parser.error("--hub-repo is required (not with --smoke or --dry-run)")
    splits = load_splits_checked(parser, args.splits, args.seed, args.hardness)
    rows = build_sft_rows(splits)
    if args.smoke:
        rows = rows[:SMOKE_ROWS]
    hparams = plan_hparams(args.seed, args.smoke)
    work = Path(args.work_dir) if args.work_dir else default_work_dir()
    if args.smoke:
        work = work / "smoke"
    ckpt_root = work / "ckpt"
    baselines_path = Path(args.baselines)
    baselines_found = baselines_path.exists()

    if args.dry_run:
        _print_plan(args, rows, hparams, ckpt_root, work, baselines_found)
        return 0

    baselines = None
    if not args.smoke and baselines_found:
        try:
            baselines = load_baselines(baselines_path)
        except ResultsInvalidError as exc:
            print(f"error: {baselines_path} is not a valid eval_baselines file:\n{exc}",
                  file=sys.stderr)
            return 2
        if baselines["seed"] != args.seed:
            print(f"error: {baselines_path} was made with seed {baselines['seed']} but this run "
                  f"uses seed {args.seed}, so the held-out examples differ; pass --seed "
                  f"{baselines['seed']} or regenerate the baselines", file=sys.stderr)
            return 2

    _print_plan(args, rows, hparams, ckpt_root, work, baselines_found)
    outcome = train(args, rows, ckpt_root, work / "trainer", hparams)
    if outcome["code"] != 0 or args.smoke:
        return outcome["code"]

    summary = outcome["summary"]
    try:
        evaluated = evaluate_models(args, splits, work, outcome["final_dir"],
                                    summary["model_revision"], baselines)
    except NoEligibleExamplesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except StaleProgressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, metrics in evaluated["models"].items():
        print(_summary_line(name, metrics))
    doc = build_sft_results(
        evaluated=evaluated, loss_first=summary["loss_first"], loss_last=summary["loss_last"],
        steps=summary["steps"], hub_repo=args.hub_repo, revision=outcome["revision"],
        model_revision=summary["model_revision"], seed=args.seed)
    try:
        write_results(args.out, doc, check_results.validate)
    except ResultsInvalidError as exc:
        print(f"error: {args.out} was not written, it failed validation:\n{exc}\n"
              f"The adapter is safe at {outcome['final_dir']} and on the Hub.", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
