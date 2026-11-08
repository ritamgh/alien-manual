"""Resumable fp16 LoRA GRPO from the SFT adapter with an exact-match reward (PRD FR-003, FR-005).

Slice S4 of the Alien Manual plan; covers E-2, E-3, E-6, R-3, B-1, B-3, DEC-1 and DEC-2. Continues
training the SFT adapter on the TRAIN split only (B-2). The reward is ``alien.reward`` exact match
(1.0 only for the oracle call, else 0.0) with at least 4 generations per prompt. A
``RewardMonitor`` counts reward groups that vary; when every group so far is identical the run
stops, keeps its checkpoints and exits 3 so the rejection-sampling fallback
(``scripts/train_rft.py``, DEC-1) can run instead, either by hand or with ``--auto-fallback``.
Which path really ran is recorded in the results file (``path``), never assumed.

Colab usage (GPU needed; HF_TOKEN in the environment or a cached hub login, never in a file):

    python scripts/train_grpo.py --hub-repo <user>/alien-grpo --seed 0
    python scripts/check_results.py results/grpo.json

It needs results/sft.json and results/eval_baselines.json made with the same ``--seed``, and the
SFT adapter (default ``<work-dir>/ckpt/final`` from scripts/train_sft.py, or ``--sft-adapter``).
Re-running the same command after an interrupted session resumes from the newest checkpoint of
this run; if the upload failed it only retries the upload and the evaluation. ``--dry-run``
prints the plan without importing any heavy library; ``--smoke`` is 3 steps on 16 rows with no
upload and no results file (it needs ``--sft-adapter``).

Exit codes: 0 ok, 1 upload failed or the results file failed validation, 2 bad arguments or
inputs (or a stale evaluation progress dir), 3 the reward is dead (checkpoints kept, run
scripts/train_rft.py).

This module also holds what both RL scripts share: the results builder ``build_grpo_results``
(kind ``grpo`` or ``rft``, chosen by ``path``), the input checks, the held-out evaluation and the
E-3 forgetting warning. scripts/train_rft.py imports them from here.

torch, transformers, peft, trl, datasets and huggingface_hub are imported only inside functions.
Anything not checkable without a GPU is marked VERIFY ON COLAB.

Layout under ``--work-dir``: ``grpo/`` (``grpo_smoke/`` for ``--smoke``) holds ``ckpt/`` with the
manifest-listed checkpoints (``step-N`` and ``final``), ``trainer/`` (the Trainer's own output)
and ``progress_tuned/`` (resumable evaluation progress).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import json
import os
import sys
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
from alien.reward import (  # noqa: E402
    DeadRewardError,
    RewardMonitor,
    make_reward_func,
    to_grpo_rows,
)
from alien.persist import StaleProgressError  # noqa: E402
from alien.scoring import NoEligibleExamplesError  # noqa: E402
from alien.splits import Splits  # noqa: E402
from run_eval import add_hardness_argument, load_splits_checked, make_run_key  # noqa: E402
from train_sft import (  # noqa: E402
    EVAL_N,
    SUMMARY_FILE,
    UPLOAD_ATTEMPTS,
    _summary_line,
    default_work_dir,
    build_run_config,
    finish_and_upload,
    load_baselines,
    mirror_checkpoint,
    refuse_reused_final,
    resume_state,
    write_run_config,
)

GRPO_HPARAMS: dict[str, Any] = {
    "num_generations": 4,
    "learning_rate": 5e-6,
    "per_device_train_batch_size": 4,  # completions per device: 4 % num_generations == 0
    "gradient_accumulation_steps": 4,  # 16 completions = 4 prompts per optimizer step
    "max_prompt_length": 512,
    "max_completion_length": 256,
    "max_steps": 100,  # 400 prompts, under the 600 train tasks: one pass, short for a T4 session
    "seed": 0,
    "fp16": True,
    "save_steps": 50,
    "logging_steps": 5,
}
SMOKE_OVERRIDES: dict[str, Any] = {"max_steps": 3, "save_steps": 2, "max_completion_length": 128}
SMOKE_ROWS = 16
RL_PATHS = ("grpo", "rft")
RL_MODELS = ("base", "prompt_only", "sft", "tuned")
FORGETTING_DROP = 0.05  # E-3: warn when normal_tool_f1 falls by more than this versus base
MONITOR_FILE = "reward_monitor.json"
EXIT_OK, EXIT_FAILED, EXIT_BAD_INPUT, EXIT_DEAD_REWARD = 0, 1, 2, 3
DEAD_MESSAGE = ("reward is dead: switching to rejection-sampling fine-tuning, "
                "see scripts/train_rft.py")


class InputsError(ValueError):
    """A required input file is missing, invalid or made with another seed."""


# ---- pure helpers ---------------------------------------------------------------------------


def plan_grpo_hparams(seed: int, smoke: bool) -> dict[str, Any]:
    hparams = dict(GRPO_HPARAMS, seed=seed)
    if smoke:
        hparams.update(SMOKE_OVERRIDES)
    return hparams


def train_tasks(splits: Splits, n: int | None = None) -> list[Task]:
    """The first ``n`` TRAIN tasks (all when ``n`` is None). Never touches dev or held-out."""
    return splits.train[:n]


def build_grpo_rows(splits: Splits, n: int | None = None) -> list[dict]:
    """GRPO rows ``{"prompt": [user message], "task_id": id}`` from the TRAIN split only (B-2)."""
    return to_grpo_rows(train_tasks(splits, n))


def build_grpo_results(
    *,
    path: str,
    evaluated: dict[str, Any],
    hub_repo: str,
    revision: str | None,
    generations_per_prompt: int,
    reward_summary: dict[str, int],
    seed: int,
    git_sha: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The ``grpo`` or ``rft`` results document, chosen by ``path``.

    ``evaluated["models"]`` holds base, prompt_only, sft and tuned. ``reward_summary`` is a
    ``RewardMonitor.summary()``.
    """
    if path not in RL_PATHS:
        raise ValueError(f"path must be one of {', '.join(RL_PATHS)}, got {path!r}")
    missing = [name for name in RL_MODELS if name not in evaluated["models"]]
    if missing:
        raise ValueError(f"{path} results need models {', '.join(RL_MODELS)}; "
                         f"missing {', '.join(missing)}")
    return {
        "kind": path,
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": git_sha or os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "path": path,
        "checkpoint": {"hub_repo": hub_repo, "revision": revision},
        "generations_per_prompt": generations_per_prompt,
        "reward_signal": {
            "groups_total": int(reward_summary["groups_total"]),
            "groups_with_variance": int(reward_summary["groups_with_variance"]),
        },
        "models": {name: evaluated["models"][name] for name in RL_MODELS},
    }


def load_prior_results(baselines_path: str | os.PathLike[str],
                       sft_path: str | os.PathLike[str], seed: int) -> dict[str, Any]:
    """The validated eval_baselines and sft documents; both must carry ``seed``.

    Raises ``InputsError`` with a plain message otherwise (the caller exits 2 before training).
    """
    baselines_path, sft_path = Path(baselines_path), Path(sft_path)
    if not baselines_path.exists():
        raise InputsError(f"{baselines_path} not found: run scripts/run_eval.py first")
    if not sft_path.exists():
        raise InputsError(f"{sft_path} not found: run scripts/train_sft.py first")
    try:
        baselines = load_baselines(baselines_path)
    except ResultsInvalidError as exc:
        raise InputsError(
            f"{baselines_path} is not a valid eval_baselines file:\n{exc}") from exc
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    errors = check_results.validate(sft)
    if errors or sft["kind"] != "sft":
        raise InputsError(f"{sft_path} is not a valid sft results file:\n"
                          + "\n".join(errors or [f"kind: expected 'sft', got {sft['kind']!r}"]))
    for label, path, doc in (("baselines", baselines_path, baselines), ("sft", sft_path, sft)):
        if doc["seed"] != seed:
            raise InputsError(
                f"{path} was made with seed {doc['seed']} but this run uses seed {seed}, so the "
                f"held-out examples differ; pass --seed {doc['seed']} or redo the {label} step")
    return {"baselines": baselines, "sft": sft}


def collect_models(prior: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    """base and prompt_only from eval_baselines, sft from sft.json, plus the tuned metrics."""
    return {
        "base": prior["baselines"]["models"]["base"],
        "prompt_only": prior["baselines"]["models"]["prompt_only"],
        "sft": prior["sft"]["models"]["sft"],
        "tuned": tuned,
    }


def forgetting_warning(models: dict[str, Any]) -> str | None:
    """E-3: a one-line warning when tuned normal_tool_f1 fell more than 0.05 below base."""
    base = models["base"]["normal_tool_f1"]["value"]
    tuned = models["tuned"]["normal_tool_f1"]["value"]
    if round(base - tuned, 9) > FORGETTING_DROP:
        return (f"warning: normal_tool_f1 fell from {base:.3f} (base) to {tuned:.3f} (tuned), "
                "more than 0.05: tuning may have hurt normal tools (E-3)")
    return None


def print_summary(models: dict[str, Any], say: Callable[[str], None] = print) -> None:
    for name in RL_MODELS:
        say(_summary_line(name, models[name]))
    warning = forgetting_warning(models)
    if warning:
        say(warning)


def announce_first_call(reward_fn: Callable[..., list[float]],
                        say: Callable[[str], None] = print) -> Callable[..., list[float]]:
    """B-1: print ``completions[0]`` and ``kwargs.keys()`` the first time the reward is called.

    Nothing else is printed; the reward function never sees or prints a token.
    """
    seen = []

    @functools.wraps(reward_fn)
    def wrapper(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
        if not seen:
            seen.append(True)
            say(f"first reward call: completions[0] = {completions[0]!r}"[:2000]
                if completions else "first reward call: no completions")
            say(f"first reward call: kwargs.keys() = {sorted(kwargs)}")
        return reward_fn(prompts, completions, **kwargs)

    return wrapper


def restore_monitor(monitor: RewardMonitor, checkpoint: Path) -> None:
    """Reload the reward counters saved beside a checkpoint, so a resumed run keeps them."""
    saved = checkpoint / MONITOR_FILE
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        monitor.groups_total = int(data["groups_total"])
        monitor.groups_with_variance = int(data["groups_with_variance"])


def run_dir(work_root: Path, name: str, smoke: bool) -> Path:
    return work_root / (f"{name}_smoke" if smoke else name)


def resolve_sft_adapter(args: argparse.Namespace, work_root: Path) -> str:
    """``--sft-adapter`` (a local path or a hub id), else the SFT run's ``ckpt/final``."""
    if args.sft_adapter:
        return args.sft_adapter
    default = work_root / "ckpt" / "final"
    if not default.is_dir():
        raise InputsError(f"no SFT adapter at {default}: run scripts/train_sft.py first, or pass "
                          "--sft-adapter with a local path or a hub id")
    return str(default)


# ---- GPU path (Colab only) ------------------------------------------------------------------


def hub_upload(folder: Path, hub_repo: str, message: str) -> str | None:
    """Upload the adapter folder; return the commit hash. The token comes from the environment
    through the hub library and is never read, stored or printed here."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub_repo, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=hub_repo, commit_message=message)
    return getattr(info, "oid", None)  # VERIFY ON COLAB: CommitInfo.oid is the commit hash


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_callbacks(ckpt_root: Path, monitor: RewardMonitor | None = None) -> list[Any]:
    """Trainer callbacks: mirror each saved checkpoint into the manifest-listed store (E-6) and,
    with a monitor, save its counters beside it and stop the run when the reward is dead."""
    from transformers import TrainerCallback

    class MirrorToManifest(TrainerCallback):
        def on_save(self, targs, tstate, control, **kwargs):
            src = Path(targs.output_dir) / f"checkpoint-{tstate.global_step}"
            if monitor is not None:
                (src / MONITOR_FILE).write_text(json.dumps(monitor.summary()), encoding="utf-8")
            mirror_checkpoint(src, ckpt_root, f"step-{tstate.global_step}")

    class StopWhenRewardDead(TrainerCallback):
        def on_step_end(self, targs, tstate, control, **kwargs):
            monitor.raise_if_dead()  # DeadRewardError propagates out of trainer.train()

    return [MirrorToManifest()] + ([StopWhenRewardDead()] if monitor is not None else [])


def build_grpo_trainer(
    args: argparse.Namespace, rows: list[dict], reward_fn: Callable[..., list[float]],
    monitor: RewardMonitor, ckpt_root: Path, trainer_dir: Path, hparams: dict[str, Any],
) -> tuple[Any, str]:
    """Load base + the SFT adapter (trainable) and build the GRPOTrainer. Returns the trainer and
    the base model revision."""
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # VERIFY ON COLAB: transformers 5 renamed torch_dtype to dtype (as in the SFT script).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, dtype=torch.float16, device_map={"": 0})
    model_revision = getattr(model.config, "_commit_hash", None) or args.revision
    # Continue the SFT LoRA weights instead of starting a new adapter, so the saved adapter is
    # complete on top of the base model. VERIFY ON COLAB: is_trainable makes them trainable and
    # GRPOTrainer accepts an already wrapped PeftModel.
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    kwargs: dict[str, Any] = {
        "output_dir": str(trainer_dir),
        "max_steps": hparams["max_steps"],
        "num_generations": hparams["num_generations"],
        "per_device_train_batch_size": hparams["per_device_train_batch_size"],
        "gradient_accumulation_steps": hparams["gradient_accumulation_steps"],
        "max_completion_length": hparams["max_completion_length"],
        "learning_rate": hparams["learning_rate"],
        "fp16": hparams["fp16"],
        "logging_steps": hparams["logging_steps"],
        "save_strategy": "steps",
        "save_steps": hparams["save_steps"],
        "save_total_limit": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": hparams["seed"],
    }
    # VERIFY ON COLAB: recent TRL versions may have dropped max_prompt_length from GRPOConfig.
    if "max_prompt_length" in {f.name for f in dataclasses.fields(GRPOConfig)}:
        kwargs["max_prompt_length"] = hparams["max_prompt_length"]
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=GRPOConfig(**kwargs),
        train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
        callbacks=build_callbacks(ckpt_root, monitor),
    )
    # fp16 mixed precision needs fp32 trainable weights; peft usually does this already.
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
    return trainer, model_revision


def train(
    args: argparse.Namespace, rows: list[dict], tasks: list[Task], ckpt_root: Path,
    trainer_dir: Path, hparams: dict[str, Any], *,
    make_trainer: Callable[..., tuple[Any, str]] | None = None,
    upload: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    """Train (or resume), then finish_and_upload. Returns code, revision, summary, final_dir.

    A dead reward (E-2) stops the run with code 3 and leaves every mirrored checkpoint in place.
    A re-run whose ``final`` checkpoint is complete skips training and only retries the upload,
    but only if that checkpoint's ``run_config.json`` matches this run; else exit code 2.
    """
    generations = hparams["num_generations"]
    monitor = RewardMonitor(num_generations=generations)
    upload = upload or (lambda folder: hub_upload(folder, args.hub_repo, "GRPO LoRA adapter"))
    state, newest = resume_state(ckpt_root)
    config = build_run_config("train_grpo", args, hparams, sft_adapter=args.sft_adapter)
    if state == "final":
        refusal = refuse_reused_final(newest, config)
        if refusal is not None:
            return refusal
        summary = json.loads((newest / SUMMARY_FILE).read_text(encoding="utf-8"))
        print(f"final adapter already saved at {newest}; skipping training")
        code, revision, final_dir = finish_and_upload(ckpt_root, None, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}

    if newest is not None:
        restore_monitor(monitor, newest)
    # The reward must group rewards with the same N the trainer generates per prompt.
    reward_fn = announce_first_call(make_reward_func(tasks, monitor, generations))
    trainer, model_revision = (make_trainer or build_grpo_trainer)(
        args, rows, reward_fn, monitor, ckpt_root, trainer_dir, hparams)

    def dead(reason: str) -> dict[str, Any]:
        print(DEAD_MESSAGE)
        print(f"{reason} Checkpoints so far are kept in {ckpt_root}. Run: python "
              "scripts/train_rft.py --hub-repo <user>/alien-rft --seed "
              f"{args.seed} (or run this script again with --auto-fallback).")
        return {"code": EXIT_DEAD_REWARD, "revision": None, "summary": None, "final_dir": None}

    try:
        try:
            if newest is not None:
                # VERIFY ON COLAB: resuming restores the LoRA weights, optimizer and step count.
                print(f"resuming from {newest}")
            trainer.train(resume_from_checkpoint=str(newest) if newest is not None else None)
        except DeadRewardError as exc:
            return dead(str(exc))
        summary = {"steps": int(trainer.state.global_step), "model_revision": model_revision,
                   "reward_signal": monitor.summary()}
        print(f"reward groups: {summary['reward_signal']}, unmatched completions: "
              f"{monitor.unmatched}, steps: {summary['steps']}")
        if args.smoke:
            return {"code": EXIT_OK, "revision": None, "summary": summary, "final_dir": None}
        if monitor.groups_with_variance == 0:
            return dead("Training ended without a single reward group that varied.")

        def write_adapter(directory: Path) -> None:
            trainer.model.save_pretrained(str(directory))
            (directory / SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
            write_run_config(directory, config, model_revision)

        code, revision, final_dir = finish_and_upload(ckpt_root, write_adapter, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}
    finally:
        trainer = None  # free the GPU before the evaluation loads its own copy
        free_gpu()


def evaluate_tuned(args: argparse.Namespace, splits: Splits, run: Path, final_dir: Path,
                   model_revision: str) -> dict[str, Any]:
    """Held-out metrics of the tuned adapter on heldout[:200], the slice run_eval uses."""
    from run_eval import make_hf_generate  # heavy imports happen inside it

    tasks: list[Task] = splits.heldout[:EVAL_N]
    tuned = make_hf_generate(MODEL_ID, model_revision, str(final_dir), DECODING)
    run_key = make_run_key(model_id=MODEL_ID, revision=args.revision,
                           resolved_revision=model_revision, adapter=str(final_dir),
                           hardness=args.hardness)
    return evaluate_variants(tasks, {"tuned": tuned}, run / "progress_tuned", seed=args.seed,
                             batch_size=args.eval_batch_size, run_key=run_key)


def finalize(args: argparse.Namespace, *, path: str, splits: Splits, run: Path,
             outcome: dict[str, Any], prior: dict[str, Any],
             generations_per_prompt: int) -> int:
    """Evaluate the tuned adapter, print the four models, write and validate the results."""
    summary = outcome["summary"]
    try:
        tuned = evaluate_tuned(args, splits, run, outcome["final_dir"], summary["model_revision"])
    except NoEligibleExamplesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except StaleProgressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    models = collect_models(prior, tuned["models"]["tuned"])
    print_summary(models)
    doc = build_grpo_results(
        path=path, evaluated={"models": models}, hub_repo=args.hub_repo,
        revision=outcome["revision"], generations_per_prompt=generations_per_prompt,
        reward_summary=summary["reward_signal"], seed=args.seed)
    try:
        write_results(args.out, doc, check_results.validate)
    except ResultsInvalidError as exc:
        print(f"error: {args.out} was not written, it failed validation:\n{exc}\n"
              f"The adapter is safe at {outcome['final_dir']} and on the Hub.", file=sys.stderr)
        return EXIT_FAILED
    print(f"wrote {args.out}")
    return EXIT_OK


# ---- command line ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser, default_out: str) -> None:
    """Flags shared with scripts/train_rft.py."""
    parser.add_argument("--hub-repo", help="HF Hub repo for the adapter, e.g. user/alien-grpo")
    parser.add_argument("--seed", type=int, default=0, help="splits, training and bootstrap seed; "
                        "must equal the seed of eval_baselines.json and sft.json")
    parser.add_argument("--work-dir", help="persistent storage (default Drive when mounted, "
                                           "else ./work)")
    parser.add_argument("--sft-adapter", help="SFT adapter to continue from: a local path or a "
                                              "hub id (default <work-dir>/ckpt/final)")
    parser.add_argument("--splits", help="JSON file from splits_to_json (default: make_splits)")
    add_hardness_argument(parser)
    parser.add_argument("--revision", default="main", help="base model hub revision")
    parser.add_argument("--baselines", default="results/eval_baselines.json")
    parser.add_argument("--sft-results", default="results/sft.json")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="a few steps on 16 rows, no upload, no results file "
                             "(needs --sft-adapter)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without importing any heavy library")


def check_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be >= 1")
    if not (args.smoke or args.dry_run) and not args.hub_repo:
        parser.error("--hub-repo is required (not with --smoke or --dry-run)")
    if args.smoke and not args.dry_run and not args.sft_adapter:
        parser.error("--smoke needs --sft-adapter (a local path or hub id)")


def load_splits(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Splits:
    return load_splits_checked(parser, args.splits, args.seed, args.hardness)


def print_common_plan(args: argparse.Namespace, work_root: Path, run: Path) -> None:
    state, newest = resume_state(run / "ckpt")
    print(f"model: {MODEL_ID} (revision {args.revision}), fp16 LoRA")
    print(f"seed: {args.seed}")
    print(f"sft adapter: {args.sft_adapter or work_root / 'ckpt' / 'final'}")
    print(f"held-out examples to evaluate: {EVAL_N}")
    print(f"work dir: {run}")
    print(f"resume: {state}" + (f" ({newest})" if newest is not None else ""))
    if args.smoke:
        print("hub upload: none (smoke)")
        print("results file: none (smoke)")
    else:
        print(f"hub upload: {args.hub_repo or '(--hub-repo missing)'}, "
              f"{UPLOAD_ATTEMPTS} attempts")
        print(f"results file: {args.out}")
        print(f"baselines: {args.baselines} "
              f"({'found' if Path(args.baselines).exists() else 'missing: required'})")
        print(f"sft results: {args.sft_results} "
              f"({'found' if Path(args.sft_results).exists() else 'missing: required'})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable fp16 LoRA GRPO from the SFT adapter, then held-out eval (S4).")
    add_common_arguments(parser, "results/grpo.json")
    parser.add_argument("--auto-fallback", action="store_true",
                        help="when the reward is dead, run scripts/train_rft.py in this process")
    return parser


def run_fallback(args: argparse.Namespace) -> int:
    """Run the rejection-sampling fallback in this process after a dead reward (E-2)."""
    import train_rft

    argv = ["--hub-repo", args.hub_repo, "--seed", str(args.seed),
            "--hardness", str(args.hardness), "--revision", args.revision,
            "--baselines", args.baselines, "--sft-results", args.sft_results,
            "--eval-batch-size", str(args.eval_batch_size), "--sft-adapter", args.sft_adapter]
    for flag, value in (("--work-dir", args.work_dir), ("--splits", args.splits)):
        if value:
            argv += [flag, value]
    if args.smoke:
        argv.append("--smoke")
    print("--auto-fallback: running scripts/train_rft.py now")
    return train_rft.main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    check_common_arguments(parser, args)
    splits = load_splits(parser, args)
    n = SMOKE_ROWS if args.smoke else None
    tasks = train_tasks(splits, n)
    rows = build_grpo_rows(splits, n)
    hparams = plan_grpo_hparams(args.seed, args.smoke)
    work_root = Path(args.work_dir) if args.work_dir else default_work_dir()
    run = run_dir(work_root, "grpo", args.smoke)

    if args.dry_run:
        print_common_plan(args, work_root, run)
        print(f"train rows: {len(rows)} (train split only)")
        print(f"generations per prompt: {hparams['num_generations']}")
        print(f"auto fallback to rejection sampling: {'on' if args.auto_fallback else 'off'}")
        print("hyper-parameters: " + json.dumps(hparams, indent=1))
        return EXIT_OK

    try:
        prior = None if args.smoke else load_prior_results(
            args.baselines, args.sft_results, args.seed)
        args.sft_adapter = resolve_sft_adapter(args, work_root)
    except InputsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    print_common_plan(args, work_root, run)
    print(f"train rows: {len(rows)} (train split only)")
    print("hyper-parameters: " + json.dumps(hparams, indent=1))
    outcome = train(args, rows, tasks, run / "ckpt", run / "trainer", hparams)
    if outcome["code"] == EXIT_DEAD_REWARD and args.auto_fallback:
        return run_fallback(args)
    if outcome["code"] != EXIT_OK or args.smoke:
        return outcome["code"]
    return finalize(args, path="grpo", splits=splits, run=run, outcome=outcome, prior=prior,
                    generations_per_prompt=hparams["num_generations"])


if __name__ == "__main__":
    sys.exit(main())
