"""Rejection-sampling fine-tuning, the fallback when GRPO cannot run (PRD DEC-1, R-3, E-2, FR-005).

Slice S4. Samples several completions per TRAIN prompt from the SFT model, keeps only those whose
exact-match reward is 1.0 (at most a few per task, no duplicates), and trains the SFT adapter
for one more low-learning-rate SFT epoch on them. Then it uploads the adapter, evaluates it on the
held-out split next to base, prompt-only and sft, and writes results/rft.json (contract kind
``rft``, ``path`` "rft"), using the shared builder in scripts/train_grpo.py.

Colab usage (run it after scripts/train_grpo.py exits 3, or when GRPO cannot run at all):

    python scripts/train_rft.py --hub-repo <user>/alien-rft --seed 0
    python scripts/check_results.py results/rft.json

Same inputs and flags as scripts/train_grpo.py (results/sft.json and results/eval_baselines.json
with the same ``--seed``, the SFT adapter, ``--work-dir``, ``--smoke``, ``--dry-run``). Sampling
is resumable: every sampled completion is saved at once in ``<work-dir>/rft/samples.jsonl``, and
training resumes from the newest manifest-listed checkpoint. A task with no correct sample is
skipped; the kept and skipped counts are printed. If no task has one, the script exits 4 and the
SFT adapter stays the final model (``tuned_path`` "sft" in the report).

Exit codes: 0 ok, 1 upload failed or the results file failed validation, 2 bad arguments or
inputs, 4 no sampled completion was correct.

torch, transformers, peft, trl, datasets and huggingface_hub are imported only inside functions.
Anything not checkable without a GPU is marked VERIFY ON COLAB.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alien.eval import DECODING, MODEL_ID, GenerateFn  # noqa: E402
from alien.generator import Task  # noqa: E402
from alien.oracle import render_prompt  # noqa: E402
from alien.persist import ProgressStore  # noqa: E402
from alien.reward import RewardMonitor, exact_match_reward  # noqa: E402
from train_grpo import (  # noqa: E402
    EXIT_BAD_INPUT,
    EXIT_OK,
    SMOKE_ROWS,
    InputsError,
    add_common_arguments,
    build_callbacks,
    check_common_arguments,
    finalize,
    free_gpu,
    hub_upload,
    load_prior_results,
    load_splits,
    print_common_plan,
    resolve_sft_adapter,
    run_dir,
    train_tasks,
)
from train_sft import (  # noqa: E402
    SFT_HPARAMS,
    SUMMARY_FILE,
    build_run_config,
    default_work_dir,
    finish_and_upload,
    refuse_reused_final,
    resume_state,
    to_prompt_completion,
    write_run_config,
)

EXIT_NO_ROWS = 4
RFT_HPARAMS: dict[str, Any] = {
    **{key: value for key, value in SFT_HPARAMS.items() if not key.startswith("lora_")},
    "learning_rate": 1e-4,  # lower than the SFT 2e-4: the adapter is already trained
    "num_train_epochs": 1,
    "samples_per_task": 8,  # k completions sampled per train prompt (the reward groups)
    "max_per_task": 2,  # at most m correct completions kept per task
    "temperature": 1.0,
    "top_p": 0.95,
    "sample_batch_size": 16,
}
SMOKE_OVERRIDES: dict[str, Any] = {
    "max_steps": 4, "max_length": 256, "save_steps": 2, "samples_per_task": 4,
}


# ---- pure helpers ---------------------------------------------------------------------------


def plan_rft_hparams(seed: int, smoke: bool) -> dict[str, Any]:
    hparams = dict(RFT_HPARAMS, seed=seed)
    if smoke:
        hparams.update(SMOKE_OVERRIDES)
    return hparams


def sample_candidates(
    tasks: Sequence[Task], generate: GenerateFn, k: int, *,
    store: ProgressStore | None = None, batch_size: int = 16,
) -> list[list[str]]:
    """``k`` sampled completions per task, in task order.

    ``generate`` takes a batch of prompts and returns one sampled completion each. With a
    ``store`` every completion is recorded at once and a re-run only generates what is missing.
    """
    done = {} if store is None else {key: v["output"] for key, v in store.results().items()}
    todo = [(f"{task.id}#{j}", render_prompt(task))
            for task in tasks for j in range(k) if f"{task.id}#{j}" not in done]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        texts = generate([prompt for _, prompt in batch])
        if len(texts) != len(batch):
            raise ValueError(f"generator returned {len(texts)} completions for "
                             f"{len(batch)} prompts")
        for (key, _), text in zip(batch, texts):
            done[key] = text
            if store is not None:
                store.record(key, {"output": text})
    return [[done[f"{task.id}#{j}"] for j in range(k)] for task in tasks]


def select_by_reward(
    tasks: Sequence[Task], completions: Sequence[Sequence[str]], threshold: float = 1.0,
    max_per_task: int = 2,
) -> list[tuple[Task, str]]:
    """The ``(task, completion)`` pairs worth training on.

    A completion is kept when its exact-match reward is at least ``threshold``. Per task the
    first ``max_per_task`` distinct kept texts (compared after stripping) survive, in sampled
    order; tasks stay in input order, so the result is deterministic.
    """
    if len(tasks) != len(completions):
        raise ValueError(f"{len(tasks)} tasks but {len(completions)} lists of completions")
    selected: list[tuple[Task, str]] = []
    for task, texts in zip(tasks, completions):
        seen: set[str] = set()
        for text in texts:
            if len(seen) >= max_per_task:
                break
            cleaned = text.strip()
            if cleaned in seen or exact_match_reward(task, cleaned) < threshold:
                continue
            seen.add(cleaned)
            selected.append((task, cleaned))
    return selected


def build_rft_rows(selected: Sequence[tuple[Task, str]]) -> list[dict]:
    """SFT chat rows ``{"messages": [user, assistant]}`` with the prompt the held-out eval uses."""
    return [{"messages": [{"role": "user", "content": render_prompt(task)},
                          {"role": "assistant", "content": text}]}
            for task, text in selected]


def reward_signal(tasks: Sequence[Task], completions: Sequence[Sequence[str]],
                  k: int) -> dict[str, int]:
    """Reward groups over the sampled completions: ``{"groups_total", "groups_with_variance"}``.

    One group is the ``k`` samples of one task; it has variance when some are right and some not.
    """
    monitor = RewardMonitor(num_generations=k)
    for task, texts in zip(tasks, completions):
        monitor.observe([exact_match_reward(task, text) for text in texts])
    return monitor.summary()


# ---- GPU path (Colab only) ------------------------------------------------------------------


def make_hf_sampler(model_id: str, revision: str, adapter: str,
                    hparams: dict[str, Any]) -> GenerateFn:
    """A sampling, batched generate function over base + the SFT adapter. The model is loaded on
    the first call, so a resumed run with nothing left to sample never touches the GPU."""
    loaded: dict[str, Any] = {}

    def load() -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                  padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # VERIFY ON COLAB: transformers 5 renamed torch_dtype to dtype (as in the SFT script).
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.float16).to("cuda")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        loaded.update(tokenizer=tokenizer, model=model)

    def generate(prompts: Sequence[str]) -> list[str]:
        import torch

        if not loaded:
            load()
        tokenizer, model = loaded["tokenizer"], loaded["model"]
        chats = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
                 for prompt in prompts]
        batch = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = batch.to("cuda")
        with torch.no_grad():
            out = model.generate(
                **batch, max_new_tokens=DECODING["max_new_tokens"], do_sample=True,
                temperature=hparams["temperature"], top_p=hparams["top_p"],
                pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate


def build_rft_trainer(
    args: argparse.Namespace, rows: list[dict], ckpt_root: Path, trainer_dir: Path,
    hparams: dict[str, Any],
) -> tuple[Any, str]:
    """Load base + the SFT adapter (trainable) and build the SFTTrainer. Returns the trainer and
    the base model revision."""
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    # VERIFY ON COLAB: transformers 5 renamed torch_dtype to dtype (as in the SFT script).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, dtype=torch.float16, device_map={"": 0})
    model_revision = getattr(model.config, "_commit_hash", None) or args.revision
    # VERIFY ON COLAB: is_trainable makes the SFT LoRA weights trainable and SFTTrainer accepts
    # an already wrapped PeftModel (no peft_config).
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    # VERIFY ON COLAB: a prompt/completion dataset makes recent TRL compute the loss on the
    # completion only (same as scripts/train_sft.py).
    dataset = Dataset.from_list([to_prompt_completion(row) for row in rows])
    kwargs: dict[str, Any] = {
        "output_dir": str(trainer_dir),
        "num_train_epochs": hparams["num_train_epochs"],
        "per_device_train_batch_size": hparams["per_device_train_batch_size"],
        "gradient_accumulation_steps": hparams["gradient_accumulation_steps"],
        "learning_rate": hparams["learning_rate"],
        "warmup_ratio": hparams["warmup_ratio"],  # VERIFY ON COLAB: transformers 5 may warn
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
        kwargs["max_steps"] = hparams["max_steps"]
    trainer = SFTTrainer(
        model=model, args=SFTConfig(**kwargs), train_dataset=dataset,
        processing_class=tokenizer, callbacks=build_callbacks(ckpt_root),
    )
    # fp16 mixed precision needs fp32 trainable weights; peft usually does this already.
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
    return trainer, model_revision


def train(
    args: argparse.Namespace, rows: list[dict], extra_summary: dict[str, Any], ckpt_root: Path,
    trainer_dir: Path, hparams: dict[str, Any], *,
    make_trainer: Callable[..., tuple[Any, str]] | None = None,
    upload: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    """Train (or resume), then finish_and_upload. Returns code, revision, summary, final_dir.

    ``extra_summary`` (the reward signal) is stored with the final adapter. A re-run whose
    ``final`` checkpoint is complete skips training and only retries the upload, but only if
    that checkpoint's ``run_config.json`` matches this run; else exit code 2.
    """
    upload = upload or (lambda folder: hub_upload(folder, args.hub_repo, "RFT LoRA adapter"))
    state, newest = resume_state(ckpt_root)
    config = build_run_config("train_rft", args, hparams, sft_adapter=args.sft_adapter)
    if state == "final":
        refusal = refuse_reused_final(newest, config)
        if refusal is not None:
            return refusal
        summary = json.loads((newest / SUMMARY_FILE).read_text(encoding="utf-8"))
        print(f"final adapter already saved at {newest}; skipping training")
        code, revision, final_dir = finish_and_upload(ckpt_root, None, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}

    trainer, model_revision = (make_trainer or build_rft_trainer)(
        args, rows, ckpt_root, trainer_dir, hparams)
    try:
        if newest is not None:
            # VERIFY ON COLAB: resuming restores the LoRA weights, optimizer and step count.
            print(f"resuming from {newest}")
        trainer.train(resume_from_checkpoint=str(newest) if newest is not None else None)
        summary = {"steps": int(trainer.state.global_step), "model_revision": model_revision,
                   **extra_summary}
        print(f"trained on {len(rows)} rows, steps: {summary['steps']}")
        if args.smoke:
            return {"code": EXIT_OK, "revision": None, "summary": summary, "final_dir": None}

        def write_adapter(directory: Path) -> None:
            trainer.model.save_pretrained(str(directory))
            (directory / SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
            write_run_config(directory, config, model_revision)

        code, revision, final_dir = finish_and_upload(ckpt_root, write_adapter, upload)
        return {"code": code, "revision": revision, "summary": summary, "final_dir": final_dir}
    finally:
        trainer = None  # free the GPU before the evaluation loads its own copy
        free_gpu()


# ---- command line ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rejection-sampling fine-tuning from the SFT adapter, then held-out eval.")
    add_common_arguments(parser, "results/rft.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    check_common_arguments(parser, args)
    splits = load_splits(parser, args)
    tasks = train_tasks(splits, SMOKE_ROWS if args.smoke else None)
    hparams = plan_rft_hparams(args.seed, args.smoke)
    k = hparams["samples_per_task"]
    work_root = Path(args.work_dir) if args.work_dir else default_work_dir()
    run = run_dir(work_root, "rft", args.smoke)
    ckpt_root = run / "ckpt"

    if args.dry_run:
        print_common_plan(args, work_root, run)
        print(f"train tasks to sample: {len(tasks)} (train split only)")
        print(f"samples per task: {k}, keep at most {hparams['max_per_task']} correct per task")
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
    print("hyper-parameters: " + json.dumps(hparams, indent=1))
    rows: list[dict] = []
    extra_summary: dict[str, Any] = {}
    if resume_state(ckpt_root)[0] != "final":
        generate = make_hf_sampler(MODEL_ID, args.revision, args.sft_adapter, hparams)
        candidates = sample_candidates(
            tasks, generate, k, store=ProgressStore(run / "samples.jsonl"),
            batch_size=hparams["sample_batch_size"])
        generate = None  # free the sampling model before training loads its own copy
        free_gpu()
        selected = select_by_reward(tasks, candidates, max_per_task=hparams["max_per_task"])
        rows = build_rft_rows(selected)
        kept_tasks = len({task.id for task, _ in selected})
        print(f"rejection sampling: kept {len(rows)} completions from {kept_tasks} tasks; "
              f"skipped {len(tasks) - kept_tasks} tasks with no correct sample")
        if not rows:
            print("error: no sampled completion matched the oracle call for any task, so there "
                  "is nothing to train on. The SFT adapter stays the final model (report it as "
                  "tuned_path 'sft'); no rft.json was written.", file=sys.stderr)
            return EXIT_NO_ROWS
        extra_summary = {"reward_signal": reward_signal(tasks, candidates, k)}

    outcome = train(args, rows, extra_summary, ckpt_root, run / "trainer", hparams)
    if outcome["code"] != EXIT_OK or args.smoke:
        return outcome["code"]
    return finalize(args, path="rft", splits=splits, run=run, outcome=outcome, prior=prior,
                    generations_per_prompt=k)


if __name__ == "__main__":
    sys.exit(main())
