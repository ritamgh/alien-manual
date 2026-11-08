"""Evaluate base, prompt-only and tuned models on the held-out split (PRD FR-002, FR-006).

Colab usage (needs a GPU and HF_TOKEN in the environment, read by the hub library only):

    python scripts/run_eval.py --which base,prompt_only --out results/eval_baselines.json
    python scripts/run_eval.py --which tuned --adapter <hub id or path>

The run is resumable: every batch is appended to --progress-dir before the next starts, so
re-running the same command after an interrupted session continues where it stopped, and the
measured GPU time of all sessions is summed into gpu_seconds. ``--fake oracle|habit`` is a
test-only switch that skips model loading and answers from the tasks themselves.

Every progress file carries a fingerprint of its run (prompts, model id, revision, adapter,
hardness, decoding), so --progress-dir must be fresh when any of those change (another --seed,
--hardness, --n, --adapter, ...): the run then exits 2 naming the file, instead of replaying
old outputs.

A ``--splits`` file must also pass the leak check of ``alien.splits.assert_no_leak``: otherwise
one plain line and exit 2.

Exits 0 on success; 1 with a plain message when the results file fails validation or a metric
has no eligible examples; 2 on bad arguments or a stale progress directory. torch,
transformers and peft are imported only inside make_hf_generate, so ``--help`` works without
them.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_results  # noqa: E402
from alien.eval import (  # noqa: E402
    BASELINE_MODELS,
    BUDGET_GPU_SECONDS,
    DECODING,
    MODEL_ID,
    GenerateFn,
    ResultsInvalidError,
    build_prompts,
    build_results,
    build_tuned_results,
    evaluate_variants,
    write_results,
)
from alien.generator import Task  # noqa: E402
from alien.oracle import call_to_json  # noqa: E402
from alien.persist import StaleProgressError  # noqa: E402
from alien.scoring import METRIC_NAMES, NoEligibleExamplesError  # noqa: E402
from alien.splits import (  # noqa: E402
    LeakError,
    Splits,
    assert_no_leak,
    make_splits,
    splits_from_json,
)

DEFAULT_N = {"heldout": 200, "dev": 100}
MAX_HARDNESS = 3  # the highest E-1 level, as in probe_base.py


def make_hf_generate(
    model_id: str, revision: str, adapter: str | None, decoding: dict[str, Any]
) -> GenerateFn:
    """Load the model in fp16 on cuda and return a greedy, batched ``GenerateFn``.

    The returned function carries ``resolved_revision``: the commit the hub served, when known.
    The token is never read here: the hub library picks up HF_TOKEN from the environment.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16
    ).to("cuda")
    resolved = getattr(model.config, "_commit_hash", None) or revision
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(prompts: Sequence[str]) -> list[str]:
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
        batch = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = batch.to("cuda")
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=decoding["max_new_tokens"],
                do_sample=decoding["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    generate.resolved_revision = resolved  # type: ignore[attr-defined]
    return generate


def make_fake_generate(tasks: Sequence[Task], kind: str) -> GenerateFn:
    """Test-only: answer with the oracle call (kind "oracle") or the habit call ("habit")."""
    answers = {}
    for task in tasks:
        call = task.oracle_call if kind == "oracle" else task.habit_call
        for variant in ("base", "prompt_only"):
            answers[build_prompts([task], variant)[0]] = call_to_json(call)

    def generate(prompts: Sequence[str]) -> list[str]:
        return [answers[prompt] for prompt in prompts]

    return generate


def add_hardness_argument(parser: argparse.ArgumentParser) -> None:
    """The ``--hardness`` flag shared with the train scripts (E-1)."""
    parser.add_argument("--hardness", type=int, default=0,
                        help="make_splits hardness (E-1 hook); use the value of the probe run "
                             f"that was re-run for headroom, 0 to {MAX_HARDNESS} (default 0)")


def make_run_key(*, model_id: str, revision: str, resolved_revision: str, adapter: str | None,
                 hardness: int, fake: str | None = None) -> str:
    """The ``run_key`` of an evaluation: what, besides the prompts, shapes the outputs.

    Every script that calls ``evaluate_variants`` passes this, so a progress directory reused
    for another base revision, adapter or checkpoint, hardness or decoding raises
    ``StaleProgressError`` instead of replaying the other run's outputs.
    """
    return json.dumps({"model_id": model_id, "revision": revision,
                       "resolved_revision": resolved_revision, "adapter": adapter,
                       "hardness": hardness, "decoding": DECODING, "fake": fake}, sort_keys=True)


def load_splits_checked(parser: argparse.ArgumentParser, splits_file: str | None,
                        seed: int | None, hardness: int) -> Splits:
    """The splits for a run: ``make_splits(seed, hardness)``, or the ``--splits`` file.

    A hardness outside 0..MAX_HARDNESS is a parser error. A ``--splits`` file records its seed
    and hardness, and both must equal this run's (a ``None`` seed is not checked): otherwise
    one plain line naming both pairs and exit 2. A ``--splits`` file must also pass
    ``assert_no_leak``; if not, one plain line naming the failed check and exit 2.
    """
    if not 0 <= hardness <= MAX_HARDNESS:
        parser.error(f"--hardness must be between 0 and {MAX_HARDNESS}, got {hardness}")
    if not splits_file:
        return make_splits(seed or 0, hardness)
    splits = splits_from_json(Path(splits_file).read_text(encoding="utf-8"))
    if (seed is not None and splits.seed != seed) or splits.hardness != hardness:
        parser.exit(2, f"error: {splits_file} was made with seed {splits.seed} and hardness "
                       f"{splits.hardness} but this run uses seed "
                       f"{splits.seed if seed is None else seed} and hardness {hardness}; "
                       "pass matching --seed and --hardness or rebuild the splits file\n")
    try:
        assert_no_leak(splits)
    except LeakError as exc:
        parser.exit(2, f"error: {splits_file} fails the leak check: {exc}; "
                       "rebuild the splits file\n")
    return splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable three-way eval: base, prompt_only and tuned on the held-out split."
    )
    parser.add_argument("--splits", help="JSON file from splits_to_json (default: make_splits)")
    parser.add_argument("--seed", type=int, help="splits and bootstrap seed (default 0)")
    add_hardness_argument(parser)
    parser.add_argument("--split", choices=("heldout", "dev"), default="heldout")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="hub revision (default main)")
    parser.add_argument("--adapter", help="LoRA path or hub id; required with --which tuned")
    parser.add_argument("--which", default=",".join(BASELINE_MODELS),
                        help="comma list from base|prompt_only|tuned (default base,prompt_only)")
    parser.add_argument("--n", type=int, help="use the first n tasks (default 200 heldout)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-dir", default="results/progress")
    parser.add_argument("--out", help="default results/eval_baselines.json, or "
                                      "results/eval_tuned.json for --which tuned")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--fake", choices=("oracle", "habit"),
                        help="TEST ONLY: skip model loading and answer from the tasks")
    return parser


def _no_checks(_doc: Any) -> list[str]:
    """A tuned run has no results kind yet, so there is no schema to check it against."""
    return []


def _summary(name: str, metrics: dict[str, Any]) -> str:
    parts = [f"{key} {metrics[key]['value']:.3f} "
             f"[{metrics[key]['ci_low']:.3f}, {metrics[key]['ci_high']:.3f}]"
             for key in METRIC_NAMES]
    return f"{name}: " + "; ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    variants = [v.strip() for v in args.which.split(",") if v.strip()]
    if sorted(variants) == sorted(BASELINE_MODELS):
        variants = list(BASELINE_MODELS)
        if args.adapter:
            parser.error("--adapter only applies to --which tuned")
    elif variants == ["tuned"]:
        if not args.adapter:
            parser.error("--which tuned needs --adapter")
    else:
        parser.error("--which takes base,prompt_only (the eval_baselines file) or tuned alone "
                     f"(run separately with its adapter), got {args.which!r}")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    splits = load_splits_checked(parser, args.splits, args.seed, args.hardness)
    seed = splits.seed if args.seed is None else args.seed
    pool = splits.heldout if args.split == "heldout" else splits.dev
    n = DEFAULT_N[args.split] if args.n is None else args.n
    if not 1 <= n <= len(pool):
        parser.error(f"--n must be between 1 and {len(pool)} for split {args.split}, got {n}")
    tasks = pool[:n]
    tuned = variants == ["tuned"]
    out = args.out or ("results/eval_tuned.json" if tuned else "results/eval_baselines.json")

    model_label = "fake" if args.fake else args.model_id
    print(f"model: {model_label}")
    print(f"revision: {'fake' if args.fake else args.revision}")
    print(f"examples: {n} ({args.split})")
    print(f"variants: {', '.join(variants)}")
    print(f"generation budget: {n * len(variants)} generations of up to "
          f"{DECODING['max_new_tokens']} new tokens, greedy; limit {BUDGET_GPU_SECONDS} s "
          f"for 3 models x 200 examples (FR-002)", flush=True)

    if args.fake:
        generate = make_fake_generate(tasks, args.fake)
    else:
        generate = make_hf_generate(args.model_id, args.revision, args.adapter, DECODING)
    revision = "fake" if args.fake else getattr(generate, "resolved_revision", args.revision)

    run_key = make_run_key(model_id=args.model_id, revision=args.revision,
                           resolved_revision=revision, adapter=args.adapter,
                           hardness=args.hardness, fake=args.fake)
    try:
        evaluated = evaluate_variants(
            tasks, {name: generate for name in variants}, args.progress_dir, seed=seed,
            n_resamples=args.n_resamples, batch_size=args.batch_size, run_key=run_key,
        )
    except NoEligibleExamplesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except StaleProgressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, metrics in evaluated["models"].items():
        print(_summary(name, metrics))
    print(f"gpu_seconds: {evaluated['gpu_seconds']:.1f} (all sessions)")

    if tuned:
        doc = build_tuned_results(evaluated, seed=seed, model_revision=revision,
                                  adapter=args.adapter, split=args.split)
        write_results(out, doc, _no_checks)
    else:
        doc = build_results(evaluated, seed=seed, model_revision=revision, split=args.split)
        try:
            write_results(out, doc, check_results.validate)
        except ResultsInvalidError as exc:
            print(f"error: {out} was not written, it failed validation:\n{exc}", file=sys.stderr)
            return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
