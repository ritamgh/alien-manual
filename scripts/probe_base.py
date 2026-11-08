"""Day-1 headroom check: run the BASE model on the 100 development tasks (PRD S1, E-1, R-1).

Colab usage (needs a GPU and HF_TOKEN in the environment, read by the hub library only):

    python scripts/probe_base.py --seed 0
    python scripts/check_results.py results/base_probe.json

The probe uses the dev split only and the plain base prompt (no "follow the docs" line). It
writes results/base_probe.json (kind ``base_probe``) with the prior-trap rate, a ``headroom``
verdict and one quoted trap example. If ``headroom`` is "low" (prior-trap rate under 10%) the
base model already follows the docs: rerun with ``--hardness 1`` (E-1) BEFORE the held-out
split is frozen. A low headroom is a finding, so the exit code is still 0.

The run is resumable: every batch is appended to ``--progress-dir`` before the next starts.
The progress file name carries the seed and hardness, so a ``--hardness 1`` rerun never
reuses outputs of the ``--hardness 0`` run. The file also carries a fingerprint of the run
(prompts, model id, revision), so changing --model-id or --revision over an existing progress
file exits 2 naming the file: use a fresh --progress-dir or delete it. ``--fake oracle|habit``
is a test-only switch that skips model loading. Exits 0 on success, 1 with a plain message
when the file fails validation or a metric has no eligible examples, 2 on bad arguments or a
stale progress file. torch is imported only when a model is loaded, so ``--help`` works
without it.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_results  # noqa: E402
import run_eval  # noqa: E402
from alien.eval import (  # noqa: E402
    DECODING,
    MODEL_ID,
    GenerateFn,
    ResultsInvalidError,
    open_progress,
    run_variant,
    write_results,
)
from alien.generator import Task  # noqa: E402
from alien.oracle import call_to_json  # noqa: E402
from alien.persist import StaleProgressError  # noqa: E402
from alien.scoring import NoEligibleExamplesError, evaluate, score_example  # noqa: E402
from alien.splits import make_splits  # noqa: E402

HEADROOM_THRESHOLD = 0.10
MAX_HARDNESS = 3
EMPTY_OUTPUT = "<empty output>"
LOW_HEADROOM_MESSAGE = (
    "Base model already follows the docs: rerun with --hardness 1 (harder twists) BEFORE "
    "freezing the held-out split; do not move the core to Gremlin Gym without Ritam's yes (B-6)"
)
MAX_HARDNESS_MESSAGE = (
    f"Hardness {MAX_HARDNESS} is the highest level and headroom is still low: ask Ritam "
    "how to proceed (B-6); do not freeze the held-out split yet."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def suggest_next_hardness(prior_trap_rate: float, hardness: int) -> int | None:
    """The hardness to rerun with (E-1), or None when headroom is fine or the cap is reached."""
    if prior_trap_rate >= HEADROOM_THRESHOLD or hardness >= MAX_HARDNESS:
        return None
    return hardness + 1


def pick_trap_example(tasks: Sequence[Task], outputs: Sequence[str]) -> dict[str, Any]:
    """The one quoted trap example: first prior-trap, else first wrong, else first trap task."""
    traps = [(task, output, score_example(task, output))
             for task, output in zip(tasks, outputs) if task.is_trap]
    if not traps:
        raise NoEligibleExamplesError("prior_trap_rate: no trap examples")
    chosen = (next((t for t in traps if t[2].prior_trap), None)
              or next((t for t in traps if t[2].tp == 0), None)
              or traps[0])
    task, output, _ = chosen
    return {
        "task_id": task.id,
        "request": task.request,
        "model_output": output if output.strip() else EMPTY_OUTPUT,
        "habit_call": task.habit_call,
        "oracle_call": task.oracle_call,
    }


def build_probe_results(
    tasks: Sequence[Task],
    outputs: Sequence[str],
    metrics: Mapping[str, Any],
    *,
    seed: int,
    model: str,
    model_revision: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The ``base_probe`` document; ``git_sha`` comes from the GIT_SHA environment variable."""
    rate = metrics["prior_trap_rate"]["value"]
    return {
        "kind": "base_probe",
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "n_examples": len(tasks),
        "split": "dev",
        "model": model,
        "model_revision": model_revision,
        "metrics": dict(metrics),
        "prior_trap_rate": rate,
        "headroom": "ok" if rate >= HEADROOM_THRESHOLD else "low",
        "trap_example": pick_trap_example(tasks, outputs),
    }


def run_probe(
    tasks: Sequence[Task],
    generate: GenerateFn,
    progress_file: str | Path,
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    batch_size: int = 8,
    run_key: str = "",
) -> tuple[list[str], dict[str, Any]]:
    """Generate (resuming from ``progress_file``) and score; return (outputs, Metrics).

    ``run_key`` names the model and revision; a progress file of another run raises
    ``StaleProgressError``.
    """
    store = open_progress(progress_file, "base", tasks, run_key)
    run = run_variant("base", tasks, generate, store, batch_size)
    outputs = run["outputs"]
    return outputs, evaluate(tasks, outputs, n_resamples=n_resamples, seed=seed)


def format_verdict(doc: Mapping[str, Any], tasks: Sequence[Task], hardness: int) -> str:
    """Plain-language verdict: the rate with its CI, the quoted example, the E-1 instruction."""
    metric = doc["metrics"]["prior_trap_rate"]
    example = doc["trap_example"]
    task = next(t for t in tasks if t.id == example["task_id"])
    score = score_example(task, example["model_output"])
    if score.prior_trap:
        did = "followed its habit instead of the doc (a prior-trap)"
    elif score.tp == 1:
        did = "followed the doc"
    else:
        did = "gave a wrong call that is not the habit call either"
    lines = [
        f"Prior-trap rate on {metric['n']} dev trap tasks: {metric['value']:.1%} "
        f"(95% CI {metric['ci_low']:.1%} to {metric['ci_high']:.1%}); headroom: {doc['headroom']}",
        f"Example trap task {example['task_id']}:",
        f"  request: {example['request']}",
        f"  the model said: {example['model_output']}",
        f"  that {did}",
        f"  the doc says: {' '.join(task.tool.doc.split())}",
        f"  doc-following call: {call_to_json(example['oracle_call'])}",
        f"  habit call: {call_to_json(example['habit_call'])}",
    ]
    if doc["headroom"] == "low":
        lines.append(LOW_HEADROOM_MESSAGE)
        if suggest_next_hardness(doc["prior_trap_rate"], hardness) is None:
            lines.append(MAX_HARDNESS_MESSAGE)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the base model on the 100 dev tasks and write results/base_probe.json."
    )
    parser.add_argument("--seed", type=int, default=0, help="splits and bootstrap seed (0)")
    parser.add_argument("--hardness", type=int, default=0,
                        help="make_splits hardness (E-1 hook); rerun with 1 if headroom is low")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="hub revision (default main)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-dir", default="results/progress_probe")
    parser.add_argument("--out", default="results/base_probe.json")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--fake", choices=("oracle", "habit"),
                        help="TEST ONLY: skip model loading and answer from the tasks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.hardness < 0:
        parser.error("--hardness must be >= 0")

    tasks = list(make_splits(args.seed, args.hardness).dev)
    label = f"fake-{args.fake}" if args.fake else "base"
    model = "fake" if args.fake else args.model_id
    print(f"model: {model}")
    print(f"revision: {'fake' if args.fake else args.revision}")
    print(f"examples: {len(tasks)} (dev), hardness {args.hardness}, plain base prompt")
    print(f"generation budget: {len(tasks)} generations of up to "
          f"{DECODING['max_new_tokens']} new tokens, greedy", flush=True)

    if args.fake:
        generate = run_eval.make_fake_generate(tasks, args.fake)
    else:
        generate = run_eval.make_hf_generate(args.model_id, args.revision, None, DECODING)
    revision = "fake" if args.fake else getattr(generate, "resolved_revision", args.revision)

    progress_file = Path(args.progress_dir) / f"{label}_seed{args.seed}_h{args.hardness}.jsonl"
    run_key = f"{args.model_id}|{args.revision}|{revision}"
    try:
        outputs, metrics = run_probe(
            tasks, generate, progress_file, seed=args.seed, n_resamples=args.n_resamples,
            batch_size=args.batch_size, run_key=run_key,
        )
        doc = build_probe_results(tasks, outputs, metrics, seed=args.seed, model=model,
                                  model_revision=revision)
        write_results(args.out, doc, check_results.validate)
    except NoEligibleExamplesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ResultsInvalidError as exc:
        print(f"error: {args.out} was not written, it failed validation:\n{exc}", file=sys.stderr)
        return 1
    except StaleProgressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    print(format_verdict(doc, tasks, args.hardness))
    return 0


if __name__ == "__main__":
    sys.exit(main())
