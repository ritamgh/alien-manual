"""Build results/final.json and report.md: the honest SC-001 verdict (PRD S5, FR-006, E-3, R-4).

Combines results/eval_baselines.json (base, prompt_only), results/sft.json (sft) and, when they
exist, results/grpo.json or results/rft.json (tuned) with the per-example model outputs of the
held-out run. Metrics come from the results files; the outputs give the paired bootstrap gaps
of SC-001 (tuned minus baseline; for prior_trap_rate baseline minus tuned, so a positive gap is
always an improvement). SC-001 is "met" only when every one of the four gaps is at least 10
points (0.10) and its 95% interval lies above zero; otherwise it is "not_met" and the report
says so on its first line.

Tuned model rule (pick_tuned): the document whose recorded ``path`` ran. If grpo.json and
rft.json both exist, the one with the higher tuned tool_call_f1 wins (a tie goes to the lower
prior_trap_rate, then to grpo.json) and the report says so. If neither exists the tuned model
is the SFT model (``tuned_path`` "sft"; SFT is the guaranteed result, R-4).

Usage after the Colab runs (paths of the per-example outputs are listed in --help):

    python scripts/build_report.py \\
        --outputs sft=<work>/progress_sft/tuned.jsonl \\
        --outputs tuned=<work>/<tuned progress dir>/tuned.jsonl
    python scripts/check_results.py results/final.json

Exits 0 on success; 1 when the outputs do not reproduce the recorded metrics or final.json fails
validation (nothing is written); 2 when an input file is missing or invalid, or on bad arguments.
Stdlib only: nothing heavy is imported, so it runs on a laptop with no GPU.
"""

from __future__ import annotations

import argparse
import json
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
from alien.eval import ResultsInvalidError, write_results  # noqa: E402
from alien.generator import Task  # noqa: E402
from alien.persist import ProgressStore  # noqa: E402
from alien.scoring import (  # noqa: E402
    METRIC_NAMES,
    ExampleScore,
    NoEligibleExamplesError,
    metrics_from_scores,
    paired_gap,
    score_example,
)
from alien.splits import make_splits  # noqa: E402

MODEL_NAMES = ("base", "prompt_only", "sft", "tuned")
BASELINES = (("vs_base", "base"), ("vs_prompt_only", "prompt_only"))
GAP_METRICS = (("prior_trap_rate", False), ("tool_call_f1", True))  # (name, higher_is_better)
E3_LIMIT = 0.05
MAX_HARDNESS = 3
LABELS = {
    "prior_trap_rate": "Prior-trap rate",
    "tool_call_f1": "Tool-call F1",
    "tool_call_validity": "Tool-call validity",
    "normal_tool_f1": "Normal-tool F1",
}
COLUMNS = (("base", "Base"), ("prompt_only", "Prompt-only"), ("sft", "SFT"), ("tuned", "Tuned"))
PATH_LABEL = {"grpo": "grpo", "rft": "rft", "sft": "sft-only"}
BASELINES_HINT = ("`python scripts/run_eval.py --which base,prompt_only --seed 0 "
                  "--out results/eval_baselines.json` (Colab, baseline eval step)")
SFT_HINT = ("`python scripts/train_sft.py --hub-repo <user>/alien-sft --seed 0 "
            "--work-dir <work>` (Colab, SFT training step)")


class ReportInputError(ValueError):
    """An input does not fit together: missing example, wrong output file, bad field."""


# ---- scoring the per-example outputs --------------------------------------------------------


def load_outputs(path: str | os.PathLike[str]) -> dict[str, str]:
    """Example id to model output text, from a ProgressStore JSONL file."""
    outputs: dict[str, str] = {}
    for example_id, payload in ProgressStore(path).results().items():
        text = payload.get("output")
        if not isinstance(text, str):
            raise ReportInputError(f"{path}: record {example_id} has no string 'output' field")
        outputs[example_id] = text
    return outputs


def score_all(tasks: Sequence[Task], outputs: Mapping[str, str]) -> list[ExampleScore]:
    """Score every task, in task order. A task without an output is an error, never a zero."""
    missing = [task.id for task in tasks if task.id not in outputs]
    if missing:
        raise ReportInputError(
            f"{len(missing)} of {len(tasks)} held-out examples have no output "
            f"(first missing: {missing[0]}); is this the output file of the full run?")
    return [score_example(task, outputs[task.id]) for task in tasks]


# ---- the SC-001 verdict ---------------------------------------------------------------------


def compute_sc001(scores_by_model: Mapping[str, Sequence[ExampleScore]], *,
                  n_resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """The contract's ``sc001``: gaps of tuned against base and prompt_only, and the verdict.

    A gap is tuned minus baseline (baseline minus tuned for prior_trap_rate), as a fraction.
    Verdict "met" iff every gap is >= 0.10 and its 95% interval lower bound is above zero.
    """
    tuned = scores_by_model["tuned"]
    gaps: dict[str, dict[str, dict[str, float]]] = {}
    for key, name in BASELINES:
        gaps[key] = {
            metric: paired_gap(tuned, scores_by_model[name], metric,
                               higher_is_better=higher, n_resamples=n_resamples, seed=seed)
            for metric, higher in GAP_METRICS
        }
    verdict = "met" if all(_passes(g) for group in gaps.values() for g in group.values()) \
        else "not_met"
    return {"verdict": verdict, "gaps": gaps}


def _passes(gap: Mapping[str, float]) -> bool:
    return gap["gap"] >= check_results.SC001_GAP - check_results.EPS and gap["ci_low"] > 0


# ---- which tuned model ----------------------------------------------------------------------


def _tuned_f1_and_trap(doc: Mapping[str, Any]) -> tuple[float, float]:
    tuned = doc["models"]["tuned"]
    return tuned["tool_call_f1"]["value"], tuned["prior_trap_rate"]["value"]


def _grpo_beats_rft(grpo_doc: Mapping[str, Any], rft_doc: Mapping[str, Any]) -> tuple[bool, str]:
    """True when the grpo file's tuned model is at least as good, with the reason in words."""
    (g_f1, g_trap), (r_f1, r_trap) = _tuned_f1_and_trap(grpo_doc), _tuned_f1_and_trap(rft_doc)
    if g_f1 != r_f1:
        return g_f1 > r_f1, f"higher tuned tool_call_f1 ({max(g_f1, r_f1):.3f} vs " \
                            f"{min(g_f1, r_f1):.3f})"
    if g_trap != r_trap:
        return g_trap < r_trap, "same tool_call_f1 and a lower prior-trap rate"
    return True, "same tool_call_f1 and prior-trap rate (grpo.json is preferred on a tie)"


def pick_tuned(grpo_doc: dict[str, Any] | None, rft_doc: dict[str, Any] | None,
               sft_doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(tuned_path, source_doc)``: the document that holds the tuned model's metrics.

    Only one RL file: that file. Both: the better one (see the module docstring). None: the
    SFT document, with tuned_path "sft". For an RL file the path is the one it records as having
    run, so a grpo.json that fell back to rejection sampling is reported as "rft".
    """
    if grpo_doc is None and rft_doc is None:
        return "sft", sft_doc
    if grpo_doc is not None and rft_doc is not None:
        chosen = grpo_doc if _grpo_beats_rft(grpo_doc, rft_doc)[0] else rft_doc
    else:
        chosen = grpo_doc if grpo_doc is not None else rft_doc
    return chosen["path"], chosen


def choice_note(grpo_doc: dict[str, Any] | None, rft_doc: dict[str, Any] | None) -> str | None:
    """A sentence saying why one RL file was chosen; None unless both files exist."""
    if grpo_doc is None or rft_doc is None:
        return None
    grpo_wins, reason = _grpo_beats_rft(grpo_doc, rft_doc)
    winner, other = ("grpo", "rft") if grpo_wins else ("rft", "grpo")
    return (f"Both results/grpo.json and results/rft.json exist. The {winner} file is reported "
            f"because its tuned model has the {reason}; the {other} file is not used here.")


# ---- the final document ---------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_outputs_match(scores_by_model: Mapping[str, Sequence[ExampleScore]],
                        models: Mapping[str, Mapping[str, Any]]) -> None:
    """The outputs must reproduce each model's recorded metric values, else wrong file/seed."""
    problems = []
    for name in MODEL_NAMES:
        recomputed = metrics_from_scores(scores_by_model[name])
        for metric in METRIC_NAMES:
            recorded = models[name][metric]["value"]
            if abs(recomputed[metric] - recorded) > 1e-9:
                problems.append(
                    f"{name}: {metric} is {recomputed[metric]:.6f} when recomputed from the "
                    f"output file but {recorded:.6f} in the results file")
    if problems:
        raise ReportInputError(
            "the output files do not match the results files (wrong file, or a different "
            "seed or example count):\n" + "\n".join(problems))


def build_final(baselines: Mapping[str, Any], sft_doc: Mapping[str, Any],
                grpo_doc: dict[str, Any] | None, rft_doc: dict[str, Any] | None,
                scores_by_model: Mapping[str, Sequence[ExampleScore]], *, seed: int,
                report_path: str, n_resamples: int = 1000, created_at: str | None = None,
                git_sha: str | None = None) -> dict[str, Any]:
    """The contract ``final`` document. Validate it (check_results) before writing it."""
    tuned_path, source = pick_tuned(grpo_doc, rft_doc, dict(sft_doc))
    models = {
        "base": baselines["models"]["base"],
        "prompt_only": baselines["models"]["prompt_only"],
        "sft": sft_doc["models"]["sft"],
        "tuned": source["models"]["sft" if tuned_path == "sft" else "tuned"],
    }
    check_outputs_match(scores_by_model, models)
    return {
        "kind": "final",
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": git_sha or os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "models": models,
        "tuned_path": tuned_path,
        "sc001": compute_sc001(scores_by_model, n_resamples=n_resamples, seed=seed),
        "report_path": report_path,
    }


# ---- the report -----------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _points(value: float) -> str:
    return f"{round(value * 100, 1) + 0.0:.1f}"  # + 0.0 turns -0.0 into 0.0


def _cell(metric: Mapping[str, Any]) -> str:
    return f"{_pct(metric['value'])} [{_pct(metric['ci_low'])}, {_pct(metric['ci_high'])}]"


def _gap_rows(sc001: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, float]]]:
    return [(label, metric, sc001["gaps"][key][metric])
            for key, label in (("vs_base", "vs base"), ("vs_prompt_only", "vs prompt-only"))
            for metric, _ in GAP_METRICS]


def _headline(final_doc: Mapping[str, Any]) -> str:
    sc001 = final_doc["sc001"]
    rows = _gap_rows(sc001)
    path = PATH_LABEL[final_doc["tuned_path"]]
    if sc001["verdict"] == "met":
        label, metric, gap = min(rows, key=lambda row: row[2]["gap"])
        return (f"SC-001 met: the tuned model ({path}) beats base and prompt-only by at least "
                f"10 points on both prior-trap rate and tool-call F1; the smallest gap is "
                f"{_points(gap['gap'])} points ({LABELS[metric]} {label}, 95% CI "
                f"{_points(gap['ci_low'])} to {_points(gap['ci_high'])}).")
    failing = [row for row in rows if not _passes(row[2])]
    label, metric, gap = min(failing, key=lambda row: row[2]["gap"])
    return (f"SC-001 NOT met: the tuned model ({path}) does not clear the bar of 10 points with "
            f"a 95% interval above zero on every comparison; the weakest is {LABELS[metric]} "
            f"{label}, a gap of {_points(gap['gap'])} points (95% CI "
            f"{_points(gap['ci_low'])} to {_points(gap['ci_high'])}).")


def _table(final_doc: Mapping[str, Any]) -> list[str]:
    models = final_doc["models"]
    lines = ["| Metric | Base | Prompt-only | SFT | Tuned |", "|---|---|---|---|---|"]
    for metric in METRIC_NAMES:
        cells = " | ".join(_cell(models[name][metric]) for name, _ in COLUMNS)
        lines.append(f"| {LABELS[metric]} | {cells} |")
    return lines


def _gaps_table(sc001: Mapping[str, Any]) -> list[str]:
    lines = ["| Comparison | Metric | Gap (points) | 95% CI (points) "
             "| At least 10 points, CI above 0 |",
             "|---|---|---|---|---|"]
    for label, metric, gap in _gap_rows(sc001):
        verdict = "yes" if _passes(gap) else "NO"
        lines.append(f"| {label} | {LABELS[metric]} | {_points(gap['gap'])} | "
                     f"{_points(gap['ci_low'])} to {_points(gap['ci_high'])} | {verdict} |")
    return lines


def _what_ran(final_doc: Mapping[str, Any], source_doc: Mapping[str, Any] | None,
              choice: str | None) -> list[str]:
    path = final_doc["tuned_path"]
    signal = (source_doc or {}).get("reward_signal")
    seen = (f" The reward differed within {signal['groups_with_variance']} of "
            f"{signal['groups_total']} groups of sampled answers." if signal else "")
    if path == "grpo":
        lines = ["Path that ran: grpo. GRPO (reinforcement learning that samples several answers "
                 "per prompt and rewards the exact-match ones) trained the tuned model after SFT."
                 + seen + " A reward that differs inside a group is what gives GRPO something to "
                 "learn from."]
    elif path == "rft":
        lines = ["Path that ran: rft. The tuned model comes from the fallback, "
                 "rejection-sampling fine-tuning (train again on the model's own answers that "
                 "matched the doc-following call exactly), because GRPO did not run or its "
                 "reward gave no usable signal." + seen]
    else:
        lines = ["Path that ran: sft-only. No GRPO or rejection-sampling result was provided, so "
                 "the tuned model is the SFT model (supervised fine-tuning on correct example "
                 "calls) and the SFT and Tuned columns are identical."]
    return lines + ([choice] if choice else [])


def _trap_example(base_probe: Mapping[str, Any] | None, trap_doc: str | None) -> list[str]:
    if base_probe is None:
        return ["The probe file was not found, so no trap example is quoted. "
                "`scripts/probe_base.py` writes results/base_probe.json."]
    example = base_probe["trap_example"]

    def as_text(call: Any) -> str:
        return call if isinstance(call, str) else json.dumps(call)

    quoted = "\n".join(f"> {line}" for line in str(example["model_output"]).splitlines() or [""])
    doc_line = (" ".join(trap_doc.split()) if trap_doc else
                "(the tool documentation is not stored in the probe file)")
    return [
        "One example from the 100-example development split, answered by the base model "
        f"(results/base_probe.json; task {example['task_id']}). It is not one of the held-out "
        "examples.",
        "",
        f"- Request: {example['request']}",
        f"- What the tool documentation says: {doc_line}",
        f"- Call that follows the documentation: `{as_text(example['oracle_call'])}`",
        f"- Call that follows habit instead: `{as_text(example['habit_call'])}`",
        "- What the base model answered:",
        "",
        quoted,
    ]


def _e3_line(final_doc: Mapping[str, Any]) -> str:
    base = final_doc["models"]["base"]["normal_tool_f1"]["value"]
    tuned = final_doc["models"]["tuned"]["normal_tool_f1"]["value"]
    span = f"(base {_pct(base)}, tuned {_pct(tuned)})"
    if round(base - tuned, 6) > E3_LIMIT:
        return (f"E-3 (forgetting): normal-tool F1 dropped by {_points(base - tuned)} points "
                f"versus base {span}. Tuning made ordinary tools worse; that cost belongs next "
                "to the gain.")
    return (f"E-3 (forgetting): normal-tool F1 did not drop by more than 5 points versus base "
            f"{span}.")


def render_report(final_doc: Mapping[str, Any], base_probe: Mapping[str, Any] | None = None, *,
                  source_doc: Mapping[str, Any] | None = None, trap_doc: str | None = None,
                  choice: str | None = None) -> str:
    """The plain-language markdown report. Its first line is the SC-001 verdict."""
    n = final_doc["models"]["tuned"]["tool_call_f1"]["n"]
    traps = final_doc["models"]["tuned"]["prior_trap_rate"]["n"]
    normal = final_doc["models"]["tuned"]["normal_tool_f1"]["n"]
    seed = final_doc["seed"]
    sections = [
        _headline(final_doc),
        "",
        "## The four metrics",
        "",
        f"Each cell is the value with its 95% bootstrap interval, on {n} held-out examples "
        f"({traps} trap examples, {normal} normal ones). Held-out means the tools and twists "
        "never appeared in training.",
        "",
        *_table(final_doc),
        "",
        "- Prior-trap rate: how often the model followed habit against the tool documentation "
        "on trap examples. Lower is better.",
        "- Tool-call F1: how often the model produced exactly the right call. Higher is better.",
        "- Tool-call validity: share of answers that are a well-formed call to a known tool.",
        "- Normal-tool F1: tool-call F1 on tools with no twist.",
        "",
        "## The SC-001 gaps",
        "",
        "SC-001 asks for a gain of at least 10 points over both baselines on both prior-trap "
        "rate and tool-call F1, with the 95% interval above zero. Gaps are improvements, in "
        "points.",
        "",
        *_gaps_table(final_doc["sc001"]),
        "",
        "A gap whose interval includes zero is not evidence of a difference between the models.",
        "",
        "## What ran",
        "",
        *_what_ran(final_doc, source_doc, choice),
        "",
        "## A trap example",
        "",
        *_trap_example(base_probe, trap_doc),
        "",
        "## Did normal tools get worse?",
        "",
        _e3_line(final_doc),
        "",
        "## Caveats",
        "",
        f"- Only {n} held-out examples (n={n}); the intervals show how much that sample alone "
        "could move the numbers.",
        f"- One run on a single seed (seed {seed}): the intervals do not include run-to-run "
        "training variation.",
        "- The reward and the score are exact-match: two valid calls with different argument "
        "forms score 0.",
        "- Only invented tools were tested; nothing here says the model follows documentation "
        "for real-world APIs.",
    ]
    return "\n".join(sections) + "\n"


# ---- command line ---------------------------------------------------------------------------


def _output_pair(text: str) -> tuple[str, str]:
    name, sep, path = text.partition("=")
    if not sep or name not in MODEL_NAMES or not path:
        raise argparse.ArgumentTypeError(
            f"expected NAME=PATH with NAME one of {', '.join(MODEL_NAMES)}, got {text!r}")
    return name, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build results/final.json and report.md (the SC-001 verdict) from the "
                    "results files and the per-example outputs of the held-out runs.",
        epilog=(
            "Per-example outputs (--outputs NAME=PATH, repeat per model; each is a ProgressStore "
            "JSONL file of {id, output}): base and prompt_only are written by run_eval.py in "
            "results/progress/base.jsonl and results/progress/prompt_only.jsonl (defaults); "
            "sft is <work-dir>/progress_sft/tuned.jsonl, written by train_sft.py; tuned is the "
            "'tuned' progress file of the grpo or rft run, in the progress directory that run "
            "used under its work dir (<progress dir>/tuned.jsonl). This script does not fix "
            "that location: check the train_grpo.py / train_rft.py output. With no grpo or "
            "rft file, tuned defaults to the sft outputs."),
    )
    parser.add_argument("--baselines", default="results/eval_baselines.json")
    parser.add_argument("--sft", default="results/sft.json")
    parser.add_argument("--grpo", default="results/grpo.json", help="optional")
    parser.add_argument("--rft", default="results/rft.json", help="optional")
    parser.add_argument("--probe", default="results/base_probe.json",
                        help="optional; the quoted trap example")
    parser.add_argument("--out", default="results/final.json")
    parser.add_argument("--report", default="report.md")
    parser.add_argument("--seed", type=int, default=0, help="must equal the baselines' seed")
    parser.add_argument("--hardness", type=int, default=0,
                        help="make_splits hardness the eval and the probe ran with (0 to "
                             f"{MAX_HARDNESS}); rebuilds the held-out tasks and the trap "
                             "example's tool documentation")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--outputs", action="append", default=[], type=_output_pair,
                        metavar="NAME=PATH", help="per-example outputs of a model (see below)")
    return parser


def _fail(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _read_doc(path: str, kind: str, hint: str, *, required: bool) -> dict[str, Any] | None:
    if not Path(path).is_file():
        if required:
            raise ReportInputError(f"{path} not found. It is written by {hint}")
        return None
    errors = check_results.validate_file(path)
    if errors:
        raise ReportInputError(f"{path} is not a valid results file:\n" + "\n".join(errors))
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc["kind"] != kind and not (kind == "rl" and doc["kind"] in ("grpo", "rft")):
        raise ReportInputError(f"{path}: expected kind {kind!r}, got {doc['kind']!r}")
    return doc


def _trap_doc_text(probe: Mapping[str, Any] | None, probe_path: str, seed: int,
                   hardness: int) -> str | None:
    """The tool doc of the probe's trap example, rebuilt from the dev split.

    None when there is no probe. A probe whose trap example is not in the rebuilt dev split
    raises ``ReportInputError``: the quoted example would be the wrong task.
    """
    if probe is None:
        return None
    example = probe["trap_example"]
    for task in make_splits(seed, hardness).dev:
        if task.id == example["task_id"] and task.request == example["request"]:
            return task.tool.doc
    raise ReportInputError(
        f"{probe_path}: its trap example ({example['task_id']}) is not in the dev split for "
        f"seed {seed}, hardness {hardness}; pass the same --hardness as the probe run")


def _write_text_atomic(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    baselines = _read_doc(args.baselines, "eval_baselines", BASELINES_HINT, required=True)
    sft_doc = _read_doc(args.sft, "sft", SFT_HINT, required=True)
    grpo_doc = _read_doc(args.grpo, "rl", "", required=False)
    rft_doc = _read_doc(args.rft, "rl", "", required=False)
    probe = None
    if Path(args.probe).is_file():
        probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    return baselines, sft_doc, grpo_doc, rft_doc, probe


def _output_paths(args: argparse.Namespace, tuned_path: str) -> dict[str, str]:
    paths = {"base": "results/progress/base.jsonl",
             "prompt_only": "results/progress/prompt_only.jsonl"}
    paths.update(dict(args.outputs))
    if "sft" not in paths:
        raise ReportInputError(
            "no sft outputs given: pass --outputs sft=<work-dir>/progress_sft/tuned.jsonl "
            f"(written by {SFT_HINT})")
    if "tuned" not in paths:
        if tuned_path != "sft":
            raise ReportInputError(
                f"no tuned outputs given for the {tuned_path} run: pass --outputs "
                "tuned=<progress dir>/tuned.jsonl (written by the train_grpo.py / "
                "train_rft.py evaluation step)")
        paths["tuned"] = paths["sft"]
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise ReportInputError("output file not found: " + "; ".join(missing)
                               + ". Run the Colab step that produces that model's results.")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0 <= args.hardness <= MAX_HARDNESS:
        parser.error(f"--hardness must be between 0 and {MAX_HARDNESS}, got {args.hardness}")
    try:
        baselines, sft_doc, grpo_doc, rft_doc, probe = _load_inputs(args)
        if baselines["seed"] != args.seed:
            raise ReportInputError(
                f"--seed {args.seed} differs from the seed {baselines['seed']} in "
                f"{args.baselines}; the held-out examples would not match")
        trap_doc = _trap_doc_text(probe, args.probe, args.seed, args.hardness)
        tuned_path, source = pick_tuned(grpo_doc, rft_doc, sft_doc)
        paths = _output_paths(args, tuned_path)
    except ReportInputError as exc:
        return _fail(str(exc))

    tasks = make_splits(args.seed, args.hardness).heldout[:baselines["n_examples"]]
    try:
        scores = {name: score_all(tasks, load_outputs(path)) for name, path in paths.items()}
        doc = build_final(baselines, sft_doc, grpo_doc, rft_doc, scores, seed=args.seed,
                          report_path=args.report, n_resamples=args.n_resamples)
        write_results(args.out, doc, check_results.validate)
    except (ReportInputError, NoEligibleExamplesError, ResultsInvalidError) as exc:
        return _fail(f"nothing was written: {exc}", 1)

    report = render_report(
        doc, probe, source_doc=None if tuned_path == "sft" else source,
        trap_doc=trap_doc,
        choice=choice_note(grpo_doc, rft_doc))
    _write_text_atomic(args.report, report)
    print(report.splitlines()[0])
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
