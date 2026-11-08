"""Resumable three-way eval harness: base, prompt-only and tuned models (FR-002, FR-006, E-6).

Pure Python: no torch or transformers import here. A model is a ``GenerateFn`` (prompts in,
completion texts out, same order), so tests use fakes and ``scripts/run_eval.py`` supplies the
real GPU generator.

Public API:

* ``MODEL_ID``, ``DECODING``, ``BUDGET_GPU_SECONDS``: the model id (verify on Colab), the
  greedy 256-token decoding of FR-002, and the 3 hour limit in seconds.
* ``build_prompts(tasks, variant)``: ``base`` and ``tuned`` get the plain prompt, ``prompt_only``
  gets the same prompt plus the one ``FOLLOW_DOCS_LINE``.
* ``run_variant(...)``: generate in batches, record every batch to a ``ProgressStore`` at once
  and skip ids already recorded, so an interrupted session loses at most one batch.
* ``run_fingerprint(name, prompts, run_key)`` and ``open_progress(path, name, tasks, run_key)``:
  every variant's progress file carries a run fingerprint, so a progress directory reused
  for another seed, split, model, revision or adapter raises ``StaleProgressError`` (see
  ``alien.persist``) instead of replaying the other run's outputs.
* ``evaluate_variants(...)``: run and score several variants; sums GPU time across sessions
  through ``progress_dir/timing.json`` so FR-002 is measured, not assumed.
* ``build_results(...)``: the ``eval_baselines`` results document (base and prompt_only only).
* ``build_tuned_results(...)``: the minimal data of a ``--which tuned`` run (no ``kind``).
* ``write_results(path, doc, validate)``: validate, then atomic temp-file + rename.

``timing.json`` shape: ``{"seconds": {"<variant>": <float>}}``, summed over every session of
the SAME run. A variant whose progress file holds no records (deleted, or never written) starts
again from 0 s, so deleting a stale progress file never carries its old seconds into the new run.

Run fingerprint: the hex SHA-256 of ``json.dumps([run_key, variant, prompts],
ensure_ascii=False)`` (UTF-8), where ``prompts`` are the variant's exact prompt strings in task
order and ``run_key`` is the caller's free-text description of everything else that changes the
outputs (model id, revision, adapter, decoding). It is stored in ``<variant>.jsonl.key`` next to
the progress file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alien.generator import Task
from alien.oracle import render_prompt
from alien.persist import ProgressStore, StaleProgressError, resume_plan  # noqa: F401
from alien.scoring import NoEligibleExamplesError, evaluate

GenerateFn = Callable[[Sequence[str]], list[str]]

MODEL_ID = "CohereLabs/tiny-aya-global"  # verify on Colab
DECODING: dict[str, Any] = {"max_new_tokens": 256, "temperature": 0.0, "do_sample": False}
BUDGET_GPU_SECONDS = 10800
VARIANTS = ("base", "prompt_only", "tuned")
BASELINE_MODELS = ("base", "prompt_only")
TIMING_FILE = "timing.json"


class ResultsInvalidError(ValueError):
    """A results document failed validation; the message lists every problem."""


def build_prompts(tasks: Sequence[Task], variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {', '.join(VARIANTS)}")
    return [render_prompt(task, follow_docs=variant == "prompt_only") for task in tasks]


def run_fingerprint(name: str, prompts: Sequence[str], run_key: str = "") -> str:
    """Hex SHA-256 naming one run of variant ``name`` (format: see the module docstring)."""
    payload = json.dumps([run_key, name, list(prompts)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def open_progress(
    path: str | os.PathLike[str], name: str, tasks: Sequence[Task], run_key: str = ""
) -> ProgressStore:
    """The variant's fingerprinted ``ProgressStore``; ``StaleProgressError`` on a mismatch."""
    return ProgressStore(path, run_fingerprint(name, build_prompts(tasks, name), run_key))


def _write_json_atomic(path: str | os.PathLike[str], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_variant(
    name: str,
    tasks: Sequence[Task],
    generate: GenerateFn,
    store: ProgressStore,
    batch_size: int = 8,
    clock: Callable[[], float] = time.monotonic,
    on_batch_seconds: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Generate the outputs still missing from ``store``; return all outputs in task order.

    ``seconds`` counts only this session's generation time. ``on_batch_seconds`` is called
    with each batch's time right after the batch is recorded.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    missing = set(resume_plan([task.id for task in tasks], store))
    todo = [task for task in tasks if task.id in missing]
    seconds = 0.0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        prompts = build_prompts(batch, name)
        began = clock()
        texts = generate(prompts)
        elapsed = clock() - began
        if len(texts) != len(batch):
            raise ValueError(
                f"generator returned {len(texts)} completions for {len(batch)} prompts"
            )
        for task, text in zip(batch, texts):
            store.record(task.id, {"output": text})
        seconds += elapsed
        if on_batch_seconds is not None:
            on_batch_seconds(elapsed)
    done = store.results()
    return {
        "outputs": [done[task.id]["output"] for task in tasks],
        "seconds": seconds,
        "generated": len(todo),
    }


def _require_eligible(tasks: Sequence[Task]) -> None:
    """E-5: fail before any GPU time is spent if a metric would have no examples."""
    if not tasks:
        raise NoEligibleExamplesError("tool_call_f1: no examples")
    if not any(task.is_trap for task in tasks):
        raise NoEligibleExamplesError("prior_trap_rate: no trap examples")
    if all(task.is_trap for task in tasks):
        raise NoEligibleExamplesError("normal_tool_f1: no normal examples")


def _read_timing(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text())["seconds"].items()}


def evaluate_variants(
    tasks: Sequence[Task],
    generators: Mapping[str, GenerateFn],
    progress_dir: str | os.PathLike[str],
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    batch_size: int = 8,
    clock: Callable[[], float] = time.monotonic,
    run_key: str = "",
) -> dict[str, Any]:
    """Run and score every variant in ``generators``, resuming from ``progress_dir``.

    ``run_key`` describes what else shapes the outputs (model, revision, adapter, decoding);
    a progress file written under a different prompt set or key raises ``StaleProgressError``
    before any generation, for every variant.

    Returns ``{"models": {name: Metrics}, "gpu_seconds": float, "n_examples": int}`` where
    ``gpu_seconds`` is the generation time of these variants over all sessions so far.
    """
    _require_eligible(tasks)
    progress_dir = Path(progress_dir)
    timing_path = progress_dir / TIMING_FILE
    timing = _read_timing(timing_path)
    stores = {name: open_progress(progress_dir / f"{name}.jsonl", name, tasks, run_key)
              for name in generators}
    for name, store in stores.items():
        if not store.done_ids():
            timing.pop(name, None)  # no records: any recorded seconds belong to another run
    models: dict[str, Any] = {}
    for name, generate in generators.items():
        def add_seconds(elapsed: float, name: str = name) -> None:
            timing[name] = timing.get(name, 0.0) + elapsed
            _write_json_atomic(timing_path, {"seconds": timing})

        store = stores[name]
        run = run_variant(name, tasks, generate, store, batch_size, clock, add_seconds)
        models[name] = evaluate(tasks, run["outputs"], n_resamples=n_resamples, seed=seed)
    return {
        "models": models,
        "gpu_seconds": float(sum(timing.get(name, 0.0) for name in generators)),
        "n_examples": len(tasks),
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_results(
    evaluated: Mapping[str, Any],
    *,
    seed: int,
    model_revision: str,
    split: str = "heldout",
    decoding: Mapping[str, Any] = DECODING,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The ``eval_baselines`` document; ``git_sha`` comes from the GIT_SHA environment variable."""
    missing = [name for name in BASELINE_MODELS if name not in evaluated["models"]]
    if missing:
        raise ValueError(f"eval_baselines needs models {', '.join(BASELINE_MODELS)}; "
                         f"missing {', '.join(missing)}")
    gpu_seconds = evaluated["gpu_seconds"]
    return {
        "kind": "eval_baselines",
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "n_examples": evaluated["n_examples"],
        "split": split,
        "model_revision": model_revision,
        "decoding": dict(decoding),
        "models": {name: evaluated["models"][name] for name in BASELINE_MODELS},
        "gpu_seconds": gpu_seconds,
        "within_budget": gpu_seconds < BUDGET_GPU_SECONDS,
    }


def build_tuned_results(
    evaluated: Mapping[str, Any],
    *,
    seed: int,
    model_revision: str,
    adapter: str,
    split: str = "heldout",
    decoding: Mapping[str, Any] = DECODING,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Data of a ``--which tuned`` run. It has no ``kind``: later cards merge it themselves."""
    return {
        "schema_version": 1,
        "created_at": created_at or _now(),
        "git_sha": os.environ.get("GIT_SHA") or None,
        "seed": seed,
        "n_examples": evaluated["n_examples"],
        "split": split,
        "model_revision": model_revision,
        "adapter": adapter,
        "decoding": dict(decoding),
        "models": {"tuned": evaluated["models"]["tuned"]},
        "gpu_seconds": evaluated["gpu_seconds"],
    }


def write_results(
    path: str | os.PathLike[str],
    doc: dict[str, Any],
    validate: Callable[[Any], list[str]],
) -> None:
    """Write ``doc`` atomically, but only if ``validate(doc)`` finds no problem."""
    errors = validate(doc)
    if errors:
        raise ResultsInvalidError("\n".join(errors))
    _write_json_atomic(path, doc)
