# Project context

## What this is

Alien Manual (Lexsi hackathon, Track 3): a generator of invented tools whose docs contradict common
habit, a scripted oracle for the correct call, and a pipeline that fine-tunes Tiny Aya Global
(model id CohereLabs/tiny-aya-global, gated) with SFT then GRPO (or rejection sampling) to follow the doc.
It reports four metrics (prior_trap_rate, tool_call_f1, tool_call_validity, normal_tool_f1) for base,
prompt-only and tuned models with bootstrap CIs, and an honest SC-001 verdict.
Source of truth: the project PRD (kept in a separate private repo; requirement ids FR-, SC-, E-, R-, B- refer to it).
How to run it on Colab: `RUNBOOK.md`.

**Status:** all code is merged and tested on the laptop (641 tests, no GPU). The full pipeline has run
on the unseen-tools split (`holdout="tools"`, seed 0, one run) on a Kaggle T4 and its results are in
`results/` (`base_probe`, `eval_baselines`, `sft`, `grpo`, `final`) with `report.md` at the root. SC-001
verdict: **met** (tuned beats base and prompt-only by 51.9 points on prior-trap rate and 45.0 on tool-call
F1, 95% CIs above zero, n=200 held-out). Read it with these caveats:
- The gain comes from SFT. The GRPO model scores exactly what SFT scores (prior-trap 6.8%, F1 94.5%), so
  GRPO added nothing measurable; only 69 of 400 reward groups had any variance.
- One seed, one run, invented tools only; the CIs do not include training variation.
- The first run (one twist family held out) failed SC-001 and is kept in `results/first_run_twist_holdout/`.
Still awaiting a GPU: the Space deploy. The `results/*.json` files were copied by hand from the Kaggle
notebook output and each was checked against a hash printed in the notebook (canonical JSON SHA-256);
`report.md` is the regenerated `render_report` output and matches the notebook's file hash.
Kaggle gotcha: a detached `!(setsid nohup ... &)` launch is killed when the cell ends; start the
pipeline with `subprocess.Popen(..., start_new_session=True)` from a cell instead.

## Directory map

- `alien/generator.py`: domains, tools, twists (flip_args, invert_op, odd_unit, rename), tasks with
  `oracle_call` and `habit_call`; `hardness` changes docs only.
- `alien/oracle.py`: `render_prompt` (plain vs `FOLLOW_DOCS_LINE`), `oracle_call`, `habit_call`,
  chat-example builders.
- `alien/splits.py`: `make_splits(seed, hardness, tools_per_domain, holdout="tools")` -> train 600 / dev 100 /
  held-out 240, disjoint by whole domain, all four twists in every split (`heldout_twist` is None), with a leak
  check; JSON round trip. `holdout="twist"` reproduces the first run (one twist family held out); its results
  are in `results/first_run_twist_holdout/` and `build_report.py` has no flag for that mode.
- `alien/scoring.py`: `parse_calls` (reads only the first 32768 chars, never raises; if the reply has a ``` fence only the
  fenced bodies are read, so a bare call beside a fenced answer is not counted), `score_example`,
  `evaluate` (4 metrics with bootstrap CI), `paired_gap`; integral floats count as valid integers (42.000000000000000000001 parses to 42.0, so it reads as 42).
- `alien/persist.py`: `ProgressStore` (resumable JSONL; optional fingerprint in `<name>.jsonl.key`,
  `StaleProgressError` on a different run), manifest-listed checkpoints (a torn `MANIFEST.json` is
  ignored), upload retry.
- `alien/reward.py`: exact-match GRPO reward, `RewardMonitor` (dead-reward detection), `to_grpo_rows`.
- `alien/eval.py`: variants base / prompt_only / tuned, `evaluate_variants`, `run_fingerprint` /
  `open_progress` (fingerprinted progress), result builders, `MODEL_ID`, `DECODING`, 3-hour GPU budget.
- `scripts/check_results.py`: validates every results JSON against the contract (stdlib only); also
  rejects `hf_`, `sk-` and `Bearer` token shapes. It scans the decoded document first and, on a hit,
  returns only path-named secret errors (never the value); odd JSON is one error, not a crash. `--help` exits 0.
- `scripts/probe_base.py`: E-1 headroom probe on the dev split; `scripts/run_eval.py`: baselines and
  `--which tuned` eval, holds the HF generate helpers, `make_run_key` (the one run-fingerprint builder:
  model, requested and resolved revision, adapter, hardness, decoding; shared by the train scripts) and
  the shared `--hardness` helpers (`add_hardness_argument`, `load_splits_checked` which also leak-checks
  `--splits` files, exit 2, `MAX_HARDNESS`); `scripts/train_sft.py`: SFT + eval + `run_config.json`
  helpers (`build_run_config`, `write_run_config`, `refuse_reused_final`);
  `scripts/train_grpo.py`: GRPO + shared RL helpers; `scripts/train_rft.py`: rejection-sampling
  fallback; `scripts/build_report.py`: final.json and report.md; `scripts/spike.py`: helpers for the notebook.
- `notebooks/spike.ipynb`: the Colab spike (AS-1..AS-3), writes results/spike.json.
- `space/app.py`, `space/requirements.txt`, `space/README.md`: the Gradio demo Space (copy with `alien/`);
  its verdict comes from `score_example`, so an oracle call plus an extra or habit call is not "matches".
- `tests/`: one `test_*.py` per module above (generator, scoring, persist, reward, eval, probe_base,
  train_sft, train_grpo, train_rft, build_report, check_results, spike, space).
- `results/`: output folder for the JSON files.

## Data flow

generator -> splits -> oracle prompts -> probe / eval / SFT / GRPO or RFT -> results JSON (one per step)
-> `scripts/check_results.py` (every file) -> `scripts/build_report.py` -> final.json + report.md.
Per-example outputs are JSONL ProgressStore files (`{id, output}`); build_report rescores them and
refuses to write if they do not reproduce the recorded metrics. Work dir layout: ckpt (SFT),
grpo, rft and progress_sft folders, each with manifest-listed checkpoints.

## Commands

```
python -m pytest -q                      # 631 tests; pyproject puts . and scripts on the path
python scripts/check_results.py results/<kind>.json
python scripts/probe_base.py --seed 0 [--hardness N]
python scripts/run_eval.py --which base,prompt_only --seed 0 [--hardness N] --out results/eval_baselines.json
python scripts/train_sft.py --hub-repo <user>/alien-sft --seed 0 [--hardness N] --work-dir <W>
python scripts/train_grpo.py --hub-repo <user>/alien-grpo --seed 0 [--hardness N] --work-dir <W>
python scripts/train_rft.py --hub-repo <user>/alien-rft --seed 0 [--hardness N] --work-dir <W>
python scripts/build_report.py --seed 0 [--hardness N] --outputs sft=<path> --outputs tuned=<path>
```

Tests need only pytest (Python 3.12). `--dry-run` on the three train scripts and `--help` on the
others work without torch. Colab steps, exit codes and the "unverified" table: `RUNBOOK.md`.

## Conventions

- Stdlib-only core; torch, transformers, peft, trl, datasets, huggingface_hub, gradio are imported
  inside functions only.
- Results JSON contract kinds: base_probe, eval_baselines, sft, grpo, rft, spike, final (no extra
  fields). Never write NaN; null triggers the E-5 error by design.
- `--fake oracle|habit` flags are test-only (skip model loading). `--smoke` never uploads or writes results.
- Anything not checkable without a GPU is marked `VERIFY ON COLAB` in the code.
- Never read, print, log or commit `HF_TOKEN` (B-3); the hub library reads it from the environment.
- Train only on the train split; the held-out split is not tuned against (B-2, B-6).

## Gotchas

- The PRD says AgentTune is Cohere's; it is Lexsi-Labs'. The code does not use it: GRPO is plain TRL.
  The PRD's AuditKIT is not used either (scripted scoring only).
- Hardness hook (E-1): `--hardness N` (0..3, default 0) on probe_base, run_eval, train_sft/grpo/rft and
  build_report (where it also picks the held-out split); no hand-made `--splits` file is needed. A
  `--splits` file's recorded seed and hardness must match the flags (else exit 2); `train_grpo
  --auto-fallback` forwards it to train_rft. Calls and ids are identical across hardness. Known gap:
  `base_probe.json` and `eval_baselines.json` do not record hardness, so the operator must pass the
  same value everywhere; a hardness-1 probe rebuilt at 0 is not detected.
- Progress dirs are fingerprinted (`<name>.jsonl.key`): changing seed, hardness, adapter, revision or
  `--n` over an existing dir makes run_eval/probe_base exit 2 ("fresh --progress-dir"); old files without
  a `.key` are refused. The train scripts' tuned evals use the same key (`StaleProgressError`, exit 2).
- `<W>/ckpt/final` holds `run_config.json` (script, seed, hardness, revision, hparams, extras). train_sft/grpo/rft
  exit 2 ("use a fresh --work-dir") on a final checkpoint from a different run or with no
  `run_config.json`; checkpoints finished under older code have none and must be retrained or deleted.
  Resume = same command, same flags, same `--work-dir`.
- `--seed` must match across eval_baselines, sft, grpo/rft and build_report (else exit 2).
- Base prompt and prompt-only prompt differ by exactly one line (`FOLLOW_DOCS_LINE`).
- Exact-match reward gives no partial credit: an SFT model that is already always right looks like a
  dead reward (exit 3) and goes to RFT.
- If both grpo and rft results exist, build_report picks the better tuned f1; supply that run's outputs.
- Model loading uses `torch_dtype=` in eval, probe and Space but `dtype=` in the train scripts; one of
  them will warn or fail on Colab (see RUNBOOK section 4).
- Trap example in the report comes from the dev split, not held-out.
