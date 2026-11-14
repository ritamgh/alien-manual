**This directory is NOT committed.**

# Rerun on new splits: SFT only, two seeds (Kaggle T4)

## What was run

- Model: Tiny Aya (`CohereLabs/tiny-aya-global`), fp16 LoRA SFT. **SFT only. No GRPO was run** (the SFT result is also the reported tuned result, `tuned_path` = `sft`).
- Run A (`runA_seed0_h1`): seed 0, hardness 1. Split = `make_splits(0, 1)` with the default `holdout="tools"`: the held-out examples come from tools never seen in training. Held-out examples scored: 200 (from final.json `n` of tool-call F1).
- Run B (`runB_seed1_h0`): seed 1, hardness 0. Split = `make_splits(1, 0)` with the default `holdout="tools"`: the held-out examples come from tools never seen in training. Held-out examples scored: 200 (from final.json `n` of tool-call F1).
- Code: commit `b179f32b6e9a7302df4375c34f7d771a69230ac9`. That is `e12d318` plus one commit that only changes `RUNBOOK.md`, so the code is the same as `e12d318`. Every final.json records this sha.
- Hardware and secrets: Kaggle T4. `HF_TOKEN` was used only to download the gated base model; `ALIEN_NO_UPLOAD=1` was set during training, so nothing was uploaded by the pipeline.
- Library versions (`versions.json`): trl 1.7.1, torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2.
- `versions.json` also holds the TRL `GRPOConfig` defaults (beta = 0.0, scale_rewards = group, loss_type = dapo). They were printed only to settle an earlier, unverified assumption about those defaults. GRPO was not run.
- Two environment fixes the notebook needed: uninstall the stale `torchao`, and set `CUDA_VISIBLE_DEVICES=0` on the T4 x2 machine. They do not change the method.

## What Run A's SFT was trained on

Run A uses hardness 1, so SFT was **trained and evaluated on the reworded, folded documents**. At hardness 1 the twist is folded into the parameter descriptions (for example a swapped from/to description) or, for the rename pattern, into a reworded "Registered as ..." line, and one benign extra Note is added (`alien/generator.py`, `_build_doc` and `_param_descriptions`).

So the claim Run A supports is: **SFT learns the reworded format too.** It does **not** support "an SFT trained on plain `Note:` lines survives reworded notes". That second experiment (train at hardness 0, test at hardness 1) was **not run**.

Run B uses the plain `Note:` wording (hardness 0) but seed 1, which gives a different set of held-out tools than seed 0. It is a second seed on the original wording, but not on the same held-out tools as the committed seed-0 result.

## Results (this rerun)

Values are point estimate [95% CI] (n = examples that metric applies to). All numbers come from each run's `results/final.json`.

### Run A: seed 0, hardness 1

| Metric | Base | Prompt-only | SFT |
|---|---|---|---|
| Prior-trap rate (lower is better) | 88.3% [83.4%, 92.8%] (n=162) | 86.4% [81.0%, 91.4%] (n=162) | 1.9% [0.0%, 4.1%] (n=162) |
| Tool-call F1 | 25.5% [20.0%, 31.5%] (n=200) | 27.0% [21.5%, 33.5%] (n=200) | 98.5% [96.5%, 100.0%] (n=200) |
| Tool-call validity | 100.0% [100.0%, 100.0%] (n=200) | 100.0% [100.0%, 100.0%] (n=200) | 100.0% [100.0%, 100.0%] (n=200) |
| Normal-tool F1 | 100.0% [100.0%, 100.0%] (n=38) | 100.0% [100.0%, 100.0%] (n=38) | 100.0% [100.0%, 100.0%] (n=38) |

SC-001 verdict: **met** (SFT is the tuned model). Gaps, positive = SFT better:

| Gap | vs base | vs prompt-only |
|---|---|---|
| Prior-trap rate (baseline minus SFT) | +86.4 pts [+81.5, +91.4] | +84.6 pts [+78.9, +89.9] |
| Tool-call F1 (SFT minus baseline) | +73.0 pts [+67.0, +79.0] | +71.5 pts [+65.0, +77.5] |

### Run B: seed 1, hardness 0

| Metric | Base | Prompt-only | SFT |
|---|---|---|---|
| Prior-trap rate (lower is better) | 56.3% [48.7%, 64.1%] (n=158) | 54.4% [46.5%, 62.6%] (n=158) | 3.2% [0.6%, 6.0%] (n=158) |
| Tool-call F1 | 52.0% [44.5%, 59.0%] (n=200) | 52.5% [45.0%, 59.0%] (n=200) | 94.5% [91.0%, 97.5%] (n=200) |
| Tool-call validity | 100.0% [100.0%, 100.0%] (n=200) | 100.0% [100.0%, 100.0%] (n=200) | 98.5% [97.0%, 100.0%] (n=200) |
| Normal-tool F1 | 95.2% [87.8%, 100.0%] (n=42) | 92.9% [83.9%, 100.0%] (n=42) | 100.0% [100.0%, 100.0%] (n=42) |

SC-001 verdict: **met** (SFT is the tuned model). Gaps, positive = SFT better:

| Gap | vs base | vs prompt-only |
|---|---|---|
| Prior-trap rate (baseline minus SFT) | +53.2 pts [+45.3, +61.0] | +51.3 pts [+43.1, +59.3] |
| Tool-call F1 (SFT minus baseline) | +42.5 pts [+35.5, +49.5] | +42.0 pts [+35.5, +49.0] |

## First pass vs rerun

First pass = recovered from the saved output of the first notebook run (notebook sha256 prefix `8c7f84bb83c3771a`), same seeds and settings; its adapters and per-reply outputs were lost, so only the final.json numbers survive. Rerun = this directory (SFT retrained with the same seeds; GPU training is not bit-reproducible). Difference = rerun minus first pass, in points.

### Run A: seed 0, hardness 1

| Model | Metric | First pass | Rerun | Difference |
|---|---|---|---|---|
| Base | Prior-trap rate | 88.3% | 88.3% | +0.0 |
| Base | Tool-call F1 | 25.5% | 25.5% | +0.0 |
| Base | Tool-call validity | 100.0% | 100.0% | +0.0 |
| Base | Normal-tool F1 | 100.0% | 100.0% | +0.0 |
| Prompt-only | Prior-trap rate | 86.4% | 86.4% | +0.0 |
| Prompt-only | Tool-call F1 | 27.0% | 27.0% | +0.0 |
| Prompt-only | Tool-call validity | 100.0% | 100.0% | +0.0 |
| Prompt-only | Normal-tool F1 | 100.0% | 100.0% | +0.0 |
| SFT | Prior-trap rate | 1.9% | 1.9% | +0.0 |
| SFT | Tool-call F1 | 98.5% | 98.5% | +0.0 |
| SFT | Tool-call validity | 100.0% | 100.0% | +0.0 |
| SFT | Normal-tool F1 | 100.0% | 100.0% | +0.0 |
| SC-001 | trap-rate gap vs base | +86.4 pts [+81.5, +91.4] | +86.4 pts [+81.5, +91.4] | +0.0 |
| SC-001 | F1 gap vs base | +73.0 pts [+67.0, +79.0] | +73.0 pts [+67.0, +79.0] | +0.0 |
| SC-001 | trap-rate gap vs prompt-only | +84.6 pts [+78.9, +89.9] | +84.6 pts [+78.9, +89.9] | +0.0 |
| SC-001 | F1 gap vs prompt-only | +71.5 pts [+65.0, +77.5] | +71.5 pts [+65.0, +77.5] | +0.0 |

First-pass SC-001 verdict: met; rerun: met. First-pass git sha: `e12d318`; rerun: `b179f32`.

### Run B: seed 1, hardness 0

| Model | Metric | First pass | Rerun | Difference |
|---|---|---|---|---|
| Base | Prior-trap rate | 56.3% | 56.3% | +0.0 |
| Base | Tool-call F1 | 52.0% | 52.0% | +0.0 |
| Base | Tool-call validity | 100.0% | 100.0% | +0.0 |
| Base | Normal-tool F1 | 95.2% | 95.2% | +0.0 |
| Prompt-only | Prior-trap rate | 54.4% | 54.4% | +0.0 |
| Prompt-only | Tool-call F1 | 52.5% | 52.5% | +0.0 |
| Prompt-only | Tool-call validity | 100.0% | 100.0% | +0.0 |
| Prompt-only | Normal-tool F1 | 92.9% | 92.9% | +0.0 |
| SFT | Prior-trap rate | 3.8% | 3.2% | -0.6 |
| SFT | Tool-call F1 | 94.0% | 94.5% | +0.5 |
| SFT | Tool-call validity | 98.5% | 98.5% | +0.0 |
| SFT | Normal-tool F1 | 100.0% | 100.0% | +0.0 |
| SC-001 | trap-rate gap vs base | +52.5 pts [+44.2, +60.5] | +53.2 pts [+45.3, +61.0] | +0.6 |
| SC-001 | F1 gap vs base | +42.0 pts [+35.0, +49.0] | +42.5 pts [+35.5, +49.5] | +0.5 |
| SC-001 | trap-rate gap vs prompt-only | +50.6 pts [+42.4, +58.8] | +51.3 pts [+43.1, +59.3] | +0.6 |
| SC-001 | F1 gap vs prompt-only | +41.5 pts [+35.0, +48.5] | +42.0 pts [+35.5, +49.0] | +0.5 |

First-pass SC-001 verdict: met; rerun: met. First-pass git sha: `e12d318`; rerun: `b179f32`.

Runs with identical metric values in both passes: Run A. Where they differ, the largest absolute difference is 0.6 points (SC-001 gaps move with the SFT numbers). Both passes are reported and neither is preferred.

## Independent re-score

The repo's own scripts were run on a copy of the repo (not this repo): for each run the run's `eval_baselines.json` and `sft.json` plus the three per-reply output files (`progress_eval/base.jsonl`, `progress_eval/prompt_only.jsonl`, `progress_sft/tuned.jsonl`) went into `scripts/build_report.py --seed S --hardness H`, then `scripts/check_results.py` on the new final.json. `build_report.py` exits 1 if the outputs do not reproduce the recorded metrics; both runs exited 0 and `check_results.py` printed `OK final` for both. The re-scored files are saved as `<run>/rescored_final.json`.

- Run A: re-scored final.json vs the Kaggle final.json differs only in: `/created_at`. All metric values and gaps are equal.
- Run B: re-scored final.json vs the Kaggle final.json differs only in: `/created_at`. All metric values and gaps are equal.

## Where the artifacts are

- Data (final.json files, per-reply outputs, splits, versions.json): https://huggingface.co/datasets/ignAtom/alien-manual-rerun-data
- SFT adapters: https://huggingface.co/ignAtom/alien-manual-sft-seed0-hardness1 and https://huggingface.co/ignAtom/alien-manual-sft-seed1-hardness0
- Collections: https://huggingface.co/collections/ignAtom/alien-manual-models-6af86497982acc9f5a5dda9f and https://huggingface.co/collections/ignAtom/alien-manual-datasets-6af86498982acc9f5a5ddaab
- Kaggle output archive (`out.zip` in the dataset repo above; local copy `out_hf.zip`): md5 `f4e44107a873715d90c5a448f507e75b`. File-level md5s of the unzipped output are in `MD5SUMS_kaggle_out` (its `committed_results/` lines refer to files that are not in this directory).

## Limits

- One training seed per run: no spread across training seeds.
- 200 held-out examples per run; the CIs reflect that, and the normal-tool metric rests on a few dozen examples (n = 38 in Run A, 42 in Run B).
- The tools are invented, and train and held-out use the same four note patterns (flipped args, inverted op, odd unit, renamed tool); the held-out tools are new but the patterns are not.
- Hardness only changes the wording of the documents, not the calls, requests or ids.
- No external benchmark; no GRPO; no train-at-hardness-0, test-at-hardness-1 run.
