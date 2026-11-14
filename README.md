# Alien Manual

Lexsi hackathon, Track 3 (agentic fine-tuning). Does a small model follow a tool's documentation when the
documentation contradicts what the tool's name suggests?

## What it is

Alien Manual generates invented tools whose docs contradict habit (for example `boost_guxdun`, whose doc says a
positive amount lowers the output), and a scripted oracle that knows the one correct call for every task.
Models are scored on exact calls with 95% bootstrap confidence intervals. We fine-tune Cohere's Tiny Aya
(`CohereLabs/tiny-aya-global`) with SFT (supervised fine-tuning on oracle calls), then GRPO (reinforcement
learning with an exact-match reward), and compare against the untouched base model and a prompt-only baseline.

## Headline result

Evaluation set: 200 held-out tasks from tools never seen in training (162 trap tasks where habit and docs
disagree, 38 normal ones). Seed 0, one run. Values are percentages with 95% CIs, copied from
`results/final.json` (also rendered in `report.md`).

| Metric | Base | Prompt-only | SFT | Tuned (SFT then GRPO) |
|---|---|---|---|---|
| Prior-trap rate (followed habit against the doc; lower is better) | 58.6 [51.9, 66.1] | 58.6 [51.5, 66.2] | 6.8 [3.1, 11.0] | 6.8 [3.2, 10.8] |
| Tool-call F1 (higher is better) | 49.5 [43.0, 56.0] | 49.5 [42.5, 56.5] | 94.5 [91.0, 97.5] | 94.5 [91.0, 97.5] |
| Normal-tool F1 (tools with no twist) | 97.4 [91.1, 100.0] | 94.7 [86.5, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

SC-001 (our success criterion: the tuned model beats both baselines by at least 10 points on prior-trap
rate and on tool-call F1, with the 95% CI above zero) is met on this unseen-tools split. The smallest gap is
45.0 points, tool-call F1 against base (CI 38.5 to 52.0). The lowest CI bound across all four comparisons is
38.0 points (tool-call F1 against prompt-only). Prior-trap gaps are 51.9 points against both baselines.

Normal tools did not get worse: normal-tool F1 went from 97.4 (base) to 100.0 (tuned).

## What was and was not shown

- Shown: after SFT, the model follows the doc on new tool names far more often than the base model or a
  prompt that says "follow the docs". The prompt-only baseline differs from base by one added line and scored
  the same on prior-trap rate and tool-call F1.
- GRPO did not add anything measurable. SFT and the tuned (SFT then GRPO) model have the same headline
  numbers (6.8% prior-trap, 94.5% F1). Credit SFT, not GRPO. Only 69 of 400 GRPO reward groups had any
  difference in reward (`results/grpo.json`), so GRPO had little to learn from.
- "F1" here is likely just exact-match accuracy: the score is 2TP/(2TP+FP+FN) and with exactly one call per
  reply that equals the share of replies with the right call. We could not check that every reply had one
  call, because per-reply outputs are not in this repo. Every overall tool-call F1 value is a multiple of 0.5% (a whole number out of 200), which fits.
- Held-out tools are new tool names, in unseen domains, built from the same four note patterns as the training
  tools: an odd unit, flipped argument order, an inverted operation, a renamed tool. Docs are at hardness 0,
  where the twist is a plain "Note:" line.
- Not shown: a new kind of rule the model never trained on, real-world APIs, or reworded notes after
  training on plain notes (see the rerun section below for what was run).
- The headline run is seed 0, one training run. A second seed (SFT only) is in the rerun section below. The
  CIs resample the 200 examples and do not include training variation.
- The 200 examples come from only 24 tools (3 domains, 8 tools each), so examples from one tool are not
  independent. The CIs resample examples, not tools, so they are probably narrower than they should be.
- For the headline run, per-reply model outputs and the trained adapters were not kept (no output files in
  this repo; the checkpoints were recorded as `local/alien-sft` and `local/alien-grpo` with no revision), so
  those scores cannot be re-derived from this repo alone. The rerun below keeps both.

## Rerun: a second seed, reworded notes, outputs kept

After the headline run we ran SFT again on Kaggle (T4), SFT only (no GRPO), and kept every reply. Same
200-task held-out size, same scorer, unseen tools. Values are percentages. Files are in
`results/rerun_seed1_hardness1/` (its README lists the exact configs and library versions); the trained
adapters and per-reply outputs are public on Hugging Face.

| Run | What changed | Base | Prompt-only | SFT |
|---|---|---|---|---|
| A | seed 0, hardness 1 (the twist is folded into the parameter text, no separate "Note:" line) | trap 88.3, F1 25.5 | trap 86.4, F1 27.0 | trap 1.9, F1 98.5 |
| B | seed 1, hardness 0 (same docs as the headline run, new split) | trap 56.3, F1 52.0 | trap 54.4, F1 52.5 | trap 3.2, F1 94.5 |

- Run A's SFT was trained on hardness-1 docs, so it shows the model learned the reworded format. It does not
  show that a model trained on plain notes survives reworded notes; that was not run.
- Run B is a second seed on the original setup: close to the headline run (6.8% trap, 94.5% F1), which
  supports that the SFT gain is not a one-seed fluke. Two seeds is still few.
- A first pass of these runs was lost when the Kaggle session was recycled. Its printed results were recovered
  from a saved notebook; run A matched exactly, run B's SFT differed slightly (3.8% trap, 94.0% F1 in the
  first pass). The results README lists both.
- All six reply files were re-scored with this repo's scorer and match the table.
- Hugging Face: models collection https://huggingface.co/collections/ignAtom/alien-manual-models-6af86497982acc9f5a5dda9f
  and datasets collection https://huggingface.co/collections/ignAtom/alien-manual-datasets-6af86498982acc9f5a5ddaab
  (LoRA adapters only; the base model's licence is cc-by-nc-4.0, so these are for non-commercial use).

## The first attempt failed, and we changed the split

The project's own decision DEC-5 (in the PRD, a separate private repo) requires saying this plainly.

The first run held out a whole twist family (the model trained without ever seeing one kind of contradiction).
It did not meet SC-001. On 200 held-out tasks: prior-trap rate base 85.2%, prompt-only 95.7%, tuned 100.0%;
the worst gap was -14.8 points (95% CI -21.2 to -8.3), meaning the tuned model followed habit more often than
base. Tool-call F1 was 38.5% (base), 38.5% (prompt-only) and 40.0% (tuned). Files are in
`results/first_run_twist_holdout/`.

We then changed the split to hold out unseen tools instead, so every twist family appears in training. The
split was changed after seeing this result, and the numbers in the headline table come from the new split. The
first-run files are kept, not deleted.

Likely reason for the failure: with seed 0 the held-out family was `flip_args` (argument order swapped
relative to habit). We re-ran the generator (`make_splits(0, holdout="twist")`) and found that none of the 75
training tasks for move-type tools (`transfer`) had swapped arguments, so the model only ever saw the habit
order for that kind of tool. The data gap is checked. That it caused the 100% habit rate is our best
explanation and was not tested separately, for example by retraining.

## Where things are

- `alien/`: generator (tools, twists, tasks), oracle, scoring and bootstrap CIs, splits with a leak check,
  reward, eval and persistence code. Standard library only; torch and friends are imported inside functions.
- `scripts/`: probe, baseline eval, SFT, GRPO (with a rejection-sampling fallback), report builder,
  results checker, and a run-everything shell script.
- `results/`: the JSON from the final run (`base_probe`, `eval_baselines`, `sft`, `grpo`, `final`, `spike`).
  `results/first_run_twist_holdout/` holds the failed first run. `results/rerun_seed1_hardness1/` holds the
  rerun (second seed, reworded notes) with per-reply outputs.
- `report.md`: the generated report for the final run (first line is the SC-001 verdict).
- `RUNBOOK.md`: how to reproduce, step by step.
- `space/`: code for a Gradio demo Space. It has not been deployed.
- `tests/`, `notebooks/spike.ipynb`: unit tests (no GPU needed) and the Colab spike notebook.
- `PROJECT_CONTEXT.md`: a developer map of the code.

To reproduce, follow `RUNBOOK.md`. It is written for a free Colab T4; the final run itself was on a Kaggle
T4, and the JSON files were copied by hand from the notebook output. To rebuild the first (failed) split, see
the held-out set note in `RUNBOOK.md`. Using the model needs a Hugging Face account that has accepted the
licence of the gated model.

## Limits, in short

- Invented tools only. Nothing here shows the model follows docs for real APIs.
- Four contradiction patterns, fixed wording, easiest docs setting. Not a new kind of rule.
- Two seeds (SFT only for the second), about 24 held-out tools per split, intervals probably too narrow.
- The gain is from SFT. GRPO is not shown to help.
- Exact match: two valid calls written differently score 0.
- The first split failed; the second was chosen after that.
