# Alien Manual runbook

Ordered steps to run the pipeline end to end on a free Colab T4 and produce `report.md`.
Requirement ids (FR-, SC-, E-, R-, B-) refer to the project PRD (kept in a separate private repo). Every
`python scripts/...` line below was checked against the script's argparse flags (see
"Verified here" at the end). Nothing in steps 2 to 7 has been run on a GPU yet.

## 0. What this is and what you need

- Alien Manual generates invented tools whose docs contradict common habit, then fine-tunes
  Tiny Aya (SFT, then GRPO or a rejection-sampling fallback) to follow the doc.
- It measures four metrics for base, prompt-only and tuned models on 200 held-out examples,
  and states honestly whether SC-001 (10+ point gains, 95% CI above zero) is met.
- Every script writes a results JSON; `python scripts/check_results.py <file>` must print `OK <kind>`.
- Needs: Colab free T4; a Hugging Face account that has accepted the licence of the gated model
  `CohereLabs/tiny-aya-global` (id and gating **unverified until step 1**); Google Drive for checkpoints.
- The HF token lives in Colab Secrets (key icon, secret name `HF_TOKEN`, notebook access on) and
  reaches scripts only as the `HF_TOKEN` environment variable. Never paste it into a cell, a
  cell output or a file; no script prints it (B-3).

## 1. Laptop sanity (no GPU)

```
python3.12 -m venv .venv
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q          # expect: 631 passed
python scripts/check_results.py        # no argument: prints usage, exit 2
python scripts/train_sft.py --dry-run  # prints the plan, imports no heavy library, exit 0
```

- `check_results.py` is not argparse; `--help` prints the usage line and exits 0.
- Exit 0 prints `OK <kind>`; exit 1 prints one problem per line, each starting with the field path.
  It scans the decoded document for token shapes first; a hit prints only an error naming the
  field path (`checkpoint.revision: contains what looks like a secret`), never the value.
- Run the tests before any Colab run (B-1). The 631 count is for the commit this file was written on.

## 2. Colab session setup (repeat after every new or reset session)

Push the merged branch to GitHub yourself first (nothing here pushes). In a Colab notebook with
T4 GPU selected (Runtime > Change runtime type), one cell per block:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```
!git clone --depth 1 --branch <branch> https://github.com/<user>/<repo>.git /content/repo
%cd /content/repo
!pip install -q "trl==1.7.1" peft datasets
```

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # never print this
```

- The repo must be public or uploaded as a zip to `/content/repo`.
- The pip line is the AS-1 fallback stack from `notebooks/spike.ipynb`; transformers and torch come
  with Colab. After the spike, install the versions recorded in `results/spike.json`.
- Results survive only if copied to Drive. After each step:
  `cp -r results /content/drive/MyDrive/alien_manual/`; in a new session, after cloning: `mkdir -p /content/drive/MyDrive/alien_manual && cp -r /content/drive/MyDrive/alien_manual/results/. results/`.
- Below, `W` = `/content/drive/MyDrive/alien_manual`. In Colab prefix each command with `!`.

## 3. The steps, in this order

- **Progress dirs.** Resume = rerun the same command with the same flags. Use a fresh
  `--progress-dir` when you change seed, hardness, adapter, revision or `--n`: `run_eval.py` and
  `probe_base.py` exit 2 ("fresh --progress-dir") for another run's file (fingerprint in
  `<name>.jsonl.key`); old files without a `.key` are refused too. The three train scripts
  fingerprint their tuned eval the same way (`StaleProgressError`, exit 2, "fresh --progress-dir")
  when the adapter, checkpoint or base revision changes over an existing progress dir.
- **Final checkpoints.** `ckpt/final` holds `run_config.json` (script, seed, hardness, base
  revision, hyper-parameters, `--sft-adapter`/`--full-sequence`). `train_sft`, `train_grpo` and
  `train_rft` exit 2 ("use a fresh --work-dir") if it differs in any of those, or if it is missing
  (a checkpoint made by earlier code, e.g. already on Drive: retrain into a fresh work dir or
  delete that `final`). Resume = rerun the same command with the same flags in the same `--work-dir`.
- **Hardness.** If step b ends on hardness N, add `--hardness N` to the commands in steps c to f
  (shown at the default 0), the same value everywhere.

### a) T-010 spike notebook FIRST (AS-1, AS-2, AS-3)

- Open `notebooks/spike.ipynb` in Colab (File > Upload notebook). Edit `REPO_URL` and `BRANCH`
  in the checkout cell, add the `HF_TOKEN` secret, then Runtime > Run all. If the install cell
  asks for a restart, restart and run all again from the checkout cell.
- Writes `results/spike.json` and downloads it. Put it in `results/`.
- Check: `python scripts/check_results.py results/spike.json` expects `OK spike`.

| Outcome | Meaning | Next move |
|---|---|---|
| `install_ok` false (AS-1) | pinned stack plus AgentTune does not install together | `!pip install -q "trl==1.7.1" peft datasets`, re-run from the import cell; training uses plain TRL anyway |
| checker prints `NaN loss in fp16` (AS-2) | fp16 LoRA diverges; `write_results` also refuses the file | load the model in 4-bit (QLoRA, `bitsandbytes`) or lower batch/sequence length, re-run. The train scripts have no 4-bit switch: that needs a code change, tell the lead |
| `grpo_step_ok` false (R-3) | GRPO step did not run | skip step e's GRPO and go straight to `train_rft.py` |
| `reward_sees_full_conversation` | see below | read `kwargs_keys` and `completions_0` |

`reward_sees_full_conversation` (AS-3): `alien/reward.py` takes the last assistant message from
`completions` (string, message list or single message dict) and finds the task by the `task_id`
dataset column, else by matching the prompt. So true or false does **not** change the reward
code. What matters: `kwargs_keys` should contain `task_id` (otherwise matching falls back to
prompt text), and `completions_0` shows whether completions arrive as text or message lists.
`train_grpo.py` prints the same keys on its first step (B-1): compare them.

### b) T-002 probe: is there headroom? (E-1, R-1)

```
python scripts/probe_base.py --seed 0
python scripts/check_results.py results/base_probe.json
```

- Writes `results/base_probe.json` (dev split, 100 tasks, plain base prompt; progress in
  `results/progress_probe/`). Rerun the same command after an interruption: it resumes.
- It prints the prior-trap rate with CI, `headroom: ok|low`, and one quoted trap example.
- Headroom "low" (rate under 10%, E-1): rerun with `--hardness 1`, then 2, then 3 (the cap),
  **before** the held-out split is frozen. Each hardness writes its own progress file.
- Hardness changes only the tool docs; task ids, requests and calls are identical. `run_eval.py`, `train_sft.py`, `train_grpo.py`, `train_rft.py`, `probe_base.py` and
  `build_report.py` all take `--hardness N` (0 to 3, default 0; outside that range every one but
  `probe_base.py` exits 2). No splits file is needed. An optional `--splits FILE` (on `run_eval.py`
  and the three train scripts) records its seed and hardness; they must equal `--seed` and
  `--hardness` or the script exits 2. The file is also leak-checked (a held-out task in train, say):
  a failing file exits 2 with "fails the leak check", `--dry-run` included.
- Still low at hardness 3: stop and ask Ritam (moving the core to Gremlin Gym needs his yes, B-6).
  A negative result may be the honest outcome.

### c) T-004 baselines

```
python scripts/run_eval.py --which base,prompt_only --seed 0 --out results/eval_baselines.json --progress-dir /content/drive/MyDrive/alien_manual/progress_eval
python scripts/check_results.py results/eval_baselines.json
```

- Writes `results/eval_baselines.json`, plus `progress_eval/{base,prompt_only}.jsonl` and
  `timing.json` on Drive. Rerun the same command after an interruption: it resumes from the
  progress files and sums GPU time across sessions.
- Look at `gpu_seconds` and `within_budget` (FR-002: 3 models x 200 examples in 10800 s). The
  file covers only the two baselines, so `within_budget: false` (or a value near 10800) means the
  tuned eval cannot fit the 3-hour promise: report it, do not hide it.
- A metric with zero eligible examples aborts with a named error and exit 1 (E-5); nothing is written.
- The base prompt and the prompt-only prompt differ by exactly one line (`FOLLOW_DOCS_LINE` in `alien/oracle.py`).

### d) T-005 SFT (guaranteed result, R-4)

```
python scripts/train_sft.py --smoke
python scripts/train_sft.py --hub-repo <user>/alien-sft --seed 0 --work-dir /content/drive/MyDrive/alien_manual
python scripts/check_results.py results/sft.json
```

- `--smoke` (optional): 8 steps, 16 rows, no upload, no results file, work under `W/smoke`.
- The real run writes checkpoints `W/ckpt/step-N` and `W/ckpt/final` (manifest-listed), uploads the
  adapter to the Hub (3 attempts), evaluates on the same 200 held-out examples, writes
  `results/sft.json` and `W/progress_sft/tuned.jsonl`.
- If the upload failed nothing is deleted; rerun the same command to retry only the upload and
  the eval. `--seed` must equal the one in `eval_baselines.json` (else exit 2).
- Exit 0 ok; 1 upload failed or `sft.json` invalid; 2 bad arguments, a stale progress dir or a
  `final` from another run. A repo name error or a bad `HF_TOKEN` shows as exit 1 after the retries.

### e) T-007 GRPO, with the rejection-sampling fallback

```
python scripts/train_grpo.py --dry-run
python scripts/train_grpo.py --hub-repo <user>/alien-grpo --seed 0 --work-dir /content/drive/MyDrive/alien_manual
python scripts/check_results.py results/grpo.json
```

- Needs `results/eval_baselines.json` and `results/sft.json` with the same seed, and the SFT
  adapter at `W/ckpt/final` (or `--sft-adapter`); it continues that adapter. Optional smoke:
  `python scripts/train_grpo.py --smoke --sft-adapter W/smoke/ckpt/step-8`.
- Writes `results/grpo.json`, `W/grpo/ckpt/`, `W/grpo/progress_tuned/tuned.jsonl`. Resume = rerun.

| Exit | Meaning | Next move |
|---|---|---|
| 0 | ok | run the checker, go to step f |
| 1 | upload failed, results invalid, or a metric had no eligible examples | rerun the same command (retries upload and eval); read the printed message |
| 2 | bad arguments; missing, invalid or other-seed `eval_baselines.json` / `sft.json`; missing SFT adapter; stale progress dir or `final` from another run | fix the input named in the message; nothing was trained |
| 3 | reward dead (every group identical, E-2); checkpoints kept; no `grpo.json` written | run `train_rft.py` below, or rerun with `--auto-fallback` (it forwards `--hardness`) |

```
python scripts/train_rft.py --hub-repo <user>/alien-rft --seed 0 --work-dir /content/drive/MyDrive/alien_manual
python scripts/check_results.py results/rft.json
```

- Same inputs and flags; samples 8 completions per task, keeps at most 2 exact-match ones, trains
  one epoch. Writes `results/rft.json`, `W/rft/samples.jsonl` (resumable) and
  `W/rft/progress_tuned/tuned.jsonl`. Extra exit **4**: no sampled completion was correct, no
  `rft.json`; SFT stays the final model (the report then says `sft`).
- An SFT model that is already always right also gives all-identical groups, so it also exits 3.
- Keep only the results file of the model you want reported: `build_report.py` reads both
  `grpo.json` and `rft.json` if present and picks the better tuned `tool_call_f1`.

### f) T-008 report

```
python scripts/build_report.py --seed 0 --hardness 0 \
  --outputs base=/content/drive/MyDrive/alien_manual/progress_eval/base.jsonl \
  --outputs prompt_only=/content/drive/MyDrive/alien_manual/progress_eval/prompt_only.jsonl \
  --outputs sft=/content/drive/MyDrive/alien_manual/progress_sft/tuned.jsonl \
  --outputs tuned=/content/drive/MyDrive/alien_manual/grpo/progress_tuned/tuned.jsonl
python scripts/check_results.py results/final.json
```

- Tuned outputs: `W/grpo/progress_tuned/tuned.jsonl`, or `W/rft/progress_tuned/tuned.jsonl` for
  an rft run. With neither `grpo.json` nor `rft.json`, omit the `tuned=` line (tuned = sft).
  The base and prompt-only defaults are `results/progress/*.jsonl`; the lines above override them
  because step c used a Drive progress dir.
- Set `--hardness` to the value step b ended on and the eval and training used: it selects the
  held-out split and rebuilds the quoted trap doc.
- Needs `results/eval_baselines.json`, `sft.json`, `grpo.json` and/or `rft.json`; `base_probe.json` optional.
- Writes `results/final.json` and `report.md`; the first line of `report.md` is the SC-001 verdict.
- Stdlib only: it also runs on the laptop with the Drive files downloaded.
- Exit 1 (nothing written): the outputs do not reproduce the recorded metrics (wrong file, wrong
  seed, or the other RL run's outputs). Exit 2: missing or invalid input, a `--seed` different
  from the baselines', `--hardness` outside 0 to 3, or a probe whose trap example is not in the
  dev split for that seed and hardness. Then open `report.md`.

### g) T-011 demo Space

1. Locally, with gradio installed in a throwaway venv, run `DEMO_FAKE=oracle python space/app.py`
   once and open the page BEFORE deploying. `build_demo`'s UI code has never been executed.
2. Create a Gradio Space. Copy `alien/` and everything in `space/` (`app.py`, `requirements.txt`,
   `README.md`) into the Space repo, with `alien/` next to `app.py`.
3. Space secrets: `ADAPTER_REPO` (hub id of the tuned adapter, e.g. `<user>/alien-grpo`),
   `HF_TOKEN` (only if the adapter or base is private), optional `MODEL_ID`; optional
   `DEMO_FAKE=oracle|habit` previews the page without a model. With neither `ADAPTER_REPO` nor
   `DEMO_FAKE` the page shows which setting is missing.

## 4. Things nobody could test here

| Item | File | Look at this if it breaks |
|---|---|---|
| Model id `CohereLabs/tiny-aya-global`, licence gating | `alien/eval.py:54` | 401/403 on download: accept the licence on the model page with the same account as the token |
| Spike notebook never run; agenttune + `trl==1.7.1` resolving together; AgentTune route not run | `notebooks/spike.ipynb` | install cell; use the plain-TRL fallback in step a |
| `dtype=` (transformers 5) in training; `torch_dtype=` in eval, probe and Space | `scripts/train_*.py`, `scripts/run_eval.py:74`, `space/app.py:152` | TypeError or deprecation warning on model load: swap the kwarg in that file |
| Chat template behaviour | `scripts/run_eval.py:85`, `scripts/train_rft.py:200` | empty or odd outputs in step b/c: print one prompt and output |
| Resolved revision (`_commit_hash`) | `scripts/run_eval.py:76`, `train_*.py` | `model_revision` shows `main` instead of a hash |
| Completion-only loss from prompt/completion rows; `--full-sequence` is the fallback | `scripts/train_sft.py:280`, `train_rft.py:238` | SFT loss looks odd or the model parrots prompts: rerun with `--full-sequence` |
| `max_length` kwarg name (older TRL: `max_seq_length`), `warmup_ratio` warning | `train_sft.py:287,293`, `train_rft.py:245,251` | TypeError building `SFTConfig` |
| Resume restores LoRA weights, optimizer and step | `train_sft.py:326`, `train_grpo.py:415`, `train_rft.py:295` | after an interruption the step counter restarts from 0 |
| `CommitInfo.oid` is the commit hash | `train_sft.py:241`, `train_grpo.py:286` | `checkpoint.revision` is null |
| Pre-wrapped `PeftModel(is_trainable=True)` accepted by `GRPOTrainer`/`SFTTrainer`; gradient checkpointing with it; saved adapter loads on the base alone | `train_grpo.py:340`, `train_rft.py:235` | error at trainer construction, or `run_eval --which tuned --adapter` cannot load the adapter |
| `GRPOConfig.max_prompt_length` may not exist | `train_grpo.py:361` | included only if the field exists; check the plan output |
| `task_id` column reaches the reward function (B-1 print) | `train_grpo.py` (`announce_first_call`), `alien/reward.py` | `unmatched` counter high, rewards all 0, exit 3 |
| Space UI and `_hf_generate` never executed | `space/app.py` | step g.1 |
| The "3 models x 200 examples in 3 hours on a free T4" claim | `alien/eval.py` (`BUDGET_GPU_SECONDS`) | `within_budget` in step c |

## 5. Honesty box

- The PRD (A-1) calls AgentTune a Cohere Labs library. It is Lexsi-Labs' (`github.com/Lexsi-Labs/AgentTune`,
  PyPI `agenttune`), as `notebooks/spike.ipynb` notes. The pipeline does not import it: GRPO is plain TRL
  `GRPOTrainer`. The PRD's "AuditKIT" is likewise not used: scoring is the scripted `alien/scoring.py`.
- SC-001 is "met" only if all four gaps (prior-trap rate and `tool_call_f1`, each against base and
  against prompt-only) are at least 10 points (0.10) **and** each 95% bootstrap CI lower bound is above
  zero. Anything less is reported as "not met". Do not edit the verdict.
- Registration, submission, the Discord and other hackathon-side steps are Ritam's alone (B-5);
  changing the held-out split after seeing results needs his yes (B-6). The report's quoted trap
  example comes from the dev split, and says so.
- Scoring reads the answer from the reply's ``` fence: if one exists, only the fenced call(s) count and
  calls outside it are ignored (a bare habit call beside a fenced answer is not counted). A float
  such as 42.000000000000000000001 reads as 42.
- Hardness is not recorded in `base_probe.json` or `eval_baselines.json` (known, deferred): pass the
  same `--hardness` everywhere. A hardness-1 probe rebuilt at 0 builds silently; nothing warns.

## Verified here

Run in the docs worktree with the pytest-only venv: `python -m pytest -q` (631 passed); `--help`
for every script above (each shows `--hardness`, and `--splits` where noted); `--dry-run` for the
three train scripts (exit 0); `--hardness 4` on all but `probe_base` (exit 2); `check_results.py`
with no argument (exit 2) and `--help` (exit 0). The `final` checkpoint refusal, the stale-progress
exits and the leak check were read from code and covered by tests. `--smoke`, `--auto-fallback`,
`--outputs` and exit codes were read from argparse and docstrings; no training, probe or download was run.
