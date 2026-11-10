#!/usr/bin/env bash
# One command for the whole pipeline on a Colab T4, safe to start in the background so a
# dropped browser page cannot kill it (RUNBOOK section 3 lists the same steps one by one).
#
#   nohup bash scripts/run_all_colab.sh /content/work > /content/run.log 2>&1 &
#   tail -n 20 /content/run.log        # poll from a short cell; the last line is ALL DONE when finished
#
# Needs HF_TOKEN in the environment. Nothing is uploaded (ALIEN_NO_UPLOAD), adapters stay in the work dir.
set -euo pipefail

WORK="${1:-/content/work}"
export ALIEN_NO_UPLOAD=1
cd "$(dirname "$0")/.."

python scripts/probe_base.py --seed 0
python scripts/check_results.py results/base_probe.json
python scripts/run_eval.py --which base,prompt_only --seed 0 \
  --out results/eval_baselines.json --progress-dir "$WORK/progress_eval"
python scripts/check_results.py results/eval_baselines.json
python scripts/train_sft.py --hub-repo local/alien-sft --seed 0 --work-dir "$WORK"
python scripts/check_results.py results/sft.json
python scripts/train_grpo.py --hub-repo local/alien-grpo --seed 0 --work-dir "$WORK" --auto-fallback
python scripts/check_results.py results/grpo.json
python scripts/build_report.py --seed 0 --hardness 0 \
  --outputs base="$WORK/progress_eval/base.jsonl" \
  --outputs prompt_only="$WORK/progress_eval/prompt_only.jsonl" \
  --outputs sft="$WORK/progress_sft/tuned.jsonl" \
  --outputs tuned="$WORK/grpo/progress_tuned/tuned.jsonl"
python scripts/check_results.py results/final.json
# Print every result file into the log, so the numbers survive even if the Colab runtime is recycled
# (a notebook cell that shows this log keeps the text in the saved notebook).
for f in results/base_probe.json results/eval_baselines.json results/sft.json results/grpo.json results/final.json report.md; do
  echo "=== $f ==="
  cat "$f"
done
echo "ALL DONE"
