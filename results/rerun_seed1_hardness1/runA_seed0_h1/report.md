SC-001 met: the tuned model (sft-only) beats base and prompt-only by at least 10 points on both prior-trap rate and tool-call F1; the smallest gap is 71.5 points (Tool-call F1 vs prompt-only, 95% CI 65.0 to 77.5).

## The four metrics

Each cell is the value with its 95% bootstrap interval, on 200 held-out examples (162 trap examples, 38 normal ones). Held-out means the tools never appeared in training.

| Metric | Base | Prompt-only | SFT | Tuned |
|---|---|---|---|---|
| Prior-trap rate | 88.3% [83.4%, 92.8%] | 86.4% [81.0%, 91.4%] | 1.9% [0.0%, 4.1%] | 1.9% [0.0%, 4.1%] |
| Tool-call F1 | 25.5% [20.0%, 31.5%] | 27.0% [21.5%, 33.5%] | 98.5% [96.5%, 100.0%] | 98.5% [96.5%, 100.0%] |
| Tool-call validity | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| Normal-tool F1 | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |

- Prior-trap rate: how often the model followed habit against the tool documentation on trap examples. Lower is better.
- Tool-call F1: how often the model produced exactly the right call. Higher is better.
- Tool-call validity: share of answers that are a well-formed call to a known tool.
- Normal-tool F1: tool-call F1 on tools with no twist.

## The SC-001 gaps

SC-001 asks for a gain of at least 10 points over both baselines on both prior-trap rate and tool-call F1, with the 95% interval above zero. Gaps are improvements, in points.

| Comparison | Metric | Gap (points) | 95% CI (points) | At least 10 points, CI above 0 |
|---|---|---|---|---|
| vs base | Prior-trap rate | 86.4 | 81.5 to 91.4 | yes |
| vs base | Tool-call F1 | 73.0 | 67.0 to 79.0 | yes |
| vs prompt-only | Prior-trap rate | 84.6 | 78.9 to 89.9 | yes |
| vs prompt-only | Tool-call F1 | 71.5 | 65.0 to 77.5 | yes |

A gap whose interval includes zero is not evidence of a difference between the models.

## What ran

Path that ran: sft-only. No GRPO or rejection-sampling result was provided, so the tuned model is the SFT model (supervised fine-tuning on correct example calls) and the SFT and Tuned columns are identical.

## A trap example

The probe file was not found, so no trap example is quoted. `scripts/probe_base.py` writes results/base_probe.json.

## Did normal tools get worse?

E-3 (forgetting): normal-tool F1 did not drop by more than 5 points versus base (base 100.0%, tuned 100.0%).

## Caveats

- Only 200 held-out examples (n=200); the intervals show how much that sample alone could move the numbers.
- One run on a single seed (seed 0): the intervals do not include run-to-run training variation.
- The reward and the score are exact-match: two valid calls with different argument forms score 0.
- Only invented tools were tested; nothing here says the model follows documentation for real-world APIs.
