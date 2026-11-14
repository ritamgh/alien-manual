SC-001 met: the tuned model (sft-only) beats base and prompt-only by at least 10 points on both prior-trap rate and tool-call F1; the smallest gap is 42.0 points (Tool-call F1 vs prompt-only, 95% CI 35.5 to 49.0).

## The four metrics

Each cell is the value with its 95% bootstrap interval, on 200 held-out examples (158 trap examples, 42 normal ones). Held-out means the tools never appeared in training.

| Metric | Base | Prompt-only | SFT | Tuned |
|---|---|---|---|---|
| Prior-trap rate | 56.3% [48.7%, 64.1%] | 54.4% [46.5%, 62.6%] | 3.2% [0.6%, 6.0%] | 3.2% [0.6%, 6.0%] |
| Tool-call F1 | 52.0% [44.5%, 59.0%] | 52.5% [45.0%, 59.0%] | 94.5% [91.0%, 97.5%] | 94.5% [91.0%, 97.5%] |
| Tool-call validity | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 98.5% [97.0%, 100.0%] | 98.5% [97.0%, 100.0%] |
| Normal-tool F1 | 95.2% [87.8%, 100.0%] | 92.9% [83.9%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |

- Prior-trap rate: how often the model followed habit against the tool documentation on trap examples. Lower is better.
- Tool-call F1: how often the model produced exactly the right call. Higher is better.
- Tool-call validity: share of answers that are a well-formed call to a known tool.
- Normal-tool F1: tool-call F1 on tools with no twist.

## The SC-001 gaps

SC-001 asks for a gain of at least 10 points over both baselines on both prior-trap rate and tool-call F1, with the 95% interval above zero. Gaps are improvements, in points.

| Comparison | Metric | Gap (points) | 95% CI (points) | At least 10 points, CI above 0 |
|---|---|---|---|---|
| vs base | Prior-trap rate | 53.2 | 45.3 to 61.0 | yes |
| vs base | Tool-call F1 | 42.5 | 35.5 to 49.5 | yes |
| vs prompt-only | Prior-trap rate | 51.3 | 43.1 to 59.3 | yes |
| vs prompt-only | Tool-call F1 | 42.0 | 35.5 to 49.0 | yes |

A gap whose interval includes zero is not evidence of a difference between the models.

## What ran

Path that ran: sft-only. No GRPO or rejection-sampling result was provided, so the tuned model is the SFT model (supervised fine-tuning on correct example calls) and the SFT and Tuned columns are identical.

## A trap example

The probe file was not found, so no trap example is quoted. `scripts/probe_base.py` writes results/base_probe.json.

## Did normal tools get worse?

E-3 (forgetting): normal-tool F1 did not drop by more than 5 points versus base (base 95.2%, tuned 100.0%).

## Caveats

- Only 200 held-out examples (n=200); the intervals show how much that sample alone could move the numbers.
- One run on a single seed (seed 1): the intervals do not include run-to-run training variation.
- The reward and the score are exact-match: two valid calls with different argument forms score 0.
- Only invented tools were tested; nothing here says the model follows documentation for real-world APIs.
