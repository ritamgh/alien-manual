SC-001 met: the tuned model (grpo) beats base and prompt-only by at least 10 points on both prior-trap rate and tool-call F1; the smallest gap is 45.0 points (Tool-call F1 vs base, 95% CI 38.5 to 52.0).

## The four metrics

Each cell is the value with its 95% bootstrap interval, on 200 held-out examples (162 trap examples, 38 normal ones). Held-out means the tools never appeared in training.

| Metric | Base | Prompt-only | SFT | Tuned |
|---|---|---|---|---|
| Prior-trap rate | 58.6% [51.9%, 66.1%] | 58.6% [51.5%, 66.2%] | 6.8% [3.1%, 11.0%] | 6.8% [3.2%, 10.8%] |
| Tool-call F1 | 49.5% [43.0%, 56.0%] | 49.5% [42.5%, 56.5%] | 94.5% [91.0%, 97.5%] | 94.5% [91.0%, 97.5%] |
| Tool-call validity | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| Normal-tool F1 | 97.4% [91.1%, 100.0%] | 94.7% [86.5%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |

- Prior-trap rate: how often the model followed habit against the tool documentation on trap examples. Lower is better.
- Tool-call F1: how often the model produced exactly the right call. Higher is better.
- Tool-call validity: share of answers that are a well-formed call to a known tool.
- Normal-tool F1: tool-call F1 on tools with no twist.

## The SC-001 gaps

SC-001 asks for a gain of at least 10 points over both baselines on both prior-trap rate and tool-call F1, with the 95% interval above zero. Gaps are improvements, in points.

| Comparison | Metric | Gap (points) | 95% CI (points) | At least 10 points, CI above 0 |
|---|---|---|---|---|
| vs base | Prior-trap rate | 51.9 | 44.6 to 59.6 | yes |
| vs base | Tool-call F1 | 45.0 | 38.5 to 52.0 | yes |
| vs prompt-only | Prior-trap rate | 51.9 | 44.2 to 59.6 | yes |
| vs prompt-only | Tool-call F1 | 45.0 | 38.0 to 52.0 | yes |

A gap whose interval includes zero is not evidence of a difference between the models.

## What ran

Path that ran: grpo. GRPO (reinforcement learning that samples several answers per prompt and rewards the exact-match ones) trained the tuned model after SFT. The reward differed within 69 of 400 groups of sampled answers. A reward that differs inside a group is what gives GRPO something to learn from.

## A trap example

One example from the 100-example development split, answered by the base model (results/base_probe.json; task dev-0002). It is not one of the held-out examples.

- Request: Lower the output of cell C-58 by 37.
- What the tool documentation says: Changes the output of a cell by a signed amount. Note: despite its name, a positive `amount` lowers the output and a negative `amount` raises it. Parameters: - entry (string): The cell to change. - amount (integer): Signed change to the output.
- Call that follows the documentation: `{"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": 37}}`
- Call that follows habit instead: `{"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": -37}}`
- What the base model answered:

> {"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": -37}}

## Did normal tools get worse?

E-3 (forgetting): normal-tool F1 did not drop by more than 5 points versus base (base 97.4%, tuned 100.0%).

## Caveats

- Only 200 held-out examples (n=200); the intervals show how much that sample alone could move the numbers.
- One run on a single seed (seed 0): the intervals do not include run-to-run training variation.
- The reward and the score are exact-match: two valid calls with different argument forms score 0.
- Only invented tools were tested; nothing here says the model follows documentation for real-world APIs.
