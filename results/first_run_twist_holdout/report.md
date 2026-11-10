SC-001 NOT met: the tuned model (grpo) does not clear the bar of 10 points with a 95% interval above zero on every comparison; the weakest is Prior-trap rate vs base, a gap of -14.8 points (95% CI -21.2 to -8.3).

Note on the sign: a gap is an improvement in points, so -14.8 means the tuned model followed habit 14.8 points MORE often than the base model, not less.

## The four metrics

Each cell is the value with its 95% bootstrap interval, on 200 held-out examples (115 trap examples, 85 normal ones). Held-out means the tools and the twist family never appeared in training (this first run used `make_splits(0, holdout="twist")`; the default is now unseen tools with every twist family in training).

| Metric | Base | Prompt-only | SFT | Tuned |
|---|---|---|---|---|
| Prior-trap rate | 85.2% [78.8%, 91.7%] | 95.7% [91.5%, 99.1%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| Tool-call F1 | 38.5% [31.5%, 45.5%] | 38.5% [31.5%, 45.5%] | 36.0% [29.5%, 43.0%] | 40.0% [32.5%, 47.0%] |
| Tool-call validity | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| Normal-tool F1 | 90.6% [83.9%, 96.4%] | 90.6% [83.9%, 96.4%] | 84.7% [76.7%, 92.0%] | 94.1% [88.5%, 98.8%] |

- Prior-trap rate: how often the model followed habit against the tool documentation on trap examples. Lower is better.
- Tool-call F1: how often the model produced exactly the right call. Higher is better.
- Tool-call validity: share of answers that are a well-formed call to a known tool.
- Normal-tool F1: tool-call F1 on tools with no twist.

## The SC-001 gaps

SC-001 asks for a gain of at least 10 points over both baselines on both prior-trap rate and tool-call F1, with the 95% interval above zero. Gaps are improvements, in points.

| Comparison | Metric | Gap (points) | 95% CI (points) | At least 10 points, CI above 0 |
|---|---|---|---|---|
| vs base | Prior-trap rate | -14.8 | -21.2 to -8.3 | NO |
| vs base | Tool-call F1 | 1.5 | -2.0 to 5.5 | NO |
| vs prompt-only | Prior-trap rate | -4.3 | -8.5 to -0.9 | NO |
| vs prompt-only | Tool-call F1 | 1.5 | -2.0 to 5.5 | NO |

A gap whose interval includes zero is not evidence of a difference between the models.

## What ran

Path that ran: grpo. GRPO (reinforcement learning that samples several answers per prompt and rewards the exact-match ones) trained the tuned model after SFT. The reward differed within 102 of 400 groups of sampled answers. A reward that differs inside a group is what gives GRPO something to learn from.

## A trap example

One example from the 100-example development split, answered by the base model (results/base_probe.json; task dev-0002). It is not one of the held-out examples.

- Request: Lower the output of cell C-58 by 37.
- What the tool documentation says: Changes the output of a cell by a signed amount. Note: despite its name, a positive `amount` lowers the output and a negative `amount` raises it. Parameters: - entry (string): The cell to change. - amount (integer): Signed change to the output.
- Call that follows the documentation: `{"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": 37}}`
- Call that follows habit instead: `{"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": -37}}`
- What the base model answered:

> {"name": "boost_guxdun", "arguments": {"entry": "C-58", "amount": -37}}

## Did normal tools get worse?

E-3 (forgetting): normal-tool F1 did not drop by more than 5 points versus base (base 90.6%, tuned 94.1%).

## Caveats

- Only 200 held-out examples (n=200); the intervals show how much that sample alone could move the numbers.
- One run on a single seed (seed 0): the intervals do not include run-to-run training variation.
- The reward and the score are exact-match: two valid calls with different argument forms score 0.
- Only invented tools were tested; nothing here says the model follows documentation for real-world APIs.
- Training used 600 tasks over 48 tools; SFT loss reached 0.006. The held-out set (`holdout="twist"`, the first run; the default is now unseen tools only) uses 24 different tools from a twist family absent from training (see the analysis below), so this run tests generalising to an unseen kind of contradiction, not only unseen tools.
