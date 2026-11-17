# Proxy check: official baseline on external proxies vs the published target score (2026-11-17)

Comparison only; nothing is selected or tuned on it. Published numbers: EVALUATION.md (full match,
official metric). External: `scripts/baseline_reference.py` + `scripts/eval_external.py`, nominal
profile, seeds 0+1, label views τ = 0 / 0.5 / 1.0 s (family = mean over the views). DFL = the 5
non-holdout matches; Metrica = the 3 sample games, degraded, scored on provider event-chain labels.

| window | target spellF1 | target macroF1 | DFL family | DFL macroF1 | DFL harsh | Metrica family | Metrica macroF1 | Metrica harsh |
|---|---|---|---|---|---|---|---|---|
| smooth25 | 0.1996 | 0.6752 | 0.1609 | 0.411 | 0.1246 | 0.1760 | 0.756 | 0.1288 |
| smooth50 | **0.2433** | 0.6724 | **0.2165** | 0.380 | 0.1781 | 0.2676 | 0.773 | 0.2107 |
| smooth125 | 0.2242 | 0.6356 | 0.1941 | 0.361 | 0.1877 | **0.2809** | 0.709 | 0.2421 |

Reading:
- **Ordering.** DFL nominal reproduces the target ordering (50 > 125 > 25), about 0.03–0.04 below in
  absolute terms. Degraded Metrica ranks 125 > 50; it is easier and rewards long smoothing.
- **macroF1 separates the proxies.**
  - Target 0.64–0.68; DFL 0.36–0.41; Metrica 0.71–0.77. The baseline predicts D only when the
    ball is missing for the whole window, so macroF1 here mostly measures how often the tracking
    loses the ball in stoppages.
  - DFL alive-dead F1 is 0.68–0.70; Metrica's is 0.88–0.92, because the provider's tracking
    lacks the ball on 98% of dead frames.
  - The target's high macroF1 suggests that its tracking also loses the ball in stoppages more
    strongly than DFL nominal assumes (dead/alive ratio drawn from 0.8–2×).
  - This is an inference from published aggregate scores, not from GT. Whether to widen the
    nominal ratio range is the captain's call.
