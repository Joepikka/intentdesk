# Milestone 5: frozen final classifier evaluation

Completed 10 October 2026. The final CLINC partition is now **consumed**. Before reading final examples, the primary model, temperatures, thresholds, model/data/source hashes and dependency lock were frozen in `artifacts/final_experiment.json`. Final results cannot select a model, change thresholds or enable routing. Re-running the identical completed experiment reuses its report.

## Primary result

The previously selected **MPNet + linear classifier with temperature scaling** achieved **96.24% accuracy** and **0.9624 macro-F1** on **4,500 known-intent test requests**. Log loss: **0.1435**; Brier: **0.0587**; ECE over 15 equal-width bins: **0.0074**. The formatting-group bootstrap accuracy interval is [95.71%, 96.80%], conditional on this benchmark rather than future traffic.

These are supervised benchmark measurements. The 109M-parameter pretrained encoder remained frozen; only downstream classifiers/heads were trained locally. All 5,500 final examples were within the serving token limit; none were truncated. Final feature encoding took 157.8 seconds on CPU.

## Descriptive fixed-model comparison

| Model | Known accuracy | Macro-F1 | Raw log loss |
| --- | ---: | ---: | ---: |
| B1_lexical | 91.96% | 0.9186 | 0.5232 |
| B2_MPNet_linear | 96.24% | 0.9624 | 0.2929 |
| N1_small | 96.20% | 0.9620 | 0.1486 |
| N2_large | 94.27% | 0.9424 | 0.2221 |

The primary model was selected before this comparison. The larger head's weaker result remains visible. No model was retrained or replaced after observing these scores.

## Unsupported requests and active behavior

On **1,000 unsupported final requests**, the frozen proposed thresholds would accept **135 (13.50%)**. This is a descriptive offline measurement of the already-failed development policy, not a new threshold-selection or activation gate. Known-intent accuracy does not solve OOS detection.

**The active app still requires review for every request: automatic coverage 0%, actions executed 0.** A final score cannot override the earlier failed development gate. Future OOS research needs fresh, reviewed development and evaluation data; the consumed final set must not be retuned into a validation set while still claiming independent final performance.

![Frozen classifier results](figures/milestone_05_final.png)

This is crowdsourced English assistant-request data, not production customer traffic. Fingerprint grouping removes formatting overlap but cannot prove the absence of semantic or pretraining overlap. The bootstrap is an uncertainty description of this sample, not a real-world reliability guarantee. No generated-answer or business-impact evaluation was performed.

[Machine-readable final evidence](final_evaluation.json). Historical Milestones 1–4 record the state at those stages; their earlier “test not evaluated” statements are superseded by this milestone.
