# Milestone 8: local robustness testing

Recorded 12 October 2026. This adds a free CPU test tool around the existing card-support suggestion workflow. It uses the same scope advisory and top-label projection as the pilot, with no new dependency, model training or cloud service.

## Actual outcome

| Observation | Recorded result |
| --- | ---: |
| Assistant-authored synthetic development cases | 60 |
| Agreement with provisional references | 38/60 (63.3%) |
| Human-reviewed reference labels supplied | 0 |
| Outside cases suggested as a supported card category | 2/12 |
| Clarification cases matched | 0/12 |
| Clarification cases suggested as a specific category | 4/12 |
| Runtime errors / unavailable scope | 0 / 0 |
| Sequential successful-request p50 / p95 | 137 / 183 ms |

Model load took 1.69 seconds and the separate warm-up 2.14 seconds. Timing describes one sequential local CPU pass, including routing and workflow projection. It excludes startup from request percentiles and makes no production latency claim.

**These are disagreements with unreviewed development labels, not independent accuracy or evidence of customer savings.** Cases deliberately probe known concerns; one previously used toothbrush example is explicitly tagged. No confidence interval or population generalization is inferred.

The [unchanged actual run report](robustness_v1/report.md) lists all 22 disagreements and per-category metrics. Its [JSON](robustness_v1/summary.json) records dataset/source/pointer hashes, warm-up protocol and separate provisional/reviewer-declared metrics. The [graph](robustness_v1/robustness.png) shows the confusion matrix and overlapping phenomenon groups. Prediction CSVs and reviewer notes are kept local.

## What the failures teach

Typos caused several supported card requests to be flagged outside; negation confused lost-card, fraud and declined-card suggestions. Misleading memory-card and SIM-PIN requests received supported card labels. Of 12 requests with missing context, contradictions or multiple actionable issues, eight were flagged outside and four received a specific category. None received clarification under the declared one-issue workflow rule.

The earlier 96.24% score concerns 4,500 known-intent benchmark requests. This suite instead checks the projected card workflow across supported, unsupported and ambiguous inputs with provisional synthetic references. Different tasks, distributions and label provenance prevent a direct score comparison. A strong benchmark alone does not establish a useful clarification policy.

## Delivery and preservation

- `intentdesk robustness-template` exports messages with blank label/reviewer fields and refuses to overwrite human work. It does not load the model or reveal draft labels in the sheet.
- `intentdesk test-robustness` creates separate timestamped report, CSV, JSON and PNG files. Optional `--labels` binds reviewed rows to the exact cases/hash and scores that subset separately.
- Failures count as wrong suggestions. Review-only contract violations are recorded as errors. Original frozen model selection, threshold checks and human pilot records remain separate.
- Local checks passed: **108 Python tests**, **2 JavaScript tests**; 16 protected files retained identical before/after hashes. This proves the tested software behavior, not model generalization.

Run instructions are in [the robustness guide](../docs/ROBUSTNESS.md). The next step is actual human review of the references, especially ambiguous cases. Any subsequent model change needs a separately versioned protocol and a new representative, untouched evaluation. Do not tune against consumed final/scope checks or describe this exposed development suite as a fresh holdout. The separate human comparison remains incomplete.
