# Milestone 9: improving the card workflow

Recorded 12 October 2026. Two separately trained candidates were compared with the preserved original workflow. All robustness references remain AI-authored and unreviewed by a person. An [AI consistency review](../docs/AI_LABEL_REVIEW.md) was performed; no human annotations or sessions were fabricated.

## Measured results

| Suite | Original | Lexical V2 | Hybrid V3 |
| --- | ---: | ---: | ---: |
| Exposed development, 60 cases | 38/60 | 54/60 | 54/60 |
| V2 check, 48 cases; development for V3 | 28/48 | 30/48 | 42/48 |
| Reserved V3 synthetic check, 32 cases | 21/32 | 23/32 | 29/32 |

Reserved V3 macro-F1: **0.6091 → 0.9048** from original to hybrid. These are provisional synthetic-reference metrics, not independent accuracy. The 32-case suite has 24 supported requests, four outside and four unclear requests, a different balance from the earlier suites. The V3 check is now consumed.

| Reserved V3 slice | Original agreement | Hybrid agreement |
| --- | ---: | ---: |
| Typos | 5/6 | 6/6 |
| Negation | 3/6 | 4/6 |
| Clarification requests | 0/4 | 4/4 |
| Outside requests suggested as supported (lower is better) | 2/4 | 1/4 |

Zero runtime errors were recorded for all three methods on all three suites. Tag groups overlap. Small hand-authored counts do not establish generalization, a production decision rule or customer benefit.

![Recorded comparisons](card_workflow_v3/comparison.png)

The [aggregate JSON](card_workflow_v3/summary.json) is copied unchanged from the actual run. The [first lexical check](card_workflow_v3/lexical_v2_first_check.json) preserves its disappointing 30/48 result and increased outside errors. Those outcomes were inspected before V3 design and became development evidence for V3. No reselection against the original final/scope tests occurred.

## Implementation

V2 learns eight workflow categories with word/character TF-IDF and a bounded negation feature. V3 adds original frozen MPNet embeddings and trains a separate class-balanced linear classifier. Synthetic contrasting, multiple-issue and vague examples add training coverage. This is task-specific classifier training, not full-transformer fine-tuning.

Training used 14,137 deduplicated rows: 11,917 audited fit examples, 909 typo derivatives and 1,311 correlated synthetic variants. V3 reused existing fit embeddings and encoded 2,220 added texts on CPU. The added-feature/fitting phase took about 264 seconds, excluding initial loading/preparation. Original tune, validation, calibration and final requests did not enter model fitting. Exact training/check fingerprints were rejected; same-author synthetic/template similarity still limits independence.

The browser desk at `/card-v3` has separate review storage, model-score disclosure, clarification wording, explicit correction and local handoffs. History is versioned and stale saves are rejected. Original `/` and `/pilot` experiment sources stay preserved. Instructions and feature mathematics are in [the update guide](../docs/CARD_WORKFLOW_UPDATE.md).

## Remaining errors

On the reserved V3 check:

- `c3-fraud-3`: fraud → clarification; benign card context precedes an unauthorized-charge statement.
- `c3-decline-3`: declined card → PIN change; the negated PIN mention still overwhelms the actual rejected payment.
- `c3-out-3`: outside → replacement delivery; an unrelated keyboard replacement confuses the card-delivery category.

The exposed 60 cases retain two negation errors and four incorrectly specific clarification suggestions. The 48-case development set retains unrelated replacement delivery, a negated flight/card comparison, forgotten-PIN taxonomy and some multi-issue/vague errors. A later live request combining a cracked card and an unauthorized purchase was suggested as fraud, confirming multiple-issue handling remains incomplete. No model was changed after the V3 check to hide these failures.

## Verification

Local checks passed **130 Python tests**, **2 JavaScript tests** and new JavaScript syntax checking. New tests cover fit-only training, exact overlap rejection, deterministic augmentation, identity checks, feature weighting, long-input refusal without prefix classification, one-shot comparisons, separate review storage, local boundaries and stale saves. One existing Starlette/httpx deprecation warning remains.

All 16 protected original source/lock/experiment/report files retain identical preservation hashes. No human comparison or human label was created. Live HTTP/browser/package checks are recorded in [verification](../docs/VERIFICATION.md); software checks do not establish production readiness.
