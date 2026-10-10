# Milestone 1: an actual trained lexical classifier

Recorded 10 October 2026, client date. Run timestamps and provenance use UTC as reported by the execution environment.

Run ID: `20261009T200903346981Z`.

## Completed

- Python 3.12.14 environment, installed project and exact `uv.lock`.
- Pinned CLINC150 source revision, checksum, upstream attribution/license, data validation and split manifest.
- Conservative formatting-variant overlap policy, resulting in 95 excluded development records.
- Majority baseline and word/character TF-IDF + logistic regression, one fixed initial configuration.
- Saved classifier, checksum, environment/source hashes, reload parity and development error records.
- CLI prediction and three graphs with split labels.
- 19 focused tests passed; a separate toy cross-entropy/SGD exercise passed its numerical gradient check.

## Actual partition counts

| Use | Examples |
| --- | ---: |
| Fit | 11,917 |
| Tune | 1,496 |
| Calibrate, unused | 1,496 |
| Official validation after overlap exclusions, unused | 3,096 |
| Official test, unchanged and not evaluated | 5,500 |
| OOS training, unused | 100 |
| Excluded | 95 |

Exclusions: 42 duplicate training variants, 49 training records overlapping protected holdouts, and 4 validation records overlapping final test fingerprints. This is a mechanical formatting-variant audit, not proof all semantic near-duplicates are absent.

## Development metrics

| Model | Accuracy | Macro-F1 | Log loss | Multiclass Brier |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.006684 | 0.000089 | 35.802720 | 1.986631 |
| Word/character TF-IDF + logistic regression | 0.949866 | 0.949413 | 0.398972 | 0.130520 |

These results are on **1,496 tune examples**. The model made 75 errors; the generated error records have not yet been manually reviewed. No final-test score, calibration success, OOS rejection rate, automated-routing success or business impact is claimed.

Training took 23.49 seconds on this environment, converging in 25 solver iterations with 28,973 sparse features. Reloaded probabilities matched the in-memory probabilities exactly for the 20 checked requests. The model file is 28,153,446 bytes. Fifty sequential warm single-request measurements had a 1.83 ms p95 for full TF-IDF plus classifier inference with two BLAS threads. That excludes cold loading, API overhead and concurrent load; it is not an end-to-end service latency claim. Peak RAM was not measured.

## Demonstrated behavior and limits

`I lost my credit card` suggested `report_lost_card` with raw score 0.8064. The score is **uncalibrated**, not a verified 80.64% chance of correctness.

`I forgot my password` suggested an unrelated known intent at a low score. Password reset is absent from the supported intent list. A separate unsupported astronomy question also received a known intent. Neither input caused an external action; every prediction remains `review_required`. These examples were typed for a smoke check, not sampled from the locked test, and are not an OOS benchmark.

## Graphs and reproducible evidence

- [Full run report](experiments/20261009T200903346981Z/summary.md)
- [Model comparison](experiments/20261009T200903346981Z/baseline_comparison.png)
- [Fit-request length histogram](experiments/20261009T200903346981Z/request_lengths.png)
- [Common tuning confusion pairs](experiments/20261009T200903346981Z/confusion_pairs.png)
- [Machine-readable run metadata](experiments/20261009T200903346981Z/run.json)
- [Data audit](data_audit.json)

Run commands in the project README to regenerate local artifacts. Generated run directories are ignored by Git by default; retain this record and the source/lockfile, and decide which reports/figures to publish when preparing a portfolio release.

## Next learning milestone

Review and explain development mistakes before expanding the model. Then implement the frozen embedding baseline and the trained neural head with a bounded experiment ledger and actual training/tuning curves. Calibration and routing require their own evidence afterward. The research map documents the papers and mathematics for those stages.

