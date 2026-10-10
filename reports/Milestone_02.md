# Milestone 2: larger local neural experiment and GB corpus

Completed 10 October 2026, using this PC only. No cloud compute or paid API was used. **Final test not evaluated.**

Current neural run: `neural_20261009T204150600532Z`. [Detailed run report](experiments/neural_20261009T204150600532Z/summary.md), [metrics](experiments/neural_20261009T204150600532Z/metrics.json), [epoch history](experiments/neural_20261009T204150600532Z/history.json), [run provenance](experiments/neural_20261009T204150600532Z/run.json).

## Measured development results

Same audited fit/tune allocation: 11,917 fit requests, 1,496 tune requests, 150 intents. Calibration and official validation are unused. Scores are preliminary development results, not an independent final-test estimate or evidence of real-customer performance.

| Model | Tune accuracy | Macro-F1 | Log loss | Multiclass Brier | Wrong predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1: word/char TF-IDF + logistic regression | 94.99% | 94.94% | 0.3990 | 0.1305 | 75 |
| B2: frozen MPNet + logistic regression | 97.66% | 97.63% | 0.2341 | 0.0733 | 35 |
| N1: frozen MPNet + small neural head | 96.79% | 96.76% | 0.1149 | 0.0480 | 48 |
| N2: frozen MPNet + larger neural head | 96.39% | 96.34% | 0.1392 | 0.0544 | 54 |

B1 comes from the preserved first milestone. All other rows come from the current neural run. Brier is the mean row-wise sum of squared class-probability errors, with range 0–2. Reported losses cast probabilities to float64 and normalize tiny row-sum roundoff; invalid probability arrays are rejected.

**The larger head did not win this comparison.** B2 leads on accuracy/macro-F1; N1 has better log loss/Brier before calibration. These metrics answer different questions. Preserve B2 and N1 as candidates for the next calibration/validation stage rather than declaring a production champion. One development split and one seed do not establish statistical superiority.

## What was actually trained

The [MPNet sentence encoder](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) has **109,486,464 pretrained parameters**, all frozen. We used pinned weights, tokenization, masked mean pooling and normalized 768-dimensional embeddings. We did not pretrain or fine-tune that transformer.

- N1: `768 → 128 → 150`, **117,782 trainable parameters**. Selected epoch 39; ran 40 epochs.
- N2: `768 → 1024 → 512 → 150`, **1,389,206 trainable parameters**. Selected epoch 11; stopped after epoch 17, following six epochs without a lower tune loss.
- Both: ReLU, dropout 0.2, AdamW lr 0.001 / weight decay 0.01, batch 128, gradient clipping 5, seed 20261010.
- Restored checkpoints produced exactly matching logits in the 20-example reload check for both heads.

N2's fit loss kept decreasing while tune loss fluctuated above its epoch-11 minimum. That is consistent with overfitting in this experiment. Adding parameters alone did not improve the observed development result. Width and depth changed together, so this does not isolate their individual effects.

![Learning curves and model comparison](figures/milestone_02_neural.png)

## Local resource measurements

Hardware: i5-11300H, about 8 GB installed RAM, MX450 with 2 GB VRAM. This experiment uses CPU only, two compute threads; the GPU is unused.

- Encoding 13,413 fit/tune requests: **389.8 seconds**, excluding initial model download/load. Batch eight, maximum 128 tokens, **zero development requests truncated**.
- Cached arrays: **41,204,992 bytes** total.
- Encoder weight file: **437,971,872 bytes**.
- Current linear-control/neural training stage: **48.4 seconds**, excluding encoding and initial download.
- Training process peak Windows working set: **613,380,096 bytes** (about 613 MB decimal). This excludes the separate encoder process; its peak RAM was not recorded in the first completed embedding invocation. Do not present this as whole-system or end-to-end peak memory.
- N1 checkpoint: 473,575 bytes. N2 checkpoint: 5,559,841 bytes. The encoder must also be loaded for text inference.

No neural API load test or steady-state latency benchmark has been run. CLI startup includes model loading and library imports, so it is not comparable with the earlier warm lexical latency measurement.

## The larger dataset

The official MS MARCO v1 passage corpus has been downloaded, safely extracted and fully audited. [Recorded provenance](retrieval_data.json).

| Item | Measured value |
| --- | ---: |
| Compressed archive | 1,035,009,698 bytes; 1.035 GB decimal |
| Uncompressed collection | 3,061,567,852 bytes; **3.062 GB decimal** |
| Passages | **8,841,823** |
| Archive SHA256 | `70667529e474322327d6441c8f7b621f2a805a7f6220897d9f80e5a8294bd62e` |
| Collection SHA256 | `86e0c5820be6b22280337a67e54cbb626645453d9ac3c377bac45abb81c5653d` |

All passage IDs were checked for contiguous unique order; all rows were checked for format, nonempty passage text and UTF-8. The archive size, ETag and server-published MD5 matched. Processing streamed data to disk; it did not load the whole collection into RAM.

This is a **separate retrieval research corpus**, without CLINC intent labels. It was not used to train the classifier. No search index, query/relevance evaluation or answer generation exists yet. Raw data remains excluded from Git under the official [non-commercial research terms](https://microsoft.github.io/msmarco/). MPNet's upstream model card lists MS MARCO training pairs: disclose that exposure in future retrieval results.

## Verification and run history

**29 tests passed.** Checks cover existing data/split/inference invariants, actual neural backward propagation, learning a small synthetic classification problem, exact checkpoint reload, common loss precision, invalid probabilities, safe archive extraction and corpus-row integrity.

Current run source hashes, checkpoint hashes and embedding-manifest hash all matched after completion. CLI smoke prediction on `I lost my credit card` returned `report_lost_card` with raw score 0.999557 and `review_required: true`. An unsupported science/story request in the first run was still assigned a known label; this illustrates why OOS evaluation remains necessary. Those smoke examples are illustrative, not an evaluation suite.

Two earlier runs are preserved: `neural_20261009T203701297822Z` and `neural_20261009T203945123371Z`. They were repeated to correct reporting precision and then float32 row-sum roundoff, not to change the seed, architecture, learning rate, data or early-stopping settings. Both heads' checkpoint SHA256 values were identical across all three runs. The current run above is authoritative. Repeating the same seed is not a multi-seed uncertainty study.

Development mistakes in the per-head error files are **not yet manually reviewed**. Calibration, OOS routing, official validation, independent final testing, a usable retrieval index and a deployed application remain unfinished.

## Next step

Work through [the neural lesson](../learning/Local_Neural_Experiment.md), explain the parameter counts and curves, and review ten mistakes per head. Then freeze candidate weights, fit probability calibration on the reserved calibration split, and evaluate route/review decisions on official validation/OOS data. Preserve final test until the complete policy is frozen.

In the separate GB-data track, acquire official query/relevance files and build/evaluate a disk-backed BM25 index. See [the expanded implementation handoff](../docs/HANDOFF.md).
