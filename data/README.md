# Dataset provenance

IntentDesk uses **CLINC150**, from Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019.

- [Paper](https://aclanthology.org/D19-1131/)
- [Official repository](https://github.com/clinc/oos-eval)
- Pinned revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- `data_full.json` SHA256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`
- Data license: Creative Commons Attribution 3.0; `prepare` preserves the full text in `CLINC150_LICENSE.txt`.
- English, crowdsourced, single-intent requests across 150 labels and 10 domains; not real customer tickets.

`uv run intentdesk prepare` downloads source files from the pinned revision, records acquisition time and checksums, and validates official counts/labels. Raw/prepared data are excluded from source control by default.

Mechanical overlap policy, fixed before modeling:

1. Keep the official test unchanged, including internal duplicates. Do not evaluate it at this stage.
2. Remove validation records that overlap test fingerprints.
3. Remove training records that overlap either protected validation or test.
4. Exclude conflicting training-label groups and keep the first row per remaining formatting-variant group.
5. Make seeded per-intent fit/tune/calibrate partitions, approximately 80/10/10. Report actual counts instead of forcing provisional totals after exclusions.

Fingerprinting uses Unicode normalization, case folding, whitespace collapse and punctuation removal. It is a conservative formatting-variant check, not semantic deduplication; it can merge punctuation-sensitive requests and miss paraphrases. The manifest records every row's allocation or exclusion. OOS training rows remain unused by models.

Prepared test content is only accessed mechanically during the original audit; model training reads **fit and tune files only**. The policy stage now reads calibration/validation files with checksum, schema and disjoint-group checks. It stratifies official validation into 620 policy-selection and 2,476 independent-check rows. The policy failed its OOS check and remains frozen; no final-test evaluation command exists.

Data transformations are documented in `src/intentdesk/data.py`; text supplied to TF-IDF is otherwise preserved. Domain names come from upstream `domains.json`. The project has not established customer-level, temporal or real-world generalization, or ruled out public benchmark exposure in future pretrained encoders.

`quality_flags.json` records an assistant-reviewed suspected development-label error. Original data/labels remain unchanged; `reports/development_label_sensitivity.json` is explicitly supplementary. The 100 `oos_development` rows remain unused by models and require an overlap/quality audit before a future OOS-training experiment.

## Separate retrieval corpus

`uv run --extra neural intentdesk prepare-retrieval` streams the official MS MARCO v1 passage collection to `data/retrieval/msmarco-v1/`. It verifies the pinned archive ETag, size and server-published MD5, records SHA256, extracts only the expected regular file, then audits all passage IDs, UTF-8 rows and the expected count of 8,841,823. An interrupted download can resume; an interrupted extraction restarts. No whole-corpus in-memory list is created.

The 1,035,009,698-byte archive and measured uncompressed size are separate fields in `reports/retrieval_data.json`. Acquisition does not create a search index or establish retrieval quality. These passages never enter the CLINC classifier. Keep raw content out of the public repository; use is restricted to non-commercial research under the [official terms](https://microsoft.github.io/msmarco/). The pinned [official registry](https://github.com/microsoft/msmarco/blob/21b4ece9f4fc10fa4cec2028bb74287629945df6/Datasets.md) and checksums are retained locally.

