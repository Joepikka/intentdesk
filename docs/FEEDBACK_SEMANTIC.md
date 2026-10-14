# Semantic S1: semantic text and ordered ratings

The user approved a bounded semantic/ordinal experiment after the completed V3 IDF comparison. This uses the separate **Semantic S1** series; the concurrently authored lexical 30-epoch experiment uses V4. V1/V2/V3 sources, artifacts and consumed evaluations remain unchanged. The existing app stays on V2 until a development-qualified successor is verified. Five-star review prediction remains separate from the original 150-intent classifier and card workflow. No 90% result is guaranteed.

## Availability and fixed allocations

The actual audit excluded 6,484 previously exposed development products and 45,851 products across all three consumed final checks. It found 327,555 eligible development reviews from 124,497 unused products and 131,785 test reviews from 85,179 unused products. These are availability counts, not fitting counts or evidence of model quality.

The pilot allocates the first eligible hashes without rating balancing: **20,000 fit**, **5,000 new development**, **5,000 calibration**, and **20,000 new final-check reviews**. Development and calibration exclude all old development products and each other's entire products. The final check excludes every product in all three consumed checks. Fit/development/calibration/test product overlap must be zero, with support for all five ratings. Original source checksums, review-length rules, rating-declaration removal, duplicate/fingerprint policy and product partitioning stay unchanged.

The separately running lexical V4 experiment additionally protects this semantic pilot's first-unused 20,000-review cohort. S1 explicitly excludes V4's 18,277 reserved products too and records zero cross-experiment product overlap. Actual S1 allocations contain 14,597 fit products, 4,651 new development products, 4,711 calibration products and 18,097 final-check products. Reserved evaluation text is inspected only for allocation/identity, never for model selection. The availability counts above precede this extra concurrent reservation.

Excluding exposed groups shifts the population toward less-reviewed products. Compare fixed candidates on this same new sample; do not describe a difference from an older test percentage as an improvement. User-level, temporal and semantic-near-duplicate independence remain unestablished. Pretrained-model exposure cannot be completely ruled out.

## Controlled comparisons

The local encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Actual loading measured **22,713,216 encoder parameters**. Files/tokenizer hashes are recorded; no remote model code executes. Its model license is Apache 2.0; this does not license the Amazon review data. It runs on two CPU threads without paid inference APIs or cloud hardware.

Every pilot candidate uses the identical 20,000 training reviews. The fixed comparisons are:

1. Training-fitted word/bigram TF-IDF plus balanced logistic regression.
2. Frozen MiniLM prefix embeddings plus the same logistic-regression configuration.
3. Frozen MiniLM beginning/ending embeddings plus the same logistic-regression configuration.
4. A trainable nominal linear head versus an ordered cumulative-link head on each of those two semantic views.
5. The unchanged full-fit V2 model as the release control. It trained on 2,912,393 fit reviews, so this control is explicitly not a same-budget algorithm comparison.

The two embedding views use 128-token windows, including special tokens. `prefix` reads the first 126 wordpieces. `ends` averages normalized embeddings from the first and last 126 wordpieces for longer reviews, then normalizes again; short reviews use the prefix once. This includes later opinions but **omits the middle** of sufficiently long reviews. It is not a full-document or arbitrary-context transformer. Both views share the same tokenizer, encoder and data. Truncation/long-review counts and resumable feature receipts are recorded.

Same-budget logistic learning curves at 5,000, 10,000 and 20,000 fit rows use separately training-fitted lexical vocabularies and identical semantic features. These smaller models are descriptive curves, not extra final-selection candidates. They provide evidence for a later scaling decision; this declared pilot does not silently expand to millions of neural-training reviews.

## Ordered prediction and training

The nominal head predicts five class logits. The ordered head learns a scalar text score and four strictly increasing thresholds. If `q_k = sigmoid(score - threshold_k)` represents `P(rating > k)`, the five probabilities are `[1-q1, q1-q2, q2-q3, q3-q4, q4]`. Positive softplus threshold gaps enforce consistency. All five original rating labels remain intact.

This is a cumulative-link implementation inspired by [CORAL](https://arxiv.org/abs/1901.07884), with explicit threshold ordering. It is not a reproduction of that paper's image/age experiments or proof of improvement on reviews. The nominal objective is weighted cross-entropy; the ordered objective is the mean of four binary cross-entropies, weighted by the same inverse fit-class frequencies. Both use AdamW, learning rate .003, batch 128, weight decay .01, maximum 30 epochs, minimum five, and patience five. Best development macro-F1, then accuracy, then NLL selects the checkpoint; earlier ties remain. Best-checkpoint fit diagnostics use only the first 2,000 fit rows.

One conditional partial fine-tune is allowed if the best frozen neural head beats the lexical baseline by at least .005 development macro-F1. It starts from the pinned encoder and that head; only the final encoder block and head train. A real forward/backward profile must have finite gradients, process RSS at most 1.5 GiB and at least 256 MiB available system memory. Batch four, encoder learning rate .00002, head rate .001, maximum three epochs, development patience one, and a cumulative 90-minute budget bound this adaptation. Only completed-epoch checkpoints can enter selection; unfinished weights never displace a valid checkpoint. Checkpoints preserve optimizer/RNG progress. This does not fine-tune every encoder layer.

Overall selection uses **new development macro-F1**, then accuracy, then NLL. V2 stays selected unless the winner gains at least .005 macro-F1 and loses no more than one percentage point of accuracy. Selection is frozen before calibration and final exposure. No final score can replace this rule.

## Confidence and uncertainty

The 5,000 calibration reviews are split by product hash into two separate product groups. One chooses temperature from a fixed grid by NLL; the other chooses an advisory confidence threshold. The threshold aims for at least 85% observed accuracy at at least 20% coverage on its calibration fold. If no threshold qualifies, all cases remain for review. This is an observed calibration target, not a statistical guarantee.

Report overall accuracy, macro-F1, per-rating errors and mean absolute star error separately from confident-subset accuracy and coverage. Temperature preserves the class argmax. Higher subset accuracy must never be claimed as 90% overall five-class accuracy. The final report includes descriptive paired product-bootstrap intervals against V2. Every app output still needs human interpretation; no banking action, external ticket or automatic retraining is authorized by confidence.

## Human work remains human

The run prepares **200 blinded fit-only review rows**, 40 per source rating, under its ignored artifact directory. `human_audit_blind.csv` has blank human-rating, ambiguity, reason and reviewer fields. `human_audit_reference.csv` is a sealed comparison sheet to inspect only afterward. CSV display cells escape formula-leading text; the original model data is unchanged. Do not publish the raw review worksheets or treat model-generated labels as completed human reviews. Do not rewrite evaluation labels based on the model.

The existing [manual-versus-assisted card pilot](CARD_SUPPORT_PILOT.md) is ready for actual participants. Completing it, assessing ambiguous review wording and demonstrating your own understanding require real human work. They remain explicitly pending until someone performs them. No customer saving or human-benefit result is invented.

## Execution and state

Run `rtk proxy uv run --no-sync python scripts/run_feedback_semantic_job.py` from the repository root. A shared OS lock refuses duplicate jobs. Checkpoints, reserved texts, cache arrays, worksheets and model files stay under ignored `artifacts/`, `models/` and `local/feedback_semantic_job/`. Progress and failures are UTF-8 logs; fitting itself makes no AI conversation calls.

The current launch uses `--wait-for-lock` because the lexical V4 job is already active. A lightweight worker waits for the shared lock without importing PyTorch or loading models, then starts this job once the slot is free. It never interrupts or modifies the other experiment. The two runs' source/configuration and final checks remain separate. Do not launch another queued copy.

Watch progress in PowerShell:

```powershell
rtk proxy powershell -NoProfile -Command "Get-Content -LiteralPath 'D:\ml model forex\intentdesk\local\feedback_semantic_job\training.log' -Tail 20 -Wait"
```

Ctrl+C stops that viewer, not the background job. Source/data/config/encoder/artifact hashes protect resume and reload. The final marker records exposure before feature extraction or prediction; interrupted final evaluation refuses reset. Keep all earlier and new markers.

**Current state: the real Semantic S1 job is queued behind the shared training lock after 185 scope-relevant Python tests passed.** Its run is `feedback_semantic_20261014T064628337312Z`. The separate lexical V4 job remains active; S1 does not run concurrently. All 16 protected original files are byte-identical. New tests verify ordinal probability/gradient math, window endings, confidence coverage, blank human worksheets, exclusion of consumed and concurrent product cohorts, actual toy training/reload/final consumption, cache resume/tamper checks and interoperability with the existing OS lock. The waiting runner imports no ML libraries.

The initial eight-fictional-text encoder profile used about 473 MB process RSS and took 10.8 seconds including a pinned download; it is not a training or throughput benchmark. [The preflight snapshot](../reports/semantic_preflight/summary.json) contains actual allocation/probe evidence. Results, scaling feasibility, any qualified app integration and final measured GitHub delivery will be recorded after the actual run. No new accuracy gain or human validation is claimed yet.
