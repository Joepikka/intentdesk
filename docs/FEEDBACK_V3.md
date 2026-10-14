# Rating error diagnosis and bounded IDF experiment

The user requested another investigation into accuracy below 90% after the completed V2 delivery. This is a new, bounded experiment. Earlier model sources, artifacts, predictions and consumed evaluations remain preserved. The original 96.24% intent result is a different task from predicting five subjective review ratings.

## Evidence before changing models

Actual V2 development accuracy/macro-F1 are 72.02%/0.5500 for the selected balanced model and 76.62%/0.4968 for the natural-weight alternative. Selecting a higher accuracy alone would sacrifice minority-rating performance. Two-star recall falls from 31.19% to 6.44% under natural weighting. The natural fit population contains 61.07% five-star reviews.

A read-only diagnostic compared the same models on the first 10,000 eligible fit hashes (natural distribution, in-sample) and the existing 10,000 development reviews. The balanced model scored 76.73% versus 72.02% accuracy and 0.6539 versus 0.5500 macro-F1. Of 2,798 development mistakes, 2,259 (80.74%) were neighboring-star errors. Long reviews also performed worse: development accuracy was 75.28% below 200 characters, 65.08% at 200–799 characters, and 51.03% at 800 or more characters. The last slice has only 194 reviews. These are observational diagnostics; they do not prove a cause or an accuracy ceiling.

The review representation retains raw HTML formatting and entity codes. Unigrams/bigrams and within-word character n-grams represent limited order; they do not fully model a changing opinion or an updated review. Public stars can be subjective and ambiguous from text alone. No label is changed or difficult example excluded to make the score rise. There is no evidence here that 90% is impossible, or that additional data/epochs would guarantee it.

The scratch neural models show a different problem: later fit loss falls while development loss rises. Their selected checkpoints remain epochs 8 and 4 after 20 completed epochs. More neural epochs are not justified by those saved curves. Their first-63-token budget is another disclosed limitation; the new full-text lexical candidate does not use that truncation.

The database query planner also selected a secondary index followed by a temporary sort of millions of complete review strings. The V3 iterator explicitly scans the primary hash key. It preserves eligibility, order and resume keys while avoiding that temporary sort. This fixes a verified resource inefficiency; it is not an explanation for the accuracy score.

## Frozen next experiment

- Same 2,912,393 eligible fit rows and five labels. Existing fit/development/product-group rules and rating-declaration cleanup stay unchanged.
- Decode HTML entities, remove markup and video-formatting markers, preserve Unicode text and sentiment words.
- Hashed word 1–2-grams and character-within-word 3–5-grams, 262,144 dimensions per view. Hash collisions remain a limitation.
- Fit document frequencies on all fit text only. Smooth IDF is `log((1+N)/(1+df))+1`; term frequency is `1+log(count)`. Normalize the combined sparse vector with L2. Neither development nor final text fits IDF.
- Two fixed SGD log-loss candidates, alpha 0.000001 and averaged weights: full inverse-frequency class weighting, and its square root. Normalize the expected fit weight to one for both alternatives.
- Maximum three training passes, minimum two, stopping after a pass without any candidate improving its own development score. Save the best checkpoint by development macro-F1 then accuracy; earlier ties win. A resumed job preserves the same stopping rule.
- Compare against the exact development-selected V2 balanced model. Retain V2 unless development macro-F1 improves by at least 0.005. No final result can change this decision.
- The changes are bundled; this tests a practical alternative, not each individual change's causal contribution.

Reserve 20,000 fresh reviews excluding **every product** in both previously consumed checks. The selection follows eligible hash order without rating balancing. This shifts the population further toward less-reviewed products; compare candidates on the same new check and do not compare its accuracy directly to an earlier population's percentage. The final marker is written before any fresh predictions; an interrupted exposure refuses reset/reuse for tuning. Model, source, configuration, dependency lock, manifest and check hashes are bound.

## Execution and completion status

Run `rtk proxy uv run --no-sync python scripts/run_feedback_v3_job.py` from the repository root. The shared OS lock prevents duplicate training. Checkpoints and UTF-8 logs live under ignored `artifacts/` and `local/feedback_v3_job/`. The job runs on two local CPU threads, in small sparse batches. There are no model API calls or paid cloud services. Raw reviews, weights and logs remain excluded from GitHub.

**Current state: training launched after 169 Python tests passed. Actual V3 results and final verification are pending.** No V3 improvement or 90% result is claimed. The live application continues to use completed V2 until a qualified new release is verified. After completion, preserve negative results as well as gains, inspect reload and aggregate graphs, integrate only a development-qualified winner, update documentation and publish verified source/aggregate evidence. Stop this bounded attempt rather than automatically launching another accuracy search.

Reproduce aggregate-only diagnosis with `rtk proxy uv run --no-sync python scripts/diagnose_feedback.py`. It opens fit/development data and never prints or publishes review text. Private example inspection informed the hypotheses but is not a human-label audit or customer validation.

The actual [aggregate diagnosis](../reports/feedback_diagnosis/summary.json) preserves class reports, confusion matrices, neighboring-star errors, length slices and source/model identities. Software verification before launch: 169 Python tests passed, including seven new V3 tests; all 16 protected files stayed byte-identical. The publication scan found no private artifacts, credentials or broken documentation links. These checks do not establish V3 accuracy.
