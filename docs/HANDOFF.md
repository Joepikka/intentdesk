# IntentDesk implementation handoff

## New user-requested investigation — October 14, 2026

After the completed V2 delivery, the user requested correction of errors and investigation of accuracy below 90%. [The V3 protocol](FEEDBACK_V3.md) records fit/development diagnosis and a bounded fit-only IDF/weighting experiment. V3 training has been launched; its actual fresh evaluation, verification and delivery remain pending. Completion statements below describe the previous delivered release. Do not edit bound V1/V2/V3 model sources or reuse consumed tests to tune this experiment. The expanded software suite now passes 169 Python tests; this is not a new model-accuracy result.

Version 2.1, 14 October 2026. Local CPU project for a beginner BTech AI/ML student graduating June 2027, targeting applied ML/AI engineering. The user requires this PC only and a free runtime. No paid model API or cloud service is required.

## Current result

Completion status: both feedback transformers completed 20 total epochs; development selected epochs 8 and 4. The original final check is consumed and all nine fixed candidates are preserved in Milestone 10. The bounded word/character experiment also completed: on its separate fresh product-group check the development-selected balanced model reached 69.13% accuracy / 0.5591 macro-F1, compared with original 67.955% / 0.5487 on that same sample. Both final checks remain consumed. The verified local app composes the selected V2 classifier with preserved regression, scratch-transformer and clustering outputs. Milestone 11 records all five candidates, including the higher-accuracy/lower-macro-F1 natural weighting. No 90% result, customer savings or production reliability is established. Remaining independent human/business validation is future work; no further accuracy-chasing training is scheduled.

The separate [Feedback ML Lab](FEEDBACK_ML_LAB.md) adds a large-data review-rating task. Two pinned complete Amazon review subsets total 1.98 GB and 4,629,544 raw reviews. Product-group partitioning and normalized exact-text auditing precede selection of 100,000 balanced fit, 10,000 natural development and 20,000 natural final-test reviews. A streaming classifier receives all 2,912,393 eligible fit reviews. Review ratings are a different target from support intent; this expansion does not establish better card routing.

The lab compares logistic regression, linear SVM, a decision tree, a random forest, KNN, a streaming SGD classifier and two compact transformers trained from random initialization. Numeric linear regression and SVD/MiniBatchKMeans add regression and unsupervised diagnostics. The custom transformer implements Q/K/V attention, padding masks, residuals, layer normalization, GELU and pooling; all its parameters train locally. This is supervised transformer classification, not foundation-model pretraining. Different feature, token and data budgets are disclosed.

New implementation lives in `feedback_data.py`, `feedback_models.py`, `feedback_transformer.py`, `feedback_lab.py`, `feedback_eval.py`, `feedback_predict.py` and `feedback_web.py`. The standalone `serve-feedback --port 8790` command serves `/ml-lab` without loading the original intent artifacts or creating pilot records. `app_v3.py` also installs the lab on a restarted full server. Original `app.py`, pilot sources, model sources and consumed experiments remain protected.

The expanded suite passes **162 Python tests and 2 JavaScript tests**, including meaningful attention/gradient/padding checks and a complete toy training/evaluation path. These software tests are separate from the actual large-data results. Preserve raw/prepared data and model checkpoints locally; only source and aggregate evidence belong in Git. The source-data card supplies no blanket license, and the project MIT license does not license the reviews.

[Milestone 10](../reports/Milestone_10.md) records the completed CPU run. The development-selected **streaming_sgd** achieved **70.37% accuracy / 0.5427 macro-F1** on 20,000 product-disjoint final review examples. Scratch models have **1,318,021 / 2,966,533** trainable parameters. Every candidate result is preserved; no final-test reselection occurred. The separate final marker now records test consumption. Preparation/training metadata remains the pre-test snapshot.

The project contains reproducible data preparation, lexical and frozen-encoder model comparisons, small/large PyTorch heads, temperature calibration, independently checked abstention, a review application and full-corpus lexical search. See README and Milestones 4–6 for measured current results. Earlier milestones are historical evidence, not current test-exposure declarations.

Earlier card-workflow verification: 130 Python tests and 2 JavaScript regression tests passed. Earlier actual-model HTTP checks, browser save/search checks and wheel asset inclusion passed. The preselected classifier reached 96.24% known-intent final accuracy. Retrieval MRR@10 is 0.2195 on the fixed 100-query development sample. The original scope review used bounded 632px/390px evidence. For the pilot, valid full-page captures at 1280px, 1920px and 390px are available, and its UI finish review returned ship. That design verdict does not establish human benefit or production reliability.

[Milestone 8](../reports/Milestone_08.md) adds `robustness-template` and `test-robustness`: a free, offline stress test of the existing pilot suggestion projection. Sixty assistant-authored cases produced 38/60 provisional label agreement, two outside-to-supported errors out of 12, and zero correct clarification suggestions out of 12. No human labels were supplied. This is development error analysis, not independent accuracy or a human pilot. Typos, negation and multiple issues remain weaknesses. The tool records a confusion matrix, per-category/tag metrics, runtime errors and sequential latency; it never trains or changes thresholds. See [instructions](ROBUSTNESS.md) and the [preserved aggregate snapshot](../reports/robustness_v1/report.md). Raw exports and reviewer notes remain ignored locally. Original consumed evaluations and pilot/application sources were preserved byte-for-byte. No MiroFish or cloud dependency was added.

[Milestone 7](../reports/Milestone_07.md) adds a concrete **card-support demonstration** at `/pilot`. The user has no permissioned business messages, so its 28 cases and reference labels are assistant-authored fiction. Each reviewer receives 14 unique tickets: seven manual and seven assisted, balanced across six card categories and outside-desk handling. Adjacent reviewer slots counterbalance case pairs. Explicit human decisions prepare local team handoffs; no external action occurs. The original top classifier label is projected into the narrow workflow without selecting a convenient lower-ranked label.

The pilot freezes its data, protocol, model pointers, dependency lock and application sources into a run binding. It uses separate `local/pilot.sqlite3` storage. Reported comparisons include only completed human sessions; smoke sessions, incomplete sessions and interrupted timing have explicit treatment. The current run has **zero completed human sessions and one excluded automated 14-ticket smoke session**. Consequently no speed, accuracy improvement or customer savings is established. Read [the pilot guide](CARD_SUPPORT_PILOT.md); export with `intentdesk pilot-report` while the server runs. [Pilot verification](VERIFICATION.md) records the checks; older reports remain historical snapshots.

Current checks are recorded in [scope verification](VERIFICATION.md). The older verification/app-smoke files remain historical snapshots. Search retains its two-second local app budget: a broad machine-learning query timed out explicitly; the photosynthesis smoke query returned five passages in about 276 ms. Neither single-query timing is a latency benchmark.

[Milestone 6](../reports/Milestone_06.md) adds a separately versioned scope advisory. It uses 100 audited OOS training rows, 51 new OOS selection rows and 100 fresh OOS check rows. New check: 97/100 flagged likely OOS; 3/100 suggested as supported, versus 13/100 under old thresholds. Known check: 78/1,496 incorrectly flagged, using previously temperature-calibrated data. This is not a fresh independent joint evaluation. Its 7.57% one-sided OOS-acceptance upper bound and reused known data do not justify automatic routing. The original failed gate remains unchanged.

The app suggests intents and records decisions. The unsupported-request policy failed development checks, so **all requests require review**. It performs no external actions. The search corpus is an independent non-commercial research benchmark, not support documentation. There is no generated-answer feature or production deployment.

## Run and verify

[Milestone 9](../reports/Milestone_09.md) adds lexical V2 and semantic/character hybrid V3 models trained only on audited fit data plus declared synthetic augmentation. V2 scored 30/48 on its reserved synthetic check with worse outside errors; the result is preserved. That check was inspected and became V3 development data. V3 scored 54/60, 42/48 and 29/32 on a separate reserved synthetic check, versus original 38/60, 28/48 and 21/32. All labels remain AI-authored and human-unreviewed. All checks are now consumed. These results do not establish independent real-traffic performance. Negation, multiple issues and unrelated replacement deliveries still fail in some examples.

The hybrid workflow runs at `/card-v3` through the separate `app_v3.py` wrapper with its own `local/card_v3_reviews.sqlite3` database and original encoder shared in memory. The development server uses port 8788. Original app/pilot sources and browser assets remain byte-identical, preserving the original pilot binding/history. V3 reviews do not create human pilot sessions. New commands: `train-card-v2`, `evaluate-card-v2`, `card-v2`, `train-card-v3`, `evaluate-card-v3`, `card-v3`. The [update guide](CARD_WORKFLOW_UPDATE.md) records preparation, source/data binding, cache resume, operating assumptions and remaining errors. At that stage, checks passed 130 Python and 2 JavaScript tests. AI consistency review stays separate from blank human label sheets.

From the directory containing `pyproject.toml`:

```powershell
uv sync --locked --extra dev --extra neural --extra web
uv run --locked --extra neural --extra web intentdesk serve
uv run --locked --extra dev --extra neural --extra web pytest -q
```

Open http://127.0.0.1:8765. Stop with Ctrl+C. One worker reuses the model. On a new checkout acquire/train artifacts with the commands in README; weights, corpus, caches and review records are deliberately excluded from source control. Preserve local artifacts and data for exact reproduction on the current PC.

## Scientific boundaries that must survive future work

- The feedback-lab final test is consumed. Preserve its `final_experiment.json`, source/data/model hashes and report. Do not alter frozen training/inference/evaluator code under the same run, delete markers or select another model using its final results. Future development requires a separately versioned protocol and untouched evaluation. Raw reviews, audit databases and weights stay local.

- Final CLINC evaluation is now consumed. `final_experiment.json` binds the previously chosen B2 calibrated primary model, original fixed comparisons, weights, data, code and dependency lock before exposure. Do not delete it, tune on final results or describe repeated comparisons as fresh independent evaluation.
- The original policy-selection/check experiment is consumed and failed. It remains frozen. The final score cannot enable routing. New OOS experiments need fresh reviewed evaluation data and a separately versioned protocol.
- Known-class calibrated scores are not probabilities that a request is in scope.
- The larger neural head did not win. The pretrained 109M-parameter encoder remained frozen. Do not claim that all its parameters were trained here.
- The 3.06 GB / 8.84M-passage corpus is for retrieval, not intent-classifier supervision. Retrieval results use a predeclared 100-query official development sample against the full index, not the leaderboard test.
- Review feedback and assistant-authored demos are not independently adjudicated labels. The original 100 OOS training allocation passed fingerprint/provenance checks and now trains the advisory. Public crowd-authored labels have not received new independent semantic adjudication.
- `scope_v1_experiment.json` binds the new detector, thresholds, source and data before its check. Its new OOS check is now consumed; never retune this version using those outcomes. The known diagnostic partition was already used to fit temperature. Preserve both the original final marker and new scope marker.

## Local application contract

FastAPI validates requests; ScopeRouter adds an advisory around the unchanged original Router. Invalid optional scope artifacts fall back to review. SQLite persists requests and versioned review decisions. Analyze again creates a new assessment and preserves the old record; it does not rewrite history. Loopback binding, JSON mutation header, host/origin restrictions, body limits and CSP support the local boundary. The UI uses text-only rendering for user/corpus content and no remote fonts/scripts. Database values are parameters, and stale decisions return 409. Feedback never automatically trains a model. Model/search contention returns retry responses. Search timeouts are explicit.

The local database is unencrypted and private to this workspace/PC account. Copy `local/` only after stopping the server to avoid incomplete WAL backups. This is not a hosted multi-user service: internet exposure would require a separate authentication/privacy/operations design.

## Portfolio delivery and remaining work

Runnable local demonstration, model/data provenance, tested API/review flow, a search baseline, final benchmark report and interview/learning guides are the current deliverable. Record your own explanation/video and complete the learning exercises before presenting the skills as yours. Disclose AI coding assistance.

The source and qualified aggregate evidence are published at [Joepikka/intentdesk](https://github.com/Joepikka/intentdesk). A hosted URL, dense retrieval/reranking, permissioned support knowledge, generated-answer drafting, improved OOS/ambiguity detection and production pilots remain separate work. No job application has been made. Commercial use requires replacing research-only data or securing the appropriate rights. A working research portfolio does not guarantee hiring eligibility or an offer.

Next research requirement: independently adjudicate representative requests against written product scope, obtain fresh known/OOS joint evaluation data and predeclare any next model version. The original reserved OOS training allocation and the new scope check are now used. Preserve both releases' results and review-only behavior. Do not expand model size solely to make the project sound impressive.

The V2 consumed source `feedback_v2.py` and original bound feedback source modules remain frozen. `feedback_release.py` composes their verified outputs for the browser; `feedback_web.py` and local assets expose separately labeled original/fresh result tables. Reproduce V2 with `scripts/run_feedback_v2_job.py` after completing the original release; a repeated unchanged run reuses results. Future model changes need new versioned source/protocol and untouched evaluation groups. Raw data/model files remain local.
