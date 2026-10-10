# Reproduction reference and historical experiment context

Start with the current [public README](../README.md). The longer command reference below preserves milestone context; later Milestones 5–7 supersede earlier prospective status. Models, caches and local databases are not published.

## Fresh clone versus the original experiment

The repository's benchmark reports and metadata-only [experiment bindings](../reports/experiment_bindings/README.md) describe the original completed local runs. A fresh clone trains new artifacts; random initialization, package/platform differences and source newline normalization can change hashes and outcomes. It must not claim exact original-weight reproduction.

The basic README sequence prepares data, trains and serves without evaluating the published final partition again. To add the optional scope advisory after `build-policy`, run `prepare-scope`, `train-scope`, then `evaluate-scope` before restarting the server. This regenerates that separate advisory/check on the same public benchmark allocation; it is a reproduction of an exposed protocol, not fresh independent evidence. Preserve the original reports in Git and archive any report you will regenerate before doing so.

On the original workspace, never remove or replace `artifacts/final_experiment.json` or `artifacts/scope_v1_experiment.json`. Completed evaluations reuse their bound reports. On a fresh clone, `evaluate-final` refuses a published report whose binding differs from newly trained artifacts. That refusal is intentional: preserve the historical report, and design any new research experiment with untouched evaluation data rather than deleting safeguards or tuning to the published final results.

Raw corpus/model downloads and training take time and storage. The large retrieval corpus is optional for the review UI/pilot. All tests and the toy gradient exercise work without those downloads. A benchmark reproduction is not evidence that the author independently derived or understands every implementation choice.

## Extended command reference

A beginner-friendly applied ML project: learn request categories, compare models, and build toward reliable routing and knowledge assistance.

Returning to the project? Begin with [Start here](START_HERE.md) for the demo and a step-by-step learning sequence.

**Concrete workflow:** open **http://127.0.0.1:8765/pilot** for the card-support demonstration. Review 14 fictional tickets, compare manual categorization with ML suggestions, and prepare local category-to-team handoffs. Six card categories, outside-desk handling and clarification make the scope explicit. No bank system is connected. The automated verification run is excluded from human results; **zero completed human comparisons means no measured human benefit yet**. See [the pilot guide](CARD_SUPPORT_PILOT.md) and [Milestone 7](../reports/Milestone_07.md).

Current verification: **97 Python tests and 2 JavaScript regression tests pass**. With the app running, export the human-comparison report using `uv run --locked intentdesk pilot-report`. The pilot stores its sessions separately in `local/pilot.sqlite3`; original review history is preserved. Reports disclose fictional provenance, interruptions, model wait time and excluded automated sessions.

**Current scope: a free local ML review app with calibrated intent suggestions, SQLite decision history and full-corpus BM25 search.** The final frozen classifier reached **96.24% accuracy on 4,500 known-intent test requests**. Its unsupported-request policy failed, so every request requires review. No external action is executed. Search uses the separate MS MARCO research corpus; it is not a support knowledge base or a generated-answer feature.

**Scope update:** a separately trained binary advisory now flags likely unsupported requests, hides misleading category suggestions and proposes an editable `oos` review decision. It flagged 97/100 fresh OOS check examples and incorrectly flagged 78/1,496 reused known examples. The original frozen thresholds would have accepted 13 of those OOS examples as supported; the new advisory accepted 3. This small check does not authorize automatic routing. See [Milestone 6](../reports/Milestone_06.md) and [the scope lesson](../learning/Scope_Detection.md). Existing queue records keep their old analysis; **Analyze again** creates a new assessment.

## Open the local app

From this project directory:

```powershell
uv sync --locked --extra dev --extra neural --extra web
uv run --locked --extra neural --extra web intentdesk serve
```

Open **http://127.0.0.1:8765**. Keep the terminal running; Ctrl+C stops it. Submit a fictional request, inspect suggestions and save a decision. Review history persists in the ignored `local/` directory. The app reuses one CPU model instance and binds to loopback only. It needs no paid APIs, cloud GPU, subscription or API key. Normal electricity/internet/storage costs remain. See [dependency/data terms](../THIRD_PARTY.md).

The final benchmark is **now consumed**, and may not be used to retune the project. [Milestone 5](../reports/Milestone_05.md) supersedes earlier milestones' historical “test not evaluated” status. The primary model was fixed before final evaluation; the final offline routing thresholds accepted 135/1,000 unsupported requests, and remain disabled. See [the current implementation handoff](HANDOFF.md) for boundaries and remaining research.

[Local app and search evidence](../reports/Milestone_04.md) · [Final classifier evaluation](../reports/Milestone_05.md) · [Learning walkthrough](../learning/Review_App_and_Search.md) · [Portfolio presentation](PORTFOLIO.md)

## Reproduce search and verify

```powershell
uv run --locked intentdesk prepare-retrieval
uv run --locked intentdesk index-retrieval
uv run --locked intentdesk evaluate-retrieval
uv run --locked intentdesk search "photosynthesis chlorophyll"
uv run --locked --extra dev --extra neural --extra web pytest -q
```

The index is disk-based and resumable. Allow 12 GiB free beyond the existing corpus while building it. The index/cache/corpus are not included in source control. Build the index before starting the app to enable its search section. The retrieval report uses a fixed 100-query official development sample against all 8,841,823 passages; it is not a leaderboard test or answer-quality result. MS MARCO is non-commercial research data. Source passages may be outdated, incorrect or unsuitable for a customer knowledge base.

To inspect the frozen final classifier report without selecting a new model:

```powershell
uv run --locked --extra neural intentdesk evaluate-final
```

On this completed workspace the command reuses the bound result. Preserve its experiment marker and report. A new source checkout does not contain these private local model artifacts; reproduce the preparation/training stages below before freezing a new independent experiment, and never silently reuse exposed final results as a new selection set.

## Reproduce the scope advisory

```powershell
uv run --locked --extra neural intentdesk prepare-scope
uv run --locked --extra neural intentdesk train-scope
uv run --locked --extra neural intentdesk evaluate-scope
```

The completed experiment reuses its bound result. Preserve `scope_v1_experiment.json` and the new consumed check; do not tune against its outcomes. No user review records fit the detector.

## Run on this Windows machine

Open PowerShell and use the project directory:

```powershell
cd 'D:\ml model forex\intentdesk'
uv sync --locked --extra dev
uv run intentdesk prepare
uv run intentdesk train
uv run intentdesk predict "I lost my credit card"
uv run pytest
uv run python ../learning/math_exercise.py
```

Python 3.12 is configured and dependencies are pinned in `uv.lock`. `prepare` requires internet access to the official public dataset on first use. Training and prediction run locally on CPU and need no paid service or API key. Re-running training creates a new run; preserve prior results rather than choosing whichever run looks best.

## What happens

1. **Prepare:** acquire pinned CLINC150 data and upstream license, validate schema and official counts, audit formatting-variant overlap, create fit/tune/calibrate/validation/locked-test files and a row manifest.
2. **Train:** learn word/character TF-IDF and a logistic-regression classifier from fit rows. Compare it with a majority baseline on tune rows only. Save the model, run metadata, development errors and graphs.
3. **Predict:** load the saved local model and return three candidate intents with uncalibrated scores. Requests always require review at this stage; unsupported requests can still receive plausible labels.

Reports: `../reports/data_audit.json`, `../reports/latest.json`, and `../reports/runs/<run_id>/summary.md`. Local artifacts: `artifacts/<run_id>/`; `artifacts/latest.json` points only to a successfully completed run. Serialized models are intended to be loaded only from your own generated artifacts; checksums detect accidental changes, not authenticity of third-party files.

The first completed run achieved **94.99% accuracy / 94.94% macro-F1 on 1,496 tuning examples**, with 75 development errors. These are preliminary development scores, not an independent final-test result. See [the milestone record](../reports/Milestone_01.md).

The model has fixed initial configurations, not an exhaustive search. `train` and `train-neural` fit weights using fit/tune only. `build-policy` separately uses calibration and official-validation data as described below. The final test is **not evaluated** by these commands.

## Larger local experiment

Everything here runs on this PC. This experiment uses CPU PyTorch, two compute threads, embedding batches of eight and cached features; it does not use a cloud GPU or the local MX450. Allow about 5 GB for source data/model files plus environment/cache overhead. Actual sizes and results are recorded in the milestone report.

```powershell
uv sync --locked --extra dev --extra neural
uv run --locked --extra neural intentdesk embed
uv run --locked --extra neural intentdesk train-neural
uv run --locked --extra neural intentdesk predict-neural "I lost my credit card"
uv run --locked intentdesk prepare-retrieval
uv run --locked --extra dev --extra neural pytest -q
```

The encoder has roughly 109 million pretrained, **frozen** parameters. N1 trains 117,782 parameters; N2 trains 1,389,206 parameters in a 768 → 1024 → 512 → 150 head. Both use the same audited fit/tune rows. `embed` resumes from its last completed batch checkpoint after interruption. `train-neural` starts a fresh run; it restores the best epoch by tune loss and saves a separate `artifacts/neural_latest.json`. The original lexical prediction command retains its original model.

`prepare-retrieval` downloads a 1.035 GB compressed archive, streams extraction and audits all 8,841,823 passages. The uncompressed size is measured in `../reports/retrieval_data.json`. It is a separate research retrieval dataset without CLINC intent labels, under [non-commercial research terms](https://microsoft.github.io/msmarco/). The later `index-retrieval` and `evaluate-retrieval` commands implement lexical search and its development benchmark. MPNet's upstream training includes MS MARCO pairs; disclose that exposure in future dense-retrieval comparisons.

Read [the local neural lesson](../learning/Local_Neural_Experiment.md) for the equations, research sources, data distinction, training settings and exercises. Read [the expanded handoff](HANDOFF.md) for the current hardware constraints and next steps. Neural run reports contain learning curves, per-model metrics, checkpoint hashes and unreviewed development mistakes. Calibration, official validation and final-test examples are not used to select these heads.

Completed expansion: [Milestone 2 — measured results and resource use](../reports/Milestone_02.md). The linear MPNet classifier reached **97.66% tuning accuracy**, versus **96.79%** for the small head and **96.39%** for the larger head. The larger architecture did not win this comparison; it remains available for learning and error analysis. That milestone recorded 29 passing tests. The final test was unused at that stage; Milestone 5 records its subsequent frozen evaluation.

## Calibrated review recommendations

The audit/calibration milestone recorded **48 tests**. The current release passes **97 Python tests plus 2 JavaScript regression tests**. [Milestone 3](../reports/Milestone_03.md) records the cache/input fixes, data-quality review, calibration result and failed routing gate. [The calibration lesson](../learning/Calibration_and_Review.md) explains the mathematics and split discipline.

```powershell
uv run --locked --extra neural intentdesk build-policy
uv run --locked --extra neural intentdesk route "I lost my credit card"
uv run --locked --extra neural python ../learning/verify_router.py
uv run --locked --extra neural python ../learning/audit_development.py
uv run --locked --extra dev --extra neural pytest -q
```

`build-policy` freezes B2/N1, fits temperatures on 1,496 calibration rows, selects thresholds using 620 validation rows and checks the frozen policy on 2,476 separate validation rows. It records check exposure and reuses a completed result instead of retuning. Keep the marker and original report: deleting them to keep selecting against the same check data would invalidate the experiment.

B2's independent in-scope log loss improved **0.2844 → 0.1408**, with unchanged 96.29% accuracy. The proposed routing rule accepted **9/80 unsupported requests**, failing its rejection requirement. Therefore `route` returns calibrated known-label suggestions with **review_required**; no external action is executed. Known-class confidence is not the probability that a request is supported. Requests exceeding the evaluated 128-token limit are reviewed without truncating and classifying their prefix.

The local app reuses one `intentdesk.policy.Router` instance. The earlier local sequential warm p95 was about 73 ms over 40 requests; CLI startup additionally imports libraries and loads the encoder. No production approval or large concurrent-load claim exists.

Shareable [graphs](../reports/figures/) and [evidence snapshot](../reports/portfolio_evidence.json) are outside Git-ignored run folders and can accompany the public portfolio without raw datasets or model weights. The next model experiment needs audited OOS training examples and fresh reviewed validation evidence. Keep the consumed check set and final-test boundary explicit.

## Understand what you are building

- [Research, papers, concepts, equations and planned graphs](../learning/Research_and_Learning.md)
- [First lesson and exercises](../learning/First_Lesson.md)
- [Data provenance and split policy](../data/README.md)
- [Current implementation handoff](HANDOFF.md)
- [Portfolio goals and presentation](PORTFOLIO.md)

The dataset is crowdsourced English assistant requests, not actual customer tickets. Benchmark scores are development evidence, not proven business impact. Domain/intents come from the upstream project; do not claim account access, banking actions, customer savings or an independently validated final score.

Scope example: `report_lost_card` is supported; password reset is not an intent in this dataset. A password-reset request may still receive a wrong known-label suggestion. The separate scope advisory is implemented, but automatic routing remains disabled because its small, partly reused check does not establish independent joint performance.

