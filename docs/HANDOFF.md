# IntentDesk implementation handoff

Version 1.7, 10 October 2026. Local CPU project for a beginner BTech AI/ML student graduating June 2027, targeting applied ML/AI engineering. The user requires this PC only and a free runtime. No paid model API or cloud service is required.

## Current result

The project contains reproducible data preparation, lexical and frozen-encoder model comparisons, small/large PyTorch heads, temperature calibration, independently checked abstention, a review application and full-corpus lexical search. See README and Milestones 4–6 for measured current results. Earlier milestones are historical evidence, not current test-exposure declarations.

Current verification: 97 Python tests and 2 JavaScript regression tests pass; actual-model HTTP checks, browser save/search checks and wheel asset inclusion pass. The preselected classifier reached 96.24% known-intent final accuracy. Retrieval MRR@10 is 0.2195 on the fixed 100-query development sample. The original scope review used bounded 632px/390px evidence. For the new pilot, valid full-page captures at 1280px, 1920px and 390px are now available, and a fresh UI finish review returned ship. That design verdict does not establish human benefit or production reliability.

[Milestone 7](../reports/Milestone_07.md) adds a concrete **card-support demonstration** at `/pilot`. The user has no permissioned business messages, so its 28 cases and reference labels are assistant-authored fiction. Each reviewer receives 14 unique tickets: seven manual and seven assisted, balanced across six card categories and outside-desk handling. Adjacent reviewer slots counterbalance case pairs. Explicit human decisions prepare local team handoffs; no external action occurs. The original top classifier label is projected into the narrow workflow without selecting a convenient lower-ranked label.

The pilot freezes its data, protocol, model pointers, dependency lock and application sources into a run binding. It uses separate `local/pilot.sqlite3` storage. Reported comparisons include only completed human sessions; smoke sessions, incomplete sessions and interrupted timing have explicit treatment. The current run has **zero completed human sessions and one excluded automated 14-ticket smoke session**. Consequently no speed, accuracy improvement or customer savings is established. Read [the pilot guide](CARD_SUPPORT_PILOT.md); export with `intentdesk pilot-report` while the server runs. [Pilot verification](VERIFICATION.md) records the checks; older reports remain historical snapshots.

Current checks are recorded in [scope verification](VERIFICATION.md). The older verification/app-smoke files remain historical snapshots. Search retains its two-second local app budget: a broad machine-learning query timed out explicitly; the photosynthesis smoke query returned five passages in about 276 ms. Neither single-query timing is a latency benchmark.

[Milestone 6](../reports/Milestone_06.md) adds a separately versioned scope advisory. It uses 100 audited OOS training rows, 51 new OOS selection rows and 100 fresh OOS check rows. New check: 97/100 flagged likely OOS; 3/100 suggested as supported, versus 13/100 under old thresholds. Known check: 78/1,496 incorrectly flagged, using previously temperature-calibrated data. This is not a fresh independent joint evaluation. Its 7.57% one-sided OOS-acceptance upper bound and reused known data do not justify automatic routing. The original failed gate remains unchanged.

The app suggests intents and records decisions. The unsupported-request policy failed development checks, so **all requests require review**. It performs no external actions. The search corpus is an independent non-commercial research benchmark, not support documentation. There is no generated-answer feature or production deployment.

## Run and verify

From the directory containing `pyproject.toml`:

```powershell
uv sync --locked --extra dev --extra neural --extra web
uv run --locked --extra neural --extra web intentdesk serve
uv run --locked --extra dev --extra neural --extra web pytest -q
```

Open http://127.0.0.1:8765. Stop with Ctrl+C. One worker reuses the model. On a new checkout acquire/train artifacts with the commands in README; weights, corpus, caches and review records are deliberately excluded from source control. Preserve local artifacts and data for exact reproduction on the current PC.

## Scientific boundaries that must survive future work

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

Public GitHub publishing, a hosted URL, dense retrieval/reranking, permissioned support knowledge, generated-answer drafting, improved OOS detection and production pilots are separate work. No job application or external publication has been made. Commercial use requires replacing research-only data or securing the appropriate rights. A working research portfolio does not guarantee hiring eligibility or an offer.

Next research requirement: independently adjudicate representative requests against written product scope, obtain fresh known/OOS joint evaluation data and predeclare any next model version. The original reserved OOS training allocation and the new scope check are now used. Preserve both releases' results and review-only behavior. Do not expand model size solely to make the project sound impressive.
