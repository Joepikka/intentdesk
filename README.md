# IntentDesk

**Local NLP support triage with calibrated ML suggestions and human review.**

IntentDesk helps a reviewer identify what a request is about and prepare the appropriate team handoff. It includes reproducible model comparisons, evaluation reports, a Python API, a browser interface and a card-support demonstration. It runs on CPU without paid model APIs.

> Student portfolio project, developed with AI coding assistance. The encoder is pretrained and frozen; downstream classifiers and neural heads were trained locally. All requests require human review. Customer savings and production reliability have not been established.

![Card-support demonstration: model suggestion awaiting a human decision](docs/assets/pilot-assisted.jpg)

## What it does

“I lost my credit card” → suggests **Lost card** → a reviewer confirms or corrects it → prepares an illustrative **Card security** handoff locally.

- **Research demo:** 150 English intent categories, scope advisory, review queue and versioned decision history.
- **Card-support pilot:** six categories plus outside-desk and clarification choices; 14 fictional tickets per reviewer, balanced between manual and assisted review.
- **Separate retrieval experiment:** disk-based BM25 search over 8.84 million MS MARCO passages. These research passages are not customer support documentation or verified answers.

No account is accessed, no external ticket is sent, and no conversational answer is generated. Review feedback does not automatically retrain a model.

## Measured results

| Experiment | Recorded outcome | Interpretation |
| --- | --- | --- |
| Preselected MPNet + calibrated linear classifier | **96.24% accuracy, 0.9624 macro-F1**, 4,500 known-intent final requests | Crowdsourced benchmark performance; frozen encoder |
| Larger neural head | **94.27%** known-intent final accuracy | Bigger did not win; comparison was fixed before final exposure |
| Original unsupported-request thresholds | Would accept **135/1,000** unsupported final requests | Failed policy remains disabled; all requests require review |
| Separate scope advisory | **97/100** fresh OOS examples flagged; **78/1,496** known examples incorrectly flagged | Known examples reuse calibration data; not an independent joint evaluation |
| Full-corpus retrieval | **MRR@10 0.2195**, mean recall@10 **0.435** | Fixed 100-query official development sample, not leaderboard test |
| Manual versus assisted pilot | **0 completed human comparisons** at publication | Automated verification excluded; no measured human improvement |
| Synthetic robustness suite | **38/60 (63.3%)** provisional label agreement; **0/12** clarification cases matched | Assistant-authored development cases, zero human-reviewed labels; not independent accuracy |

[Final classifier report](reports/Milestone_05.md) · [Scope limitations](reports/Milestone_06.md) · [Workflow protocol](docs/CARD_SUPPORT_PILOT.md) · [Machine-readable evidence](reports/release_evidence.json)

[Robustness findings](reports/Milestone_08.md) expose typos, negation and ambiguity weaknesses in the current card workflow. The [recorded run](reports/robustness_v1/report.md) preserves its disagreements and confusion matrix. Its task and references differ from the known-intent benchmark above, so the two percentages are not directly comparable.

![Frozen final model comparison and unsupported-request result](reports/figures/milestone_05_final.png)

## What the repository demonstrates

Python, NumPy, scikit-learn, PyTorch, NLP, TF-IDF, transfer learning, neural network training, ReLU, dropout, cross-entropy, AdamW, temperature calibration, data leakage checks, macro-F1, error analysis, out-of-scope detection, FastAPI, SQLite, BM25 and automated testing.

The project compares a majority baseline, word/character TF-IDF logistic regression, a linear classifier on frozen MPNet embeddings, and small/large PyTorch heads. The encoder has approximately **109 million frozen parameters**. The trained heads have **117,782** and **1,389,206** parameters. It does not demonstrate training a foundation model, reinforcement learning, computer vision or production deployment.

```mermaid
flowchart LR
    A[Request] --> B[Frozen MPNet encoder]
    B --> C[Trained classifier and calibration]
    B --> D[Separate scope advisory]
    C --> E[Suggestions or review fallback]
    D --> E
    E --> F[Human confirms or corrects]
    F --> G[SQLite decision history]
    F --> H[Local illustrative team handoff]
```

## Run locally

Requirements: Python **3.12**, [uv](https://docs.astral.sh/uv/), and internet access for the first dataset/model downloads. This project was built on an 8 GB RAM laptop using CPU execution and two compute threads. Exact timings depend on hardware.

```bash
git clone https://github.com/Joepikka/intentdesk.git
cd intentdesk
uv sync --locked --extra dev --extra neural --extra web

# Acquire audited data and train the local models.
uv run --locked intentdesk prepare
uv run --locked intentdesk train
uv run --locked --extra neural intentdesk embed
uv run --locked --extra neural intentdesk train-neural
uv run --locked --extra neural intentdesk build-policy

uv run --locked --extra neural --extra web intentdesk serve
```

Open **http://127.0.0.1:8765** or **http://127.0.0.1:8765/pilot**. Keep the server terminal running. Trained weights, dataset files, indexes and local review databases are deliberately excluded from GitHub, so a fresh clone needs preparation/training. The pilot can still be used with explicit manual review when the optional scope advisory is unavailable.

For the separately trained scope advisory, optional retrieval index and experiment-preservation rules, read [reproduction instructions](docs/REPRODUCING.md). Retrieval needs roughly 3.06 GB of uncompressed corpus plus a 4.70 GB index and additional build/cache space. It is optional for the card pilot.

The published final and scope reports are historical consumed evaluations. A fresh checkout is a new experiment, not the original weights. Never tune against the published final results or delete frozen experiment markers to reselect a model. Preserve the evidence before generating new reports.

## Verify

```bash
uv run --locked --extra dev --extra neural --extra web pytest -q
node --test tests/queue_page.test.mjs
uv run --locked python learning/math_exercise.py

# With the app running; exports actual recorded human results only.
uv run --locked intentdesk pilot-report

# Without the server; first export a blank sheet for a person to label.
uv run --locked intentdesk robustness-template
uv run --locked --extra neural intentdesk test-robustness
# After actual review: append --labels local/robustness_labels.csv
```

Current local checks (12 October 2026): **108 Python tests and 2 JavaScript tests passed**. Tests use synthetic fixtures and do not require model weights or the large corpus. CI runs the same tests. See [verification scope](docs/VERIFICATION.md).

## Explore the project

| Start with | Purpose |
| --- | --- |
| [Five-minute demonstration](docs/DEMO.md) | Walkthrough with examples, screenshots and honest narration |
| [Learning and interview preparation](docs/INTERVIEW_PREP.md) | Equations, code tracing, exercises and questions to answer yourself |
| [Research and learning map](learning/Research_and_Learning.md) | Papers, concepts and implementation connections |
| [Portfolio evidence](docs/PORTFOLIO.md) | Skills, supported resume wording and limitations |
| [Card-support pilot guide](docs/CARD_SUPPORT_PILOT.md) | Categories, human comparison, timing and data limitations |
| [Local robustness testing](docs/ROBUSTNESS.md) | Synthetic stress cases, blank label sheet, error analysis and graphs |
| [Implementation handoff](docs/HANDOFF.md) | Current architecture, frozen experiments and next work |
| [Third-party terms](THIRD_PARTY.md) | Dataset, model and dependency attribution |

Core code is under `src/intentdesk/`, tests under `tests/`, and recorded aggregate reports/figures under `reports/`. Historical milestone counts and test-exposure statements describe those stages; Milestones 5–8 establish the later state.

## Limitations and next work

This is a local English research application, not a banking service. The pilot tickets/reference labels are assistant-authored fiction without independent adjudication. The scope check is small and partly uses reused data. Public pretrained-model exposure to benchmark examples cannot be ruled out. Search can time out under its two-second app budget.

Next: independently review the robustness labels and investigate the recorded failures, complete real human comparison sessions, then seek permissioned representative data and a new untouched joint evaluation. Later model changes require a separate protocol and fresh evaluation. Laya and MiroFish are not integrated here.

## Authorship and licensing

AI assistance was used for implementation, debugging, documentation and experiment scaffolding. Repository ownership does not mean every line was written unaided or every concept mastered. The author should demonstrate understanding by explaining, running, debugging and extending the work; [the learning checklist](docs/INTERVIEW_PREP.md) is intentionally uncompleted.

Original project code and documentation are available under the [MIT license](LICENSE). Third-party datasets, weights and dependencies retain their own terms. CLINC150 attribution is preserved; MS MARCO is restricted to non-commercial research. No raw corpus, model weights, credentials or user-review records are included.
