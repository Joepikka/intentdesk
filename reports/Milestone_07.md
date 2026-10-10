# Milestone 7: a concrete support workflow and human-comparison instrument

Completed 10 October 2026. The user selected a **clearly labeled demonstration workflow**, because permissioned business support data is unavailable.

The new `/pilot` surface adapts IntentDesk to six card-support issues, with Outside this desk and Needs clarification as explicit alternatives. A saved category prepares an illustrative team handoff locally. It never executes a financial action or sends a ticket externally. The existing 150-class research demo remains available at `/`.

The existing classifier, scope detector, weights and thresholds are unchanged. Adaptation is an explicit six-category task boundary and category-to-team map. It is not domain fine-tuning. The original final and scope checks remain consumed; both completed result-reuse commands passed without reevaluation.

## Data and measurement

There are 28 assistant-authored fictional tickets, four per category stratum including outside requests. They are demonstration material, not real traffic or independently adjudicated evaluation labels. A reviewer sees 14 unique messages, balanced by category between seven manual and seven assisted decisions. Adjacent reviewer slots counterbalance case pairs; mixed order is seeded. Manual clients receive no model suggestion or expected label. Expected labels are used server-side for scoring after completion.

Server timestamps record decision time, with model wait reported separately. Explicit interruptions, clock reversal and periods over five minutes are excluded from timing while retaining their category decisions. Only completed human sessions enter aggregate outcomes. Smoke sessions remain separate. The fixed protocol, case fingerprint, UI/code, model pointers and lockfile are bound to each study run; changed bindings create separate runs with old records preserved.

Reviewer-level paired differences and exploratory bootstrap intervals are exported after actual human sessions. Shared synthetic cases, convenience reviewers, learning and small samples limit those intervals. This is not a powered business experiment.

## Verified behavior

- 97 Python tests and 2 JavaScript regression tests passed.
- The actual model supplied Change PIN for the PIN example. Browser approval prepared Card access; the next manual ticket had no model suggestion.
- An oracle-driven, explicitly marked smoke session completed all 14 decisions. A repeated decision returned 409; no external action occurred.
- Completed human comparisons remained **0 before and after automation**. Exported human accuracy/timing metrics are null. There is no measured human speed benefit and no human-effect figure yet.
- Genuine full-page assisted-flow screenshots were inspected at 1280px, the user's 1920px width, and 390px. Saved handoff/manual states were inspected at desktop width; results at desktop/mobile. No page overflow was observed at those inspected widths. The static detector could not resolve absolute `/assets/` CSS URLs and reported false default-font/link warnings; real browser CSS was loaded and inspected. The finish-review record specifies its final scope.

Read [the pilot instructions and evaluation limits](../docs/CARD_SUPPORT_PILOT.md), [current recorded comparison](../docs/VERIFICATION.md), [smoke evidence](../docs/VERIFICATION.md) and the implementation handoff. All software remains local and free; no new dependency, cloud training or model API is required.
