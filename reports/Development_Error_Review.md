# Development error review

Reviewer: AI assistant, 10 October 2026. **Not reviewed or approved by a human.** This covers the first ten stored mistakes from each neural head: 20 error occurrences, 13 distinct development requests. It is a convenience sample, not a frequency estimate across all errors. No calibration, validation or final-test requests were inspected for this review. No labels were changed and no additional weights were trained.

Source: neural run `neural_20261009T204150600532Z`, `N1_small_errors.json` and `N2_large_errors.json`. Original run artifacts are preserved.

| Development ID | Heads | Observed mistake | Assessment and consequence |
| --- | --- | --- | --- |
| train:01300 | N1, N2 | “tell me about today weather”: recorded card-replacement duration, predicted weather | Suspected upstream annotation error. Flag for human review; preserve original metrics and report a separately named one-row sensitivity analysis. |
| train:01895 | N1, N2 | Changing “my username” predicted as changing the assistant's name | User/assistant entity confusion. A future training-only augmentation study should test pronouns and ownership without copying tuning requests. |
| train:02065 | N1 | Adding “carots” to a shopping list predicted as ordering | Explicit list-update phrase was missed. Potential spelling/action confusion. Verify low-confidence review behavior; do not silently relabel the request. |
| train:02602 | N1, N2 | Checking a named existing reservation predicted as asking whether a venue accepts reservations | Intent overlap: confirmation versus availability. Context and explicit clarification matter. |
| train:03121 | N1 | “what would i be referred to as” predicted as asking the assistant's name | Pronoun/identity ambiguity; same entity-confusion family as username changes. |
| train:03280 | N1, N2 | Baton Rouge timezone predicted as current time | Related concepts with distinct labels; needs precise time-versus-timezone reasoning. |
| train:03680 | N1 | “when can i receive rewards on my visa card” predicted redeeming rewards, recorded rewards balance | Wording does not clearly ask a balance. Ambiguous annotation/context; require clarification rather than treating the stored label as an obvious semantic correction. |
| train:03734 | N1, N2 | “pay a bribe using my money market account” predicted transfer, recorded pay bill | Unusual, underspecified request with overlapping money-movement labels. This prototype performs no financial action. The example does not establish the correct operational intent. |
| train:04320 | N1, N2 | “excuse me” predicted cancel, recorded repeat | Missing dialogue context; could mean interruption, repetition or apology. A single-turn classifier cannot resolve every interpretation. |
| train:04673 | N1, N2 | “yes you are” predicted thanks, recorded yes | Short conversational fragment. Context dependence; confidence is insufficient evidence of correctness. |
| train:01405 | N2 | Current time in a named timezone predicted timezone | Geographic/timezone wording dominates the requested action. Larger capacity did not fix the distinction. |
| train:01950 | N2 | “are you from the uk” predicted maybe, recorded assistant origin | Yes/no surface form conflicts with the topic. Candidate selection should use measured results rather than network size. |
| train:04923 | N2 | Clearing a to-do list predicted listing it | Read-versus-modify confusion. A future action system would need explicit action validation beyond an intent prediction. |

## Changes made now

1. Fix missing/corrupted embedding-cache handling and input validation rather than allowing false successful runs.
2. Preserve the stronger B2/N1 candidates; keep N2 as an experimental comparison.
3. Add separate calibration and semantic-support thresholds, with an independent policy check and fail-closed review behavior.
4. Flag the suspected weather/card label issue in `data/quality_flags.json`. Run `uv run --locked --extra neural python learning/audit_development.py` for the supplementary sensitivity report. The primary benchmark remains unchanged.

These changes address integrity and decision-making defects. They do not prove that the model now understands every ambiguous example. Improving classification further needs a new, predeclared training-data experiment and fresh independent evidence. Do not repeatedly tune against the policy-check partition.
