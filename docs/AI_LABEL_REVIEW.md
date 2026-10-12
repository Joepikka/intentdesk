# AI consistency review of synthetic references

Recorded 12 October 2026. This is an AI review against the declared card-desk definitions, **not a human annotation, independent adjudication or permissioned customer dataset**. The local human label sheet was left blank. No reviewer identity or completed human session was invented.

The original 60 references were checked for consistency with the written one-issue workflow:

- The 36 supported examples each assert one actionable lost-card, unauthorized-transaction, physical-damage, declined-payment, PIN-change or already-ordered replacement-delivery issue. A negated mention of another issue should not count as a second actionable issue.
- The 12 outside examples request a different service or refer to a non-banking object. “Card” can mean a memory or greeting card; a SIM PIN is not a bank-card PIN. An unrelated replacement delivery is outside this desk.
- The 12 clarification examples omit the issue, request multiple actionable issues, contradict themselves or explicitly lack enough information to resolve the issue. This desk deliberately asks the reviewer to choose one issue first; a different business could support multi-label routing instead.

No provisional label was changed during this consistency review. Particular judgment calls remain: an unfamiliar charge could justify fraud review even before authorization is resolved, a stolen card may be associated with fraud, and forgotten-PIN handling may overlap PIN-change procedures in a real business. The supplied synthetic policy separates those meanings. A person should resolve these judgments before claiming reference-label correctness.

The later 48-case and 32-case checks follow the same definitions and are also AI-authored, provisional and unreviewed by a person. They do not substitute for real customer traffic. Case-family generation produces many correlated training variants, so training-row counts do not mean that many independently collected examples.

To supply actual human review, follow [the robustness guide](ROBUSTNESS.md). Keep label/reviewer/note fields local. The model comparisons and this document must never be copied into the human label sheet as if a person supplied them.
