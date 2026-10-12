# Synthetic robustness testing

This free local tool checks the existing card-support suggestion workflow against 60 assistant-authored development cases. It requires no MiroFish, LLM API, cloud service or new dependency. It does not retrain a model, change thresholds, reopen the original benchmark or create human pilot sessions.

The suite includes six card categories, outside-desk requests and clarification cases. Phenomena include typos, negation, emotion, distracting context, misleading keywords, other known intents, ambiguity, contradictions and multiple actionable issues. The multiple-issue policy requires clarification because this demonstration supports one category per request. That is a declared product rule, not proof that every business should handle those requests identically.

## Review labels before seeing predictions

```powershell
uv run --locked intentdesk robustness-template
```

This creates `local/robustness_labels.csv` without loading a model. The sheet contains messages, tags and blank `reviewed_label`/`reviewer` fields. It deliberately omits draft labels and model predictions. Use one of these labels:

`report_lost_card`, `report_fraud`, `damaged_card`, `card_declined`, `pin_change`, `replacement_card_duration`, `oos`, `uncertain`.

A person should supply a category and anonymous reviewer code, with notes for ambiguous examples, before viewing predictions or draft answers in source. An independent second reviewer can resolve disagreement. Identity, independence and whether a reviewer saw predictions are not technically verified; the tool describes these as reviewer-declared labels. It never automatically marks generated labels as adjudicated. Keep private reviewer notes local. Export refuses to overwrite an existing sheet.

## Run the local model

Existing trained artifacts are required, as described in the README. The server does not need to run. On a small laptop, stop the server before testing if memory is tight; this CLI loads its own model instance.

```powershell
# Runs now against explicitly provisional labels; useful for finding development failures.
uv run --locked --extra neural intentdesk test-robustness

# After a person has actually completed the label sheet:
uv run --locked --extra neural intentdesk test-robustness --labels local/robustness_labels.csv
```

Each invocation creates a separate timestamped directory under ignored `reports/robustness/` containing `report.md`, `summary.json`, `predictions.csv` and `robustness.png`. Previous runs are preserved. Optional `--cases PATH` accepts a versioned synthetic suite in the same format; `robustness-template --output PATH` selects another sheet location. Reviewed rows must match the exact dataset hash, case identifiers and original messages.

The model is loaded once and receives one declared warm-up request. Every case is then evaluated once in fixed file order. Predictions reuse the same scope advisory and top-label projection as the pilot. Model startup/warm-up are reported separately. Sequential p50/p95 reflect this single local run, not production load or human handling speed. Runtime failures count as incorrect suggestions, appear in the confusion matrix's Error column, and retain their elapsed times in CSV; successful-request latency percentiles exclude them.

## Interpret the report

- **Provisional agreement:** all cases scored against assistant-authored, unreviewed labels. Per-category precision/recall/F1, confusion matrix and phenomenon scores inherit that limitation.
- **Reviewer-declared agreement:** a separate subset scored against supplied reviewed labels; omitted labels never count as reviewed. This subset does not replace or silently mix with provisional outcomes.
- **Outside-desk errors:** how many outside references receive one of the six supported suggestions; known card cases incorrectly flagged outside are also counted.
- **Ambiguity:** how often unclear/multiple-issue references receive a specific category instead of clarification.
- **Tags overlap:** a case can belong to several phenomenon groups; group counts are not independent experiments.

Graphs always identify synthetic development data and unreviewed provisional references. No confidence interval or generalization claim is inferred from this hand-selected suite. Several examples target already observed failure types; it is not a fresh untouched holdout. Even human review of synthetic labels does not establish real-traffic performance or customer savings.

Use disagreements to inspect references and document failures. Any later tuning must be followed by a separate untouched, representative and independently adjudicated evaluation. Preserve the original consumed final/scope checks and the separate 14-ticket human study. Automated robustness checks cannot substitute for human sessions.
