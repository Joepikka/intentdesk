# Card-support pilot: what it does and how to try it

This is a concrete demonstration of a support-review task. A reviewer reads a ticket, chooses its category and prepares a local handoff to an illustrative team. The user confirmed there is no permissioned customer dataset; all 28 source tickets were authored for this demonstration. They have not received independent label adjudication. No actual account, payment or external helpdesk is connected.

Open **http://127.0.0.1:8765/pilot** while IntentDesk is running.

1. Read the category guide before starting. The six supported issues are lost cards, suspected fraud, damaged cards, declined payments, PIN changes and replacement-card delivery. Other issues go to general review; unclear requests need clarification.
2. Enter an anonymous reviewer code such as `reviewer01`, then choose **Start or resume comparison**. Use one code per actual person; the app does not independently verify identities.
3. Complete 14 different tickets. Seven have model assistance and seven hide it. Choose your own category even when a suggestion is available. **Use suggestion** fills the category; it does not save it.
4. Choose **Save category & prepare handoff**. Inspect the recorded category and prepared team, then choose **Open next ticket**. Nothing is sent externally.
5. Mark interruptions before saving. A refresh or resumed session keeps the original case timer; it does not restart it.
6. After all 14 decisions, inspect the comparison table and download aggregate results. Do not look up the synthetic answers in source or have an assistant complete your human comparison.

To export the current report after people have completed comparisons:

```powershell
uv run --locked --extra neural --extra web intentdesk serve
# In a second terminal, from the same project directory:
uv run --locked intentdesk pilot-report
```

The export produces `reports/Pilot_Comparison.md` and `reports/pilot_comparison.json`. It generates a figure from completed human records only. With zero completed humans, the report explicitly says the human effects are unknown; no human-effect figure is invented. The app's download provides the current aggregate protocol/results JSON.

## What the model contributes

The existing frozen MPNet/B2 classifier and scope advisory are reused without training or threshold changes. The pilot restricts the task to six existing labels. A top prediction for another service, such as weather, is **outside this desk** even though it belongs to the original 150-class benchmark. It never searches down the candidate list until it finds a card label. Likely unsupported input goes to general review; unavailable or uncertain scope evidence suggests clarification. Every choice is reviewed.

The category-to-team table is an explicit workflow rule, not a learned claim about real bank departments. For example, Change PIN prepares Card access; Suspected fraud prepares Fraud review. A business would need to validate its own team ownership and process before adopting that mapping.

## What the comparison measures

- Strict category agreement with the demonstration label, including error and clarification counts.
- Server case-exposure-to-decision-receipt time, including network/render delay. Inference wait is measured before exposure and reported separately; decision time is not total end-to-end handling time.
- Changes to the model's suggestion in assisted trials.
- Interrupted, clock-reversed or over-300-second cases excluded from timing. Their decisions remain in accuracy counts. The exclusion rule is fixed before sessions begin.

Each person sees two different messages per category stratum, one per condition. Adjacent reviewer slots use the same pairs with opposite condition assignments; order is randomized and mixed. This reduces some order/difficulty imbalance. It does not remove learning, fatigue, selection bias, duplicate participants or the dependence created by shared synthetic tickets. Incomplete paired slots also weaken counterbalancing.

The aggregate JSON includes exploratory 95% percentile bootstrap intervals over reviewer-level assisted-minus-manual differences, using 2,000 resamples and a fixed seed. At least two completed reviewers are needed for an interval. Timing differences require at least four eligible timed decisions per condition for that reviewer. These are conditional exploratory summaries, not a sample-size calculation or a powered superiority/non-inferiority trial. A narrow interval from a few people or easy synthetic tickets is not strong external evidence.

The study binds its protocol, dataset fingerprint, category/team mapping, implementation, UI, model pointers and dependency lock at startup. Changes create a separate run rather than pooling incompatible outcomes. Local SQLite records preserve earlier runs. Pilot records are separate from the original review queue and never train the classifier. No original benchmark test is reopened or retuned.

## What remains necessary for a business claim

This pilot establishes that the workflow and measurement instrument run. It does **not** establish faster work, lower errors, customer savings or production reliability. Automated verification sessions are marked `smoke` and excluded from human aggregates; their oracle-driven decisions and times have no human-performance meaning.

For a real pilot, agree on one business's task and departments, obtain permissioned de-identified representative tickets, independently adjudicate categories and ambiguity, define measurable quality and timing criteria, and plan adequate reviewer/sample sizes before observing outcomes. Keep new training/selection data separate from the untouched comparison. Include model wait and operational costs when estimating business value. Do not merely replace the source constants with customer text or treat public benchmark success as that validation.
