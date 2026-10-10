"""Assemble shareable reports from completed frozen results, never rerun experiments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from intentdesk.common import read_json, sha256_file, write_json

root = Path(__file__).resolve().parents[1]
final = read_json(root/"reports/final_evaluation.json")
retrieval = read_json(root/"reports/retrieval_evaluation.json")
index = read_json(root/"reports/retrieval_index.json")
primary = final["primary_calibrated_model"]
models = final["secondary_fixed_model_comparison"]
proposed = final["proposed_thresholds_offline_only"]
figures = root/"reports/figures"
figures.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
names = ["Lexical", "MPNet linear", "Small NN", "Large NN"]
values = [m["accuracy"]*100 for m in models.values()]
axes[0].bar(names, values, color=["#84988a", "#146146", "#5f8d78", "#b4c7b7"])
axes[0].set(ylim=(0, 100), ylabel="Accuracy (%)", title="Frozen models: 4,500 known test requests")
axes[0].tick_params(axis="x", rotation=22)
for i, value in enumerate(values):
    axes[0].text(i, value+1.2, f"{value:.2f}", ha="center", fontsize=8)
bins = primary["reliability"]["bins"]
axes[1].plot([0,1], [0,1], linestyle=":", color="gray")
axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#146146")
axes[1].set(xlim=(0,1), ylim=(0,1), xlabel="Mean known-class confidence", ylabel="Observed accuracy", title="Fixed calibrated primary model")
axes[2].bar(["Unsupported accepted", "Unsupported rejected"], [proposed["accepted_oos"], 1000-proposed["accepted_oos"]], color=["#a8673b", "#84988a"])
axes[2].set(ylabel="Unsupported test requests", title="Offline thresholds only: routing disabled")
axes[2].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(figures/"milestone_05_final.png", dpi=160)
plt.close(fig)

report4 = f"""# Milestone 4: free local app and full-corpus search

Completed 10 October 2026. The app provides a Python API, browser review workbench, SQLite request/decision history and local BM25 search. No paid API or cloud is required. The classifier's failed development gate remains active: all requests require review, and no external actions occur.

## Retrieval evidence

- Indexed **{index['passages']:,} passages**, from the separately audited **3.062 GB** MS MARCO corpus.
- Finished index: **{index['index_bytes']/1e9:.3f} GB** on disk; indexing/integrity/hash time **{index['build_seconds']:.1f} seconds** across this run's committed work.
- A 64 MiB SQLite page cache, 5,000-row transactions, resumable byte offsets and disk-based merges keep the corpus out of a giant Python/RAM structure. This is not a claim that the whole process uses only 64 MiB.
- Row count, FTS integrity, SQLite quick-check and finished index SHA-256 passed.
- Fixed sample: **{retrieval['query_count']} official development queries**, chosen by hashed ID order before retrieval, searched against the complete corpus. Query/relevance source revisions and checksums are preserved.
- **MRR@10: {retrieval['mrr_at_10']:.4f}**; **mean recall@10: {retrieval['recall_at_10']:.4f}**; hit rate@10: {retrieval['hit_rate_at_10']:.4f}.
- Query-bootstrap MRR interval: [{retrieval['mrr_bootstrap_95_percent'][0]:.4f}, {retrieval['mrr_bootstrap_95_percent'][1]:.4f}]. Small development sample and incomplete relevance judgments limit its interpretation.
- Sequential benchmark median **{retrieval['median_ms']:.1f} ms**, p95 **{retrieval['p95_ms']:.1f} ms**; **{retrieval['timeouts_counted_as_zero']}** ten-second timeouts counted as zero retrieval success. These include per-query read-only SQLite connection work on this PC, not API/concurrent performance.

This is a lexical retrieval baseline. It does not use the MPNet encoder, neural reranking, a support knowledge base or generated answers. Search results are passages, not verified facts. MS MARCO is non-commercial research data and is not included in the source repository.

## Review application

Run `uv run --locked --extra neural --extra web intentdesk serve`, then open http://127.0.0.1:8765. Submit a fictional request, compare known-label scores, choose an intent/`oos`/`uncertain`, save a decision and inspect its history. SQLite persists these records in the ignored `local/` folder. Feedback never automatically retrains a model. Optimistic versions prevent stale-tab overwrites. Model/search operations return retry messages when busy; long searches have a two-second interactive budget.

The application binds only to loopback and uses a single reusable model process. Request validation, body-size limits, host/origin checks, a required mutation header, CSP and text-only rendering cover this local-demo boundary. It has no multi-user authentication or internet deployment design. Local review records are unencrypted files accessible to the PC owner.

[Search evidence](retrieval_evaluation.json), [index metadata](retrieval_index.json), [learning walkthrough](../learning/Review_App_and_Search.md). Browser screenshots and verification evidence accompany the final handoff. Demo reviews written by the assistant are labeled as such and are not independent human validation.
"""
(root/"reports/Milestone_04.md").write_text(report4, encoding="utf-8")
rows = "\n".join(f"| {name} | {m['accuracy']*100:.2f}% | {m['macro_f1']:.4f} | {m['log_loss']:.4f} |" for name,m in models.items())
report5 = f"""# Milestone 5: frozen final classifier evaluation

Completed 10 October 2026. The final CLINC partition is now **consumed**. Before reading final examples, the primary model, temperatures, thresholds, model/data/source hashes and dependency lock were frozen in `artifacts/final_experiment.json`. Final results cannot select a model, change thresholds or enable routing. Re-running the identical completed experiment reuses its report.

## Primary result

The previously selected **MPNet + linear classifier with temperature scaling** achieved **{primary['accuracy']*100:.2f}% accuracy** and **{primary['macro_f1']:.4f} macro-F1** on **4,500 known-intent test requests**. Log loss: **{primary['log_loss']:.4f}**; Brier: **{primary['multiclass_brier']:.4f}**; ECE over 15 equal-width bins: **{primary['reliability']['ece']:.4f}**. The formatting-group bootstrap accuracy interval is [{primary['group_bootstrap_accuracy_95_percent'][0]*100:.2f}%, {primary['group_bootstrap_accuracy_95_percent'][1]*100:.2f}%], conditional on this benchmark rather than future traffic.

These are supervised benchmark measurements. The 109M-parameter pretrained encoder remained frozen; only downstream classifiers/heads were trained locally. All 5,500 final examples were within the serving token limit; none were truncated. Final feature encoding took {final['encoding_seconds']:.1f} seconds on CPU.

## Descriptive fixed-model comparison

| Model | Known accuracy | Macro-F1 | Raw log loss |
| --- | ---: | ---: | ---: |
{rows}

The primary model was selected before this comparison. The larger head's weaker result remains visible. No model was retrained or replaced after observing these scores.

## Unsupported requests and active behavior

On **1,000 unsupported final requests**, the frozen proposed thresholds would accept **{proposed['accepted_oos']} ({proposed['oos_false_acceptance']*100:.2f}%)**. This is a descriptive offline measurement of the already-failed development policy, not a new threshold-selection or activation gate. Known-intent accuracy does not solve OOS detection.

**The active app still requires review for every request: automatic coverage 0%, actions executed 0.** A final score cannot override the earlier failed development gate. Future OOS research needs fresh, reviewed development and evaluation data; the consumed final set must not be retuned into a validation set while still claiming independent final performance.

![Frozen classifier results](figures/milestone_05_final.png)

This is crowdsourced English assistant-request data, not production customer traffic. Fingerprint grouping removes formatting overlap but cannot prove the absence of semantic or pretraining overlap. The bootstrap is an uncertainty description of this sample, not a real-world reliability guarantee. No generated-answer or business-impact evaluation was performed.

[Machine-readable final evidence](final_evaluation.json). Historical Milestones 1–4 record the state at those stages; their earlier “test not evaluated” statements are superseded by this milestone.
"""
(root/"reports/Milestone_05.md").write_text(report5, encoding="utf-8")
write_json(root/"reports/release_evidence.json", {
    "scope": "Current release assembled from preserved experiment reports. Nested flags describe what each operation evaluated; top-level flags describe current release state.",
    "classifier_final": final, "retrieval_development": {k:v for k,v in retrieval.items() if k != "rows"},
    "retrieval_index": index, "test_evaluated": True, "automatic_routing": False,
    "paid_services_required": False, "final_figure_sha256": sha256_file(figures/"milestone_05_final.png")})
print("Final/retrieval reports, graph and release evidence assembled without rerunning evaluation.")
