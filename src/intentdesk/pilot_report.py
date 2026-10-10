"""Export recorded pilot outcomes. Never fabricate participants or run a trial."""

from pathlib import Path
import json
from urllib.request import urlopen

from intentdesk.common import write_json


def export_report(root: Path, port=8765):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a local app port between 1024 and 65535.")
    with urlopen(f"http://127.0.0.1:{port}/api/pilot/report", timeout=30) as response:
        report = json.load(response)
    write_json(root/"reports/pilot_comparison.json", report)
    lines = ["# Card-support demonstration comparison", "",
             f"Generated from recorded local sessions: {report['generated_at']}", "",
             "**Demonstration data; business value is not established.** No permissioned customer traffic or independent label adjudication is present.", "",
             f"Completed human sessions: {report['completed_human_sessions']}. Incomplete human sessions: {report['incomplete_human_sessions']}. Automated verification sessions excluded: {report['smoke_sessions_excluded']}.", "",
             f"Frozen study binding: `{report['run_id']}`. Protocol, dataset, workflow, implementation and model pointers are recorded in the adjacent JSON.", ""]
    if report["completed_human_sessions"]:
        lines += ["| Measure | Manual | Assisted |", "| --- | ---: | ---: |"]
        for name, key in [("Decisions", "count"), ("Correct", "correct"), ("Accuracy", "accuracy"),
                          ("Mean decision seconds", "mean_decision_seconds"), ("Timing exclusions", "timing_exclusions"),
                          ("Uncertain decisions", "uncertain"), ("Suggestion overrides", "suggestion_overrides")]:
            values = [report["conditions"][c][key] for c in ("manual", "assisted")]
            display = ["Not available" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
            lines.append(f"| {name} | {' | '.join(display)} |")
        lines += ["", "Intervals in the JSON resample reviewers, not individual tickets. They are exploratory, require at least two completed reviewers, and do not establish a powered quality-preserving speed improvement."]
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        conditions = [report["conditions"][c] for c in ("manual", "assisted")]
        axes[0].bar(["Manual", "Assisted"], [100*c["accuracy"] for c in conditions], color=["#83988b", "#146146"])
        axes[0].set(ylabel="Strict category accuracy (%)", ylim=(0, 100))
        if all(c["mean_decision_seconds"] is not None for c in conditions):
            axes[1].bar(["Manual", "Assisted"], [c["mean_decision_seconds"] for c in conditions], color=["#83988b", "#146146"])
            axes[1].set(ylabel="Mean decision seconds (timed tickets)")
        else:
            axes[1].text(.5, .5, "No eligible timing comparison", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_axis_off()
        fig.suptitle(f"Synthetic tickets · {report['completed_human_sessions']} completed human sessions · exploratory")
        fig.tight_layout(); (root/"reports/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(root/"reports/figures/pilot_comparison.png", dpi=160); plt.close(fig)
        lines += ["", "![Recorded demonstration outcomes](figures/pilot_comparison.png)"]
    else:
        lines += ["No completed human comparison exists yet. Accuracy, speed improvement and human-effect graphs are therefore unavailable. Automated test times are excluded rather than presented as human performance."]
    lines += ["", "Run the local app, open `/pilot`, read the category guide, and complete a comparison with one reviewer code per actual person. Keep breaks marked. Re-run `intentdesk pilot-report` to export recorded outcomes. Do not complete human sessions programmatically.", "",
              "For a real business claim: obtain permissioned de-identified tickets, define representative categories and team ownership with the business, obtain independent labels, pre-register a powered comparison and validate on an untouched sample. None of those steps is replaced by this demonstration."]
    (root/"reports/Pilot_Comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"report": str(root/"reports/Pilot_Comparison.md"), "human_sessions": report["completed_human_sessions"],
            "smoke_sessions_excluded": report["smoke_sessions_excluded"], "business_value_established": False}
