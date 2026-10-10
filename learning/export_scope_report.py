"""Present preserved scope results without exposing or rerunning check data."""
from pathlib import Path

import matplotlib.pyplot as plt

from intentdesk.common import read_json, write_json, sha256_file

root = Path(__file__).resolve().parents[1]
r = read_json(root/"reports/scope_evaluation.json")
audit = read_json(root/"reports/scope_data_audit.json")
metadata = read_json(root/"artifacts"/r["run_id"]/"metadata.json")
m, b = r["check"], r["old_thresholds_comparison"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
names = ["Old thresholds", "New advisory"]
axes[0].bar(names, [b['fresh_oos_supported_candidates'], m['fresh_oos_supported_candidates']], color=['#83988b','#146146'])
axes[0].set(title='Fresh OOS check: 100 requests', ylabel='Incorrectly suggested as supported', ylim=(0,100))
for i, value in enumerate([b['fresh_oos_supported_candidates'], m['fresh_oos_supported_candidates']]):
    axes[0].text(i, value+2, str(value), ha='center')
values = [100*b['known_supported_candidates']/m['known_count'], 100*m['known_supported_candidates']/m['known_count']]
axes[1].bar(names, values, color=['#83988b','#146146'])
axes[1].set(title='Known check: reused calibration data', ylabel='Retained as supported candidates (%)', ylim=(0,100))
for i, value in enumerate(values): axes[1].text(i, value+1, f'{value:.2f}%', ha='center')
fig.suptitle('Frozen scope experiment; every request still requires review', fontsize=12)
fig.tight_layout()
path = root/'reports/figures/milestone_06_scope.png'
fig.savefig(path, dpi=160); plt.close(fig)
text = f'''# Milestone 6: separately trained scope advisory

Completed 10 October 2026. This release adds an advisory detector for unsupported requests. It leaves the original classifier, failed routing gate and consumed final evaluation unchanged. **Every request still requires human review; no automatic action occurs.**

## Data and method

The unused original 100 OOS training examples passed provenance and fingerprint-overlap checks. The pinned CLINC `data_oos_plus.json` contains 250 OOS training rows; 99 match existing fingerprints and are excluded from the new allocation. The remaining **151 previously unused OOS examples** are split deterministically into 51 for threshold selection and 100 for checking. They retain public crowd-authored labels; no new independent semantic adjudication is claimed.

The original fit set is split into **{metadata['reference_rows']:,} reference examples** and **{metadata['known_training_rows']:,} known examples for the binary detector**. A query's own vector is excluded from the nearest-neighbor reference during detector training. Six features combine nearest-reference cosine, top-five mean cosine, maximum prototype cosine, known-class maximum probability, top-two margin and normalized entropy. StandardScaler and balanced logistic regression (`C=1`) fit on the detector training allocation only. The 109M-parameter encoder and original intent classifier remain frozen.

Known threshold-selection data reuse the original tune set. The known check contains **{m['known_count']:,} reused calibration examples**, previously used to fit the original temperature. Thus this is **not a fresh independent joint known/OOS test**. The 100 OOS check examples are new to this project's model/threshold selection, but come from the same public benchmark family.

Protocol, source/data/code/model hashes and thresholds were frozen before new check inference. The new check is now consumed. Completed commands reuse the bound result and refuse changed bindings or reports. Do not adjust this model or its thresholds using these check outcomes.

## Measured outcome

| Outcome on the same check examples | Old frozen thresholds | New frozen advisory |
| --- | ---: | ---: |
| Fresh OOS suggested as supported | {b['fresh_oos_supported_candidates']}/100 | {m['fresh_oos_supported_candidates']}/100 |
| Known retained as supported candidates (reused data) | {b['known_supported_candidates']}/{m['known_count']} | {m['known_supported_candidates']}/{m['known_count']} |

The new detector flags **{m['fresh_oos_flagged_likely_oos']}/100 fresh OOS examples** as likely unsupported, and incorrectly flags **{m['known_incorrectly_flagged_oos']}/{m['known_count']} known requests**. There are {m['fresh_oos_uncertain']} OOS examples in the middle uncertainty band in this run; its two selected thresholds coincide. A user can still choose `uncertain` manually.

The one-sided 95% exact upper bound on OOS false acceptance is **{100*m['oos_acceptance_upper_95']:.2f}%**. This exceeds the old 5% target even though the observed rate is 3%. Combined with the reused known check, that rules out treating this as authorization for automatic routing. The internal support score is uncalibrated and is not displayed as a probability of being supported.

![Scope results](figures/milestone_06_scope.png)

## Application behavior

Likely unsupported requests display an explicit explanation, hide misleading known-label suggestions and prefill `oos` as an editable review suggestion. Saving remains a separate user action. Supported candidates retain their label shortlist and require confirmation. The original motorcycle example was a visible development smoke case, not part of the held-out benchmark; the running app now flags it as likely unsupported.

Older records preserve their original stored analysis. **Analyze again** creates a new queue entry; it never rewrites old analysis or decisions. The screen identifies earlier assessments and avoids presenting their old technical gate warning as a current processing error. Review feedback does not train a model.

## Reproduce and learn

```powershell
uv run --locked --extra neural intentdesk prepare-scope
uv run --locked --extra neural intentdesk train-scope
uv run --locked --extra neural intentdesk evaluate-scope
uv run --locked --extra neural --extra web intentdesk serve
```

Read [the scope lesson](../learning/Scope_Detection.md), [data audit](scope_data_audit.json) and [frozen evaluation](scope_evaluation.json). These results demonstrate a bounded improvement to a research advisory, not reliable rejection of every possible unsupported query. Fresh production-like data, independent review and a new untouched joint evaluation are required before proposing automatic routing. Buying a motorcycle and open-ended general assistance remain outside this application's capabilities.
'''
(root/'reports/Milestone_06.md').write_text(text, encoding='utf-8')
write_json(root/'reports/scope_release.json', {'run_id':r['run_id'], 'check':m,
    'old_thresholds_comparison':b, 'automatic_actions':False, 'old_classifier_final_unchanged':True,
    'fresh_independent_joint_evaluation':False, 'data_audit_sha256':sha256_file(root/'reports/scope_data_audit.json'),
    'evaluation_sha256':sha256_file(root/'reports/scope_evaluation.json'), 'figure_sha256':sha256_file(path)})
print('Scope report, graph and release evidence exported from the frozen result.')
