# Publication verification scope

On 10 October 2026 the local full suite passed **97 Python tests** and **2 JavaScript tests**. The suites use synthetic fixtures: they check split hygiene, training/backpropagation on toy data, cache/input contracts, frozen experiment protection, calibration, scope features, request boundaries, review history, stale saves, retrieval behavior and pilot allocation/measurement rules. They do not establish customer benefit.

The actual locally trained model and browser review flow were also checked. An explicitly marked automated pilot session saved 14 decisions and prepared local handoffs; manual trials hid suggestions and repeated saves returned 409. Completed human comparisons remained **zero**. Automated decisions and elapsed times are excluded from human-effect metrics.

The wheel was built and source-byte parity checked for Python entry points/modules and seven browser assets. Pilot screens were inspected at 1280px, 1920px and 390px. These are bounded implementation/interface checks, not a security certification or production load test.

Private local verification captures, session IDs, machine runtime logs and user-review databases are excluded from the public repository. The repository includes fictional screenshots, aggregate benchmark reports and reproducible tests. GitHub Actions adds fresh source-test evidence when the workflow runs; its status must be checked rather than inferred from these local results.

No original final classifier or scope check was rerun for publication. Their existing bound results were reused. Preserve their consumed-evaluation status and do not tune against published outcomes.

## Robustness addition: 12 October 2026

The full local suite passed **108 Python tests** (including 11 robustness tests) and **2 JavaScript tests**. One existing Starlette/httpx TestClient deprecation warning remains. New checks cover exact case/hash matching for reviewed labels, blank template fields, overwrite protection, confusion-matrix orientation, runtime failures, review-only contract violations and separate report directories. The end-to-end fixture produces actual report/CSV/PNG files without model downloads.

The actual CLI created a blank 60-case label sheet with zero reviewed labels and then tested the trained local model. It completed 60 requests with zero runtime errors or unavailable-scope results. Agreement with assistant-authored provisional labels was **38/60**, with **0/12** clarification cases matched and **2/12** outside cases suggested as supported. These outcomes are retained in the [aggregate snapshot](../reports/robustness_v1/report.md), not treated as passing model criteria. Sequential request p50/p95 were about **137/183 ms**, excluding model startup and warm-up; this is a single local CPU pass, not a load benchmark.

Before/after hashes of 16 protected source, lock, experiment-marker and report files matched. This includes original classifier/scope code, pilot/application sources, consumed experiment markers and the existing human comparison report. No model was trained, threshold changed, consumed benchmark reopened or human session created. Dynamic prediction exports, the blank review sheet and reviewer notes remain local. The published snapshot contains the aggregate JSON, graph and report from the actual run.

The updated wheel built successfully; its CLI entry point and new robustness module match source bytes. The three published snapshot files match the actual run bytes. The graph was visually inspected. Publication scanning found no credential/generated-database issues or broken local documentation links among 117 candidate files (about 2.04 MB).
