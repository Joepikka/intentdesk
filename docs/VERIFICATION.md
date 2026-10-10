# Publication verification scope

On 10 October 2026 the local full suite passed **97 Python tests** and **2 JavaScript tests**. The suites use synthetic fixtures: they check split hygiene, training/backpropagation on toy data, cache/input contracts, frozen experiment protection, calibration, scope features, request boundaries, review history, stale saves, retrieval behavior and pilot allocation/measurement rules. They do not establish customer benefit.

The actual locally trained model and browser review flow were also checked. An explicitly marked automated pilot session saved 14 decisions and prepared local handoffs; manual trials hid suggestions and repeated saves returned 409. Completed human comparisons remained **zero**. Automated decisions and elapsed times are excluded from human-effect metrics.

The wheel was built and source-byte parity checked for Python entry points/modules and seven browser assets. Pilot screens were inspected at 1280px, 1920px and 390px. These are bounded implementation/interface checks, not a security certification or production load test.

Private local verification captures, session IDs, machine runtime logs and user-review databases are excluded from the public repository. The repository includes fictional screenshots, aggregate benchmark reports and reproducible tests. GitHub Actions adds fresh source-test evidence when the workflow runs; its status must be checked rather than inferred from these local results.

No original final classifier or scope check was rerun for publication. Their existing bound results were reused. Preserve their consumed-evaluation status and do not tune against published outcomes.
