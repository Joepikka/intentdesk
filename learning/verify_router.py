"""Exercise the frozen router without tuning or reading held-out test data."""

from pathlib import Path
import time
import numpy as np
import psutil

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.policy import Router


def main():
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    router = Router(root)
    construction_seconds = time.perf_counter()-started
    examples = ["I lost my credit card", "What is my bank balance?",
                "Where is my delivery?", "Explain quantum entanglement and write a space opera"]
    for text in examples:
        router.route(text)
    timings, responses = [], []
    for i in range(40):
        response = router.route(examples[i % len(examples)])
        timings.append(response["elapsed_ms"])
        if i < len(examples):
            responses.append({"fixture_text": examples[i], "response": response})
        assert response["action_executed"] is False
        if not router.metadata["gate"]["passed"]:
            assert response["route_recommendation"] == "review_required"
    overlong = router.route("word " * 200)
    assert overlong["route_recommendation"] == "review_required" and not overlong["candidates"]
    for text in (None, "", " "*5, "a"*4001):
        try:
            router.route(text)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid request was accepted.")
    directory = root/"artifacts"/router.metadata["run_id"]
    binding = router.metadata["binding"]
    sources_match = all(sha256_file(root/"src/intentdesk"/name) == digest
                        for name, digest in binding["source_hashes"].items())
    inputs_match = all(sha256_file(root/"data/prepared"/(name+".jsonl")) == digest
                       for name, digest in binding["input_hashes"].items())
    assert sources_match and inputs_match
    result = {"policy_run": router.metadata["run_id"], "gate_passed": router.metadata["gate"]["passed"],
              "source_hashes_match": sources_match, "input_hashes_match": inputs_match,
              "constructor_seconds_excluding_python_imports": construction_seconds,
              "warm_local_requests": len(timings), "warmup_requests": len(examples),
              "warm_latency_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95))},
              "scope": "One persistent Router; sequential requests, CPU; excludes imports/model construction; not an API load test.",
              "process_peak_working_set_bytes": psutil.Process().memory_info().peak_wset,
              "overlong_request_reviewed_without_classification": not overlong["candidates"],
              "invalid_request_tests_passed": True, "test_evaluated": False, "fixture_results": responses}
    write_json(root/"reports/router_smoke.json", result)
    print({k: v for k, v in result.items() if k != "fixture_results"})


if __name__ == "__main__":
    main()
