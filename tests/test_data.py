from copy import deepcopy

import pytest

from intentdesk.data import assign_partitions, build_records, fingerprint, validate_official_contract


def fixture_data():
    domains = {"demo": ["delivery", "password", "balance"]}
    payload = {k: [] for k in ["train", "val", "test", "oos_train", "oos_val", "oos_test"]}
    for label in domains["demo"]:
        payload["train"].extend([[f"training request {label} number {i}", label] for i in range(20)])
        payload["val"].append([f"validation request {label}", label])
        payload["test"].append([f"final request {label}", label])
    for source in ["oos_train", "oos_val", "oos_test"]:
        payload[source].append([f"unsupported request {source}", "oos"])
    return payload, domains


def test_partition_is_deterministic_and_groups_do_not_leak():
    payload, domains = fixture_data()
    a = build_records(payload, domains)
    b = deepcopy(a)
    assert assign_partitions(a) == assign_partitions(b)
    assert a == b
    groups = {}
    for row in a:
        if row["working_split"] not in {"excluded", "oos_development"}:
            groups.setdefault(row["duplicate_group"], set()).add(row["working_split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert {r["intent"] for r in a if r["working_split"] == "fit"} == set(domains["demo"])


def test_test_is_preserved_and_overlap_is_removed_from_development():
    payload, domains = fixture_data()
    payload["test"].append(["MY shared Request!", "delivery"])
    payload["train"].append(["my shared request", "delivery"])
    payload["val"].append(["My shared request.", "delivery"])
    rows = build_records(payload, domains)
    assign_partitions(rows)
    shared = [r for r in rows if r["duplicate_group"] == fingerprint("my shared request")]
    assert {r["working_split"] for r in shared if r["source_split"] == "test"} == {"locked_test"}
    assert all(r["working_split"] == "excluded" for r in shared if r["source_split"] != "test")
    assert len([r for r in rows if r["working_split"] == "locked_test"]) == len(payload["test"]) + len(payload["oos_test"])


def test_conflicting_training_labels_are_excluded_not_arbitrarily_selected():
    payload, domains = fixture_data()
    payload["train"].extend([["same ambiguous request", "delivery"], ["Same ambiguous request!", "balance"]])
    rows = build_records(payload, domains)
    audit = assign_partitions(rows)
    assert audit["exclusions"]["conflicting_training_labels"] == 2


def test_oos_never_enters_fit_tune_or_calibration():
    payload, domains = fixture_data()
    rows = build_records(payload, domains)
    assign_partitions(rows)
    assert all(not row["is_oos"] for row in rows if row["working_split"] in {"fit", "tune", "calibrate"})


@pytest.mark.parametrize("pair", [["", "delivery"], [12, "delivery"], ["hello", "unknown"], ["hello"]])
def test_invalid_records_fail_explicitly(pair):
    payload, domains = fixture_data()
    payload["train"][0] = pair
    with pytest.raises(ValueError):
        build_records(payload, domains)


def test_official_contract_rejects_wrong_counts():
    payload, domains = fixture_data()
    with pytest.raises(ValueError, match="split counts"):
        validate_official_contract(build_records(payload, domains))

