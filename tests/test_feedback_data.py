import json
import sqlite3

import pytest

from intentdesk.feedback_data import ingest, product_split, text_key, select_rows, clean_rating_declarations


def products():
    result = {}
    for i in range(1000):
        name = f"fixture_product_{i}"; result.setdefault(product_split(name),name)
    return result


def test_product_assignment_is_stable_and_covers_three_partitions():
    values = products()
    assert set(values) == {"fit","development","test"}
    assert all(product_split(name)==split for split,name in values.items())
    assert text_key("Really GREAT!!") == text_key("really great")
    with pytest.raises(ValueError):
        product_split("")


def test_explicit_star_shortcuts_are_removed_without_using_target_labels():
    assert clean_rating_declarations("Five Stars. Works well, 5 out of 5.")==". Works well, ."
    assert clean_rating_declarations("1-star item; 2/5 rating")=="item; rating"


def test_disk_audit_excludes_conflicting_and_cross_partition_duplicates(tmp_path):
    groups = products()
    def row(text,rating,product):
        return {"title":"fixture", "text":text, "rating":rating, "parent_asin":product,
                "user_id":"must_not_enter_derived_data"}
    rows = [row("a sufficiently long duplicate text",1,groups["fit"]),
            row("a sufficiently long duplicate text",1,groups["test"]),
            row("a conflicting rating example",2,groups["fit"]),
            row("a conflicting rating example",5,groups["fit"]),
            row("a valid training example for audit",3,groups["fit"]),
            row("a distinct valid development example",4,groups["development"]),
            row("a distinct valid final test example",5,groups["test"]),
            row("invalid rating row data",10,groups["fit"])]
    path = tmp_path/"source.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    db_path = tmp_path/"audit.sqlite3"
    result = ingest([path],db_path)
    assert result["source_rows"]==8 and result["invalid_rows"]==1
    assert result["cross_partition_duplicate_groups"]==1 and result["conflicting_rating_groups"]==1
    with sqlite3.connect(db_path) as db:
        fit, development, test = (select_rows(db,s,20) for s in ("fit","development","test"))
        assert [r["rating"] for r in fit]==[3]
        assert [r["rating"] for r in development]==[4]
        assert [r["rating"] for r in test]==[5]
        fields = [r[1] for r in db.execute("PRAGMA table_info(reviews)")]
        assert "user_id" not in fields
    with pytest.raises(ValueError,match="already exists"):
        ingest([path],db_path)
