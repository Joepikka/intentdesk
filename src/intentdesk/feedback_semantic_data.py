"""V4 allocations: untouched development/calibration/test product groups."""
from collections import Counter
import csv
import json
import re
import sqlite3

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY, load_partition
from intentdesk.feedback_lab import directory_for
from intentdesk import feedback_v2, feedback_v3


CONFIG = {"fit_rows": 20000, "development_rows": 5000,
          "calibration_rows": 5000, "test_rows": 20000, "human_audit_per_rating": 40}


def csv_display(text):
    """Keep untrusted review text from becoming an Excel/Calc formula."""
    return "'"+text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def previous_products(root):
    development = {r["product"] for r in load_partition(root, "development")}
    test = {r["product"] for r in load_partition(root, "test")}
    receipts = {}
    for name, directory in (("v1", directory_for(root)),
                            ("v2", feedback_v2.current_directory(root)),
                            ("v3", feedback_v3.current_directory(root))):
        marker = read_json(directory/"final_experiment.json")
        if marker["status"] != "completed":
            raise ValueError("Complete/preserve earlier evaluations before V4 allocation.")
        receipts[name] = sha256_file(directory/"final_experiment.json")
        if name != "v1":
            path = directory/"fresh_check.jsonl"
            expected = read_json(directory/"metadata.json")["protocol"]["fresh_check"]["sha256"]
            if sha256_file(path) != expected:
                raise ValueError("An earlier consumed sample changed.")
            test.update(r["product"] for r in read_records(path))
    return development, test, receipts


def connect(root):
    return sqlite3.connect((root/DIRECTORY/"audit.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)


def concurrent_reservation(root):
    """Exclude a separately running lexical experiment's reserved products too."""
    pointer = root/"artifacts/feedback_v4_latest.json"
    if not pointer.exists():
        return set(),None
    name = read_json(pointer)["run_id"]
    if not re.fullmatch(r"feedback_v4_[0-9TZ]+",name):
        raise ValueError("Invalid concurrent experiment path.")
    directory = root/"artifacts"/name
    check = read_json(directory/"protocol.json")["fresh_check"]
    path = directory/"fresh_check.jsonl"
    if sha256_file(path)!=check["sha256"]:
        raise ValueError("Concurrent reserved check changed.")
    products = {r["product"] for r in read_records(path)}
    return products,{"run_id":name,"sha256":check["sha256"],"products":len(products)}


def block_table(db, products):
    db.execute("CREATE TEMP TABLE blocked(product TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO blocked VALUES(?)", ((p,) for p in products))


def query_rows(db, split, limit):
    mask = {"fit": 1, "development": 2, "test": 4}[split]
    return [dict(zip(("id", "text", "rating", "product"), r)) for r in db.execute(
        """SELECT hash,text,rating,product FROM reviews r INDEXED BY sqlite_autoindex_reviews_1
        WHERE split=? AND ratings IN(1,2,4,8,16) AND splits=?
        AND NOT EXISTS(SELECT 1 FROM blocked b WHERE b.product=r.product)
        ORDER BY hash LIMIT ?""", (split, mask, limit))]


def audit_available(root):
    old_dev, old_test, receipts = previous_products(root)
    output = {"previous_development_products": len(old_dev), "previous_test_products": len(old_test),
              "previous_markers": receipts, "unused": {}}
    for split, blocked in (("development", old_dev), ("test", old_test)):
        with connect(root) as db:
            block_table(db, blocked)
            mask = 2 if split == "development" else 4
            rows, products = db.execute("""SELECT count(*),count(DISTINCT product) FROM reviews r
                WHERE split=? AND ratings IN(1,2,4,8,16) AND splits=?
                AND NOT EXISTS(SELECT 1 FROM blocked b WHERE b.product=r.product)""", (split, mask)).fetchone()
            output["unused"][split] = {"rows": rows, "products": products}
    return output


def reserve(root, directory):
    old_dev, old_test, receipts = previous_products(root)
    parallel,parallel_receipt = concurrent_reservation(root)
    with connect(root) as db:
        block_table(db, set())
        fit = query_rows(db, "fit", CONFIG["fit_rows"])
    with connect(root) as db:
        block_table(db, old_dev)
        development = query_rows(db, "development", CONFIG["development_rows"])
        new_dev_products = {r["product"] for r in development}
        db.executemany("INSERT OR IGNORE INTO blocked VALUES(?)", ((p,) for p in new_dev_products))
        calibration = query_rows(db, "development", CONFIG["calibration_rows"])
    with connect(root) as db:
        block_table(db, old_test | parallel)
        test = query_rows(db, "test", CONFIG["test_rows"])
    partitions = {"fit": fit, "development": development, "calibration": calibration, "test": test}
    groups = {n: {r["product"] for r in rows} for n, rows in partitions.items()}
    for a, ga in groups.items():
        for b, gb in groups.items():
            if a != b and ga & gb:
                raise ValueError("V4 product groups overlap.")
    if (groups["development"] | groups["calibration"]) & old_dev or groups["test"] & (old_test | parallel):
        raise ValueError("V4 reused previously exposed development/test products.")
    receipt = {"previous_development_products": len(old_dev), "previous_test_products": len(old_test),
               "concurrent_reservation":parallel_receipt,"concurrent_product_overlap":0,
               "previous_markers": receipts, "partitions": {}, "product_overlap": 0,
               "selection": "First eligible review hashes; natural class distribution; all fit candidates share identical rows",
               "limitations": "Excluding used groups shifts toward less-reviewed products; not user/temporal/semantic-near-duplicate independence"}
    for name, rows in partitions.items():
        if len(rows) != CONFIG[name+"_rows"] or {r["rating"] for r in rows} != {1,2,3,4,5}:
            raise ValueError("Insufficient unused support for the fixed V4 protocol.")
        path = directory/(name+".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        receipt["partitions"][name] = {"rows": len(rows), "products": len(groups[name]),
            "rating_counts": dict(Counter(r["rating"] for r in rows)), "sha256": sha256_file(path)}
    # This is a blank human exercise, never model training or measured human data.
    audit = []
    for rating in range(1,6):
        audit.extend([r for r in fit if r["rating"] == rating][:CONFIG["human_audit_per_rating"]])
    audit.sort(key=lambda r: r["id"])
    with (directory/"human_audit_blind.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "review_text", "human_rating", "ambiguous", "reason", "reviewer"])
        writer.writerows([r["id"], csv_display(r["text"]), "", "", "", ""] for r in audit)
    with (directory/"human_audit_reference.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "source_rating"])
        writer.writerows([r["id"], r["rating"]] for r in audit)
    write_json(directory/"human_audit_status.json", {"prepared": len(audit), "completed_human_rows": 0,
        "scope": "Fit-only blinded exercise; no human labels fabricated; not an evaluation-label rewrite"})
    return receipt


def load(directory, name, receipt):
    path = directory/(name+".jsonl")
    if sha256_file(path) != receipt["partitions"][name]["sha256"]:
        raise ValueError("V4 reserved partition changed.")
    return read_records(path)
