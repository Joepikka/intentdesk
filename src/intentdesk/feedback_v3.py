"""Bounded fit-only IDF experiment; all earlier feedback releases stay frozen."""
from collections import Counter
from datetime import datetime, timezone
import html
import json
import re
import sqlite3
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY, load_partition
from intentdesk.feedback_lab import checked_metadata, directory_for
from intentdesk.feedback_models import classification
from intentdesk import feedback_v2 as previous


CONFIG = {
    "version": "feedback_v3", "seed": 20261014, "features_per_view": 2**18,
    "word_ngrams": [1, 2], "character_ngrams": [3, 5],
    "alpha": 1e-6, "batch_size": 256, "threads": 2,
    "maximum_epochs": 3, "minimum_epochs": 2, "patience": 1,
    "weight_exponents": {"idf_balanced": 1., "idf_moderate": .5},
    "fresh_check_rows": 20000, "minimum_macro_f1_gain": .005,
    "selection": "Development macro-F1, accuracy tie-break, earliest epoch; retain V2 unless gain >= .005",
    "scope": "Same five ratings; no change of labels, sample difficulty, or promised 90%",
}
CONTROL = "previous_word_char_balanced"


def clean_markup(text):
    # Formatting is not customer sentiment. Decode before stripping encoded tags.
    text = html.unescape(text)
    text = re.sub(r"\[\[VIDEOID:[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    return " ".join(text.casefold().split())


def counts(texts):
    common = dict(n_features=CONFIG["features_per_view"], alternate_sign=False,
                  norm=None, dtype=np.float32, preprocessor=clean_markup)
    word = HashingVectorizer(**common, ngram_range=tuple(CONFIG["word_ngrams"])).transform(texts)
    char = HashingVectorizer(**common, analyzer="char_wb",
                             ngram_range=tuple(CONFIG["character_ngrams"])).transform(texts)
    x = hstack([word, char], format="csr", dtype=np.float32)
    x.sum_duplicates()
    return x


def document_frequency(x):
    # Hash collisions have already been summed within each row: count documents,
    # not term occurrences. IDF lives in hash space and collisions remain a limit.
    return np.bincount(x.indices, minlength=x.shape[1]).astype(np.int64)


def idf_features(x, idf):
    x = x.copy()
    np.log(x.data, out=x.data)
    x.data += np.float32(1.)
    x.data *= idf[x.indices]
    return normalize(x, norm="l2", copy=False)


def fit_rows(root, after_key="", limit=None):
    """Read hash order without sorting millions of full review strings on disk."""
    database = (root / DIRECTORY / "audit.sqlite3").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(database, uri=True) as db:
        query = """SELECT hash,text,rating,product FROM reviews
            INDEXED BY sqlite_autoindex_reviews_1
            WHERE hash>? AND split='fit' AND ratings IN(1,2,4,8,16) AND splits=1
            ORDER BY hash"""
        args = [after_key]
        if limit is not None:
            query += " LIMIT ?"
            args.append(limit)
        cursor = db.execute(query, args)
        while batch := cursor.fetchmany(CONFIG["batch_size"]):
            yield [dict(zip(("id", "text", "rating", "product"), r)) for r in batch]


def binding(root):
    old = directory_for(root)
    v2 = previous.current_directory(root)
    names = ("feedback_v3.py", "feedback_v2.py", "feedback_data.py",
             "feedback_models.py", "feedback_lab.py", "common.py")
    return {"sources": {n: sha256_file(root/"src/intentdesk"/n) for n in names},
            "manifest": sha256_file(root/DIRECTORY/"manifest.json"),
            "lock": sha256_file(root/"uv.lock"), "config": CONFIG,
            "previous_metadata": sha256_file(v2/"metadata.json"),
            "previous_final_marker": sha256_file(v2/"final_experiment.json"),
            "original_metadata": sha256_file(old/"metadata.json"),
            "original_final_marker": sha256_file(old/"final_experiment.json")}


def current_directory(root):
    name = read_json(root/"artifacts/feedback_v3_latest.json")["run_id"]
    if not re.fullmatch(r"feedback_v3_[0-9TZ]+", name):
        raise ValueError("Invalid V3 experiment path.")
    return root/"artifacts"/name


def save_checkpoint(directory, state):
    path = directory/"resume.joblib"
    temporary = path.with_suffix(".tmp")
    joblib.dump(state, temporary)
    temporary.replace(path)
    write_json(directory/"resume.json", {"sha256": sha256_file(path),
        "phase": state["phase"], "epoch": state.get("epoch", 0), "seen": state["seen"]})


def reserve_check(root, directory):
    # Previously consumed review text is opened only to exclude entire products.
    blocked = {r["product"] for r in load_partition(root, "test")}
    old_fresh = previous.current_directory(root)/"fresh_check.jsonl"
    old_metadata = read_json(previous.current_directory(root)/"metadata.json")
    if sha256_file(old_fresh) != old_metadata["protocol"]["fresh_check"]["sha256"]:
        raise ValueError("Previous consumed check changed.")
    blocked.update(r["product"] for r in read_records(old_fresh))
    database = (root/DIRECTORY/"audit.sqlite3").resolve().as_uri()+"?mode=ro"
    with sqlite3.connect(database, uri=True) as db:
        db.execute("CREATE TEMP TABLE blocked(product TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO blocked VALUES(?)", ((p,) for p in blocked))
        rows = [dict(zip(("id", "text", "rating", "product"), r)) for r in db.execute(
            """SELECT hash,text,rating,product FROM reviews r INDEXED BY sqlite_autoindex_reviews_1
            WHERE split='test' AND ratings IN(1,2,4,8,16) AND splits=4
            AND NOT EXISTS(SELECT 1 FROM blocked b WHERE b.product=r.product)
            ORDER BY hash LIMIT ?""", (CONFIG["fresh_check_rows"],))]
    if len(rows) != CONFIG["fresh_check_rows"] or blocked & {r["product"] for r in rows}:
        raise ValueError("Insufficient untouched products or product overlap.")
    path = directory/"fresh_check.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    return {"rows": len(rows), "products": len({r["product"] for r in rows}),
            "blocked_previous_products": len(blocked), "previous_product_overlap": 0,
            "sha256": sha256_file(path), "rating_counts": dict(Counter(r["rating"] for r in rows)),
            "selection": "First eligible hashes excluding every product in both consumed checks; unbalanced",
            "population_limit": "Further exclusion shifts toward less-reviewed products; compare models only on this same check"}


def prepare(root):
    identity = binding(root)
    pointer = root/"artifacts/feedback_v3_latest.json"
    if pointer.exists():
        directory = current_directory(root)
        if read_json(directory/"protocol.json")["binding"] != identity:
            raise ValueError("V3 source/data/config changed; preserve this run and use a new version.")
        return directory
    baseline = checked_metadata(root)
    _, v2_metadata = previous.checked(root)
    if v2_metadata["selected_classifier"] != "word_char_balanced":
        raise ValueError("V3 protocol requires the recorded preselected V2 control.")
    if read_json(previous.current_directory(root)/"final_experiment.json")["status"] != "completed":
        raise ValueError("Previous release must be completed before this experiment.")
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != baseline["data"]["audit_sha256"]:
        raise ValueError("Audited database changed.")
    name = "feedback_v3_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root/"artifacts"/name
    directory.mkdir(parents=True, exist_ok=False)
    check = reserve_check(root, directory)
    write_json(directory/"protocol.json", {"binding": identity, "fresh_check": check,
        "fit_rows": sum(baseline["data"]["available_by_partition_and_rating"]["fit"].values()),
        "hypothesis": "Remove formatting noise, fit IDF on all fit rows, and test moderate versus full inverse-frequency weighting with bounded passes",
        "causal_limit": "Bundled changes test a practical alternative; they do not isolate each cause",
        "selection_partition": "Existing development data, already exposed; fresh products never select model or epochs",
        "candidates": [CONTROL]+list(CONFIG["weight_exponents"]),
        "no_neural_extension": "Saved neural development loss deteriorated at later epochs"})
    write_json(pointer, {"run_id": name})
    return directory


def predict_model(model, idf, texts):
    output = []
    with threadpool_limits(limits=CONFIG["threads"]):
        for start in range(0, len(texts), CONFIG["batch_size"]):
            output.extend(model.predict(idf_features(counts(texts[start:start+CONFIG["batch_size"]]), idf)).tolist())
    return output


def predict_control(root, texts):
    # Load the exact preselected V2 model; no use of previous final predictions.
    models = joblib.load(previous.current_directory(root)/"models.joblib")
    output = []
    with threadpool_limits(limits=CONFIG["threads"]):
        for start in range(0, len(texts), CONFIG["batch_size"]):
            output.extend(models["word_char_balanced"].predict(previous.features(texts[start:start+CONFIG["batch_size"]])).tolist())
    return output


def metrics(rows, predicted):
    y = np.array([r["rating"] for r in rows])
    delta = np.abs(y-np.asarray(predicted))
    return {**classification(y, predicted), "adjacent_errors": int((delta==1).sum()),
            "all_errors": int((delta>0).sum()), "mean_absolute_star_error": float(delta.mean())}


def score(m):
    return (m["macro_f1"], m["accuracy"])


def selection(results):
    best = sorted(results, key=lambda n: (-results[n]["macro_f1"], -results[n]["accuracy"], n))[0]
    return best if results[best]["macro_f1"]-results[CONTROL]["macro_f1"] >= CONFIG["minimum_macro_f1_gain"] else CONTROL


def train(root):
    directory = prepare(root)
    protocol = read_json(directory/"protocol.json")
    if (directory/"metadata.json").exists():
        return checked(root)[1]
    total = protocol["fit_rows"]
    manifest = read_json(root/DIRECTORY/"manifest.json")
    class_counts = {int(k): v for k, v in manifest["available_by_partition_and_rating"]["fit"].items()}
    state = {"phase": "idf", "seen": 0, "last": "", "df": np.zeros(2*CONFIG["features_per_view"], dtype=np.int64)}
    if (directory/"resume.json").exists():
        if sha256_file(directory/"resume.joblib") != read_json(directory/"resume.json")["sha256"]:
            raise ValueError("V3 resume checkpoint changed.")
        state = joblib.load(directory/"resume.joblib")
    if state["phase"] == "idf":
        for index, batch in enumerate(fit_rows(root, state["last"])):
            x = counts([r["text"] for r in batch])
            state["df"] += document_frequency(x)
            state["seen"] += len(batch)
            state["last"] = batch[-1]["id"]
            if index % 256 == 0:
                save_checkpoint(directory, state)
                print(f"Fit-only IDF: {state['seen']:,}/{total:,}", flush=True)
        if state["seen"] != total or np.any(state["df"] > total):
            raise ValueError("IDF fit count or document frequency is invalid.")
        idf = (np.log((1+total)/(1+state["df"]))+1).astype(np.float32)
        joblib.dump(idf, directory/"idf.joblib")
        state = {"phase": "training", "epoch": 1, "seen": 0, "last": "", "history": [],
                 "best": {}, "stale": 0, "idf_sha256": sha256_file(directory/"idf.joblib"),
                 "models": {n: SGDClassifier(loss="log_loss", alpha=CONFIG["alpha"], average=True,
                     random_state=CONFIG["seed"]) for n in CONFIG["weight_exponents"]}}
        save_checkpoint(directory, state)
    if sha256_file(directory/"idf.joblib") != state["idf_sha256"]:
        raise ValueError("Fit-only IDF checkpoint changed.")
    idf = joblib.load(directory/"idf.joblib")
    development = load_partition(root, "development")
    if "control" not in state:
        state["control"] = metrics(development, predict_control(root, [r["text"] for r in development]))
        save_checkpoint(directory, state)
    started = time.perf_counter()
    while (state["epoch"] <= CONFIG["maximum_epochs"] and not
           (state["epoch"]-1 >= CONFIG["minimum_epochs"] and state["stale"] >= CONFIG["patience"])):
        epoch = state["epoch"]
        for index, batch in enumerate(fit_rows(root, state["last"])):
            y = np.array([r["rating"] for r in batch])
            with threadpool_limits(limits=CONFIG["threads"]):
                x = idf_features(counts([r["text"] for r in batch]), idf)
                for name, exponent in CONFIG["weight_exponents"].items():
                    # Normalize average weighting to one under fit priors so
                    # exponent comparisons do not silently change loss scale.
                    raw = {k: (total/(5*v))**exponent for k, v in class_counts.items()}
                    mean = sum(class_counts[k]/total*raw[k] for k in raw)
                    weights = np.array([raw[int(v)]/mean for v in y])
                    state["models"][name].partial_fit(x, y, classes=np.arange(1, 6), sample_weight=weights)
            state["seen"] += len(batch)
            state["last"] = batch[-1]["id"]
            if index % 256 == 0:
                save_checkpoint(directory, state)
                print(f"V3 epoch {epoch}/{CONFIG['maximum_epochs']}: {state['seen']:,}/{total:,} fit reviews", flush=True)
        if state["seen"] != total:
            raise ValueError("V3 training pass count changed.")
        improved = False
        epoch_metrics = {}
        for name, model in state["models"].items():
            result = metrics(development, predict_model(model, idf, [r["text"] for r in development]))
            epoch_metrics[name] = result
            if name not in state["best"] or score(result) > score(state["best"][name]["metrics"]):
                model_path = directory/(name+"_best.joblib")
                temporary = model_path.with_suffix(".tmp")
                joblib.dump(model, temporary)
                temporary.replace(model_path)
                state["best"][name] = {"epoch": epoch, "metrics": result, "sha256": sha256_file(model_path)}
                improved = True
            print(f"{name} epoch {epoch}: development accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
        state["history"].append({"epoch": epoch, "development": epoch_metrics})
        state["stale"] = 0 if improved else state["stale"]+1
        state.update(epoch=epoch+1, seen=0, last="")
        save_checkpoint(directory, state)
        write_json(directory/"history.json", state["history"])
        if epoch >= CONFIG["minimum_epochs"] and state["stale"] >= CONFIG["patience"]:
            break
    results = {CONTROL: state["control"], **{n: b["metrics"] for n, b in state["best"].items()}}
    selected = selection(results)
    fit_sample = [r for batch in fit_rows(root, limit=10000) for r in batch]
    fit_metrics = {CONTROL: metrics(fit_sample, predict_control(root, [r["text"] for r in fit_sample]))}
    for name, best in state["best"].items():
        path = directory/(name+"_best.joblib")
        if sha256_file(path) != best["sha256"]:
            raise ValueError("Best V3 checkpoint changed.")
        fit_metrics[name] = metrics(fit_sample, predict_model(joblib.load(path), idf, [r["text"] for r in fit_sample]))
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != manifest["audit_sha256"]:
        raise ValueError("Audited fit database changed during V3 training.")
    metadata = {"run_id": directory.name, "binding": protocol["binding"], "protocol": protocol,
        "development": results, "history": state["history"], "best": state["best"],
        "fit_diagnostic_first_10000": fit_metrics, "selected_classifier": selected,
        "fit_rows_per_pass": total, "completed_epochs": len(state["history"]),
        "idf_fit_rows": total, "idf_sha256": state["idf_sha256"],
        "training_seconds_this_invocation_excludes_idf": time.perf_counter()-started,
        "test_exposed": False, "automatic_actions": False, "target_accuracy_guaranteed": False}
    write_json(directory/"metadata.json", metadata)
    write_json(root/"artifacts/feedback_v3_latest.json", {"run_id": directory.name,
        "metadata_sha256": sha256_file(directory/"metadata.json")})
    return metadata


def checked(root):
    directory = current_directory(root)
    pointer = read_json(root/"artifacts/feedback_v3_latest.json")
    if sha256_file(directory/"metadata.json") != pointer.get("metadata_sha256"):
        raise ValueError("V3 training incomplete or metadata changed.")
    metadata = read_json(directory/"metadata.json")
    if metadata["binding"] != binding(root) or sha256_file(directory/"idf.joblib") != metadata["idf_sha256"]:
        raise ValueError("V3 source/data/IDF changed.")
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != read_json(root/DIRECTORY/"manifest.json")["audit_sha256"]:
        raise ValueError("V3 audited data changed.")
    for name, receipt in metadata["best"].items():
        if name not in CONFIG["weight_exponents"] or sha256_file(directory/(name+"_best.joblib")) != receipt["sha256"]:
            raise ValueError("V3 model checkpoint changed.")
    return directory, metadata


def evaluate(root):
    directory, metadata = checked(root)
    check_path = directory/"fresh_check.jsonl"
    identity = {"metadata": sha256_file(directory/"metadata.json"), "check": sha256_file(check_path)}
    if identity["check"] != metadata["protocol"]["fresh_check"]["sha256"]:
        raise ValueError("V3 reserved check changed.")
    marker_path = directory/"final_experiment.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["status"] != "completed" or marker["binding"] != identity:
            raise ValueError("V3 check consumed/interrupted/changed; never reset it to retune.")
        path = (root/marker["report"]).resolve()
        if not path.is_relative_to((root/"reports/feedback_v3_runs").resolve()) or sha256_file(path) != marker["report_sha256"]:
            raise ValueError("Completed V3 report changed.")
        return read_json(path)
    write_json(marker_path, {"status": "started", "binding": identity, "test_exposed": True})
    rows = read_records(check_path)
    texts = [r["text"] for r in rows]
    results = {CONTROL: metrics(rows, predict_control(root, texts))}
    idf = joblib.load(directory/"idf.joblib")
    for name in metadata["best"]:
        results[name] = metrics(rows, predict_model(joblib.load(directory/(name+"_best.joblib")), idf, texts))
    fit_counts = read_json(root/DIRECTORY/"manifest.json")["available_by_partition_and_rating"]["fit"]
    majority = max(fit_counts, key=fit_counts.get)
    report = {"run_id": directory.name, "training": metadata, "classification": results,
        "natural_training_majority": metrics(rows, [int(majority)]*len(rows)),
        "selected_classifier": metadata["selected_classifier"], "selection_unchanged_after_test": True,
        "test_exposed": True, "automatic_actions": False, "business_value_established": False,
        "limitations": ["Same five subjective stars; 90% not guaranteed", "Already exposed development partition",
            "Fresh check excludes all previously tested products, shifting population", "Bundled changes do not isolate causal contributions",
            "Lexical features retain limited word order; hash collisions remain", "No user/temporal/semantic-near-duplicate independence claim"]}
    target = root/"reports/feedback_v3_runs"/directory.name/"summary.json"
    write_json(target, report)
    write_json(marker_path, {"status": "completed", "binding": identity, "test_exposed": True,
        "report": target.relative_to(root).as_posix(), "report_sha256": sha256_file(target)})
    return report


class FeedbackV3:
    def __init__(self, root):
        self.root = root
        self.directory, self.metadata = checked(root)
        self.idf = joblib.load(self.directory/"idf.joblib")
        name = self.metadata["selected_classifier"]
        self.model = None if name == CONTROL else joblib.load(self.directory/(name+"_best.joblib"))

    def predict(self, texts):
        return predict_control(self.root, texts) if self.model is None else predict_model(self.model, self.idf, texts)
