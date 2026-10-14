"""Bounded word/character experiment; preserves the consumed feedback release."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import math
import json
import re
import sqlite3
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY, eligible_rows, load_partition
from intentdesk.feedback_lab import checked_metadata, directory_for
from intentdesk.feedback_models import classification, streaming_vectorizer


CONFIG={"version":"feedback_v2", "seed":20261014, "hash_features_per_view":2**18,
    "word_ngrams":[1,2], "character_ngrams":[3,5], "alpha":1e-6,
    "epochs":1, "batch_size":1024, "prediction_batch":256, "threads":2,
    "fresh_check_rows":20000, "minimum_macro_f1_gain":.005,
    "selection":"Maximum development macro-F1, accuracy tie-break, alphabetic final tie; retain original unless gain >= .005",
    "scope":"Same five review ratings; no label merging or target accuracy guarantee"}
CANDIDATES=("original_streaming","original_prior_half","original_prior_full",
            "word_char_balanced","word_char_natural")


def features(texts):
    word=HashingVectorizer(n_features=CONFIG["hash_features_per_view"],alternate_sign=False,
        ngram_range=tuple(CONFIG["word_ngrams"]),norm="l2",dtype=np.float32).transform(texts)
    char=HashingVectorizer(n_features=CONFIG["hash_features_per_view"],alternate_sign=False,
        analyzer="char_wb",ngram_range=tuple(CONFIG["character_ngrams"]),norm="l2",dtype=np.float32).transform(texts)
    combined=hstack([word,char],format="csr",dtype=np.float32)
    combined.data*=np.float32(1/math.sqrt(2))
    return combined


def adjust_prior(probabilities, priors, exponent):
    p=np.asarray(probabilities,dtype=np.float64)*np.power(np.asarray(priors),exponent)
    return p/np.maximum(p.sum(1,keepdims=True),1e-30)


def selection(metrics):
    best=sorted(metrics,key=lambda n:(-metrics[n]["macro_f1"],-metrics[n]["accuracy"],n))[0]
    if metrics[best]["macro_f1"]-metrics["original_streaming"]["macro_f1"] < CONFIG["minimum_macro_f1_gain"]:
        return "original_streaming"
    return best


def current_directory(root):
    pointer=read_json(root/"artifacts/feedback_v2_latest.json")
    name=pointer["run_id"]
    if not re.fullmatch(r"feedback_v2_[0-9TZ]+",name):
        raise ValueError("Invalid feedback V2 directory.")
    return root/"artifacts"/name


def source_binding(root):
    return {"source_sha256":sha256_file(Path(__file__)),
        "baseline_metadata_sha256":sha256_file(directory_for(root)/"metadata.json"),
        "baseline_final_marker_sha256":sha256_file(directory_for(root)/"final_experiment.json"),
        "manifest_sha256":sha256_file(root/DIRECTORY/"manifest.json"),
        "uv_lock_sha256":sha256_file(root/"uv.lock"),"config":CONFIG}


def reserve_check(root, target, previous_test, limit):
    """Exclude entire products seen in the old check, not merely its review IDs."""
    blocked={row["product"] for row in previous_test}
    with sqlite3.connect(root/DIRECTORY/"audit.sqlite3") as db:
        db.execute("CREATE TEMP TABLE blocked(product TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO blocked VALUES(?)",((p,) for p in blocked))
        query="""SELECT hash,text,rating,product FROM reviews r WHERE split='test'
            AND ratings IN(1,2,4,8,16) AND splits=4
            AND NOT EXISTS(SELECT 1 FROM blocked b WHERE b.product=r.product)
            ORDER BY hash LIMIT ?"""
        rows=[{"id":h,"text":t,"rating":y,"product":p} for h,t,y,p in db.execute(query,(limit,))]
    if len(rows)!=limit:
        raise ValueError("Insufficient untouched product groups for the declared new check.")
    if blocked & {r["product"] for r in rows}:
        raise ValueError("The new check reused a previously tested product.")
    path=target/"fresh_check.jsonl"
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    return {"rows":len(rows),"products":len({r["product"] for r in rows}),
        "blocked_previous_products":len(blocked),"previous_product_overlap":0,
        "sha256":sha256_file(path),"rating_counts":dict(Counter(r["rating"] for r in rows)),
        "selection":"First eligible hashes after excluding every previously tested product; no rating balancing",
        "population_limit":"Excluding previously tested products shifts toward less-reviewed products; compare versions on this same check"}


def prepare(root):
    baseline=checked_metadata(root)
    old_marker=read_json(directory_for(root)/"final_experiment.json")
    if old_marker["status"]!="completed":
        raise ValueError("Wait for the original feedback experiment to complete.")
    identity=source_binding(root)
    pointer=root/"artifacts/feedback_v2_latest.json"
    if pointer.exists():
        directory=current_directory(root)
        if read_json(directory/"protocol.json")["binding"]!=identity:
            raise ValueError("V2 source/data/protocol changed; preserve this experiment before a new version.")
        return directory
    if sha256_file(root/DIRECTORY/"audit.sqlite3")!=baseline["data"]["audit_sha256"]:
        raise ValueError("The audited review database changed.")
    name="feedback_v2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory=root/"artifacts"/name;directory.mkdir(parents=True,exist_ok=False)
    # Existing test is already consumed; read only to exclude its entire products.
    check=reserve_check(root,directory,load_partition(root,"test"),CONFIG["fresh_check_rows"])
    write_json(directory/"protocol.json",{"binding":identity,"fresh_check":check,
        "candidates":list(CANDIDATES),"fit_rows":sum(baseline["data"]["available_by_partition_and_rating"]["fit"].values()),
        "baseline_run":baseline["run_id"],"previous_test_consumed":True,
        "hypothesis":"Character context and lower shrinkage may help lexical errors; natural class priors may improve accuracy but harm minority performance",
        "no_more_neural_epochs":"Both neural models' development performance deteriorated after their early selected checkpoints"})
    write_json(pointer,{"run_id":name})
    return directory


def baseline_model(root):
    checked_metadata(root)
    return joblib.load(directory_for(root)/"streaming_sgd.joblib")


def predict_candidates(root,models,texts,priors,original=None):
    if original is None:
        original=baseline_model(root)
    outputs={name:[] for name in CANDIDATES}
    for start in range(0,len(texts),CONFIG["prediction_batch"]):
        batch=texts[start:start+CONFIG["prediction_batch"]]
        with threadpool_limits(limits=CONFIG["threads"]):
            p=original.predict_proba(streaming_vectorizer().transform(batch))
            outputs["original_streaming"].extend((p.argmax(1)+1).tolist())
            for name,power in (("original_prior_half",.5),("original_prior_full",1.)):
                outputs[name].extend((adjust_prior(p,priors,power).argmax(1)+1).tolist())
            x=features(batch)
            for name in ("word_char_balanced","word_char_natural"):
                outputs[name].extend(models[name].predict(x).tolist())
    return outputs


def train(root):
    directory=prepare(root); protocol=read_json(directory/"protocol.json")
    metadata_path=directory/"metadata.json"
    if metadata_path.exists():
        return checked(root)[1]
    baseline=checked_metadata(root)
    counts={int(k):v for k,v in baseline["data"]["available_by_partition_and_rating"]["fit"].items()}
    total=sum(counts.values());priors=[counts[k]/total for k in range(1,6)]
    path=directory/"models.joblib";progress=directory/"progress.json"
    models={n:SGDClassifier(loss="log_loss",alpha=CONFIG["alpha"],average=True,random_state=CONFIG["seed"])
            for n in ("word_char_balanced","word_char_natural")}
    seen,last=0,"";started=time.perf_counter()
    if progress.exists():
        previous=read_json(progress)
        if previous["sha256"]!=sha256_file(path):
            raise ValueError("V2 resume checkpoint changed.")
        models=joblib.load(path);seen,last=previous["fit_rows"],previous["last_key"]
    for index,batch in enumerate(eligible_rows(root,"fit",batch_size=CONFIG["batch_size"],after_key=last)):
        y=np.array([r["rating"] for r in batch]);weights=np.array([total/(5*counts[int(v)]) for v in y])
        with threadpool_limits(limits=CONFIG["threads"]):
            x=features([r["text"] for r in batch])
            models["word_char_balanced"].partial_fit(x,y,classes=np.arange(1,6),sample_weight=weights)
            models["word_char_natural"].partial_fit(x,y,classes=np.arange(1,6))
        seen+=len(batch);last=batch[-1]["id"]
        if index%100==0:
            temporary=path.with_suffix(".tmp");joblib.dump(models,temporary);temporary.replace(path)
            write_json(progress,{"sha256":sha256_file(path),"fit_rows":seen,"last_key":last})
            print(f"Feedback V2 word/character training: {seen:,}/{total:,} reviews (two fixed weightings)",flush=True)
    if seen!=total:
        raise ValueError("V2 streaming count differs from the frozen full-fit allocation.")
    temporary=path.with_suffix(".tmp");joblib.dump(models,temporary);temporary.replace(path)
    write_json(progress,{"sha256":sha256_file(path),"fit_rows":seen,"last_key":last,"completed":True})
    development=load_partition(root,"development")
    predicted=predict_candidates(root,models,[r["text"] for r in development],priors)
    metrics={n:classification([r["rating"] for r in development],p) for n,p in predicted.items()}
    selected=selection(metrics)
    metadata={"run_id":directory.name,"binding":protocol["binding"],"protocol":protocol,
        "development":metrics,"selected_classifier":selected,"fit_rows":seen,"fit_priors":priors,
        "elapsed_this_invocation_seconds":time.perf_counter()-started,"models_sha256":sha256_file(path),
        "test_exposed":False,"automatic_actions":False,"target_accuracy_guaranteed":False}
    write_json(metadata_path,metadata)
    write_json(root/"artifacts/feedback_v2_latest.json",{"run_id":directory.name,"metadata_sha256":sha256_file(metadata_path)})
    return metadata


def checked(root):
    directory=current_directory(root)
    pointer=read_json(root/"artifacts/feedback_v2_latest.json")
    if sha256_file(directory/"metadata.json")!=pointer.get("metadata_sha256"):
        raise ValueError("V2 training is incomplete or metadata changed.")
    metadata=read_json(directory/"metadata.json")
    if metadata["binding"]!=source_binding(root) or metadata["models_sha256"]!=sha256_file(directory/"models.joblib"):
        raise ValueError("V2 bound source/data/model changed.")
    return directory,metadata


def evaluate(root):
    directory,metadata=checked(root)
    marker_path=directory/"final_experiment.json";check_path=directory/"fresh_check.jsonl"
    binding={"metadata_sha256":sha256_file(directory/"metadata.json"),"check_sha256":sha256_file(check_path)}
    if binding["check_sha256"]!=metadata["protocol"]["fresh_check"]["sha256"]:
        raise ValueError("V2 fresh check changed.")
    if marker_path.exists():
        marker=read_json(marker_path)
        if marker["status"]!="completed" or marker["binding"]!=binding:
            raise ValueError("V2 final check consumed/interrupted or changed; never reset it to retune.")
        path=root/marker["report"]
        if sha256_file(path)!=marker["report_sha256"]:
            raise ValueError("V2 completed report changed.")
        return read_json(path)
    write_json(marker_path,{"status":"started","binding":binding,"test_exposed":True})
    rows=read_records(check_path)
    models=joblib.load(directory/"models.joblib")
    predictions=predict_candidates(root,models,[r["text"] for r in rows],metadata["fit_priors"])
    metrics={n:classification([r["rating"] for r in rows],p) for n,p in predictions.items()}
    selected=metadata["selected_classifier"]
    report={"run_id":directory.name,"training":metadata,"classification":metrics,
        "selected_classifier":selected,"selection_unchanged_after_test":True,"test_exposed":True,
        "previous_final_test_reused":False,"automatic_actions":False,"business_value_established":False,
        "limitations":["Same five subjective star labels; no 90% guarantee",
            "New check excludes all previously tested products and shifts toward less-reviewed products",
            "Development partition was already used for earlier selection; new independent final check is consumed",
            "Product disjointness does not ensure temporal/user/semantic duplicate independence"]}
    output=root/"reports/feedback_v2_runs"/directory.name;output.mkdir(parents=True,exist_ok=False)
    write_json(output/"summary.json",report)
    from matplotlib import pyplot as plt
    fig,axis=plt.subplots(figsize=(10,5));names=list(metrics)
    axis.barh(names,[metrics[n]["macro_f1"] for n in names],color=["#146146" if n==selected else "#839c90" for n in names])
    axis.set(xlim=(0,1),xlabel="Fresh product-group check macro-F1",title="Bounded word/character and prior-weighting experiment")
    fig.tight_layout();fig.savefig(output/"comparison.png",dpi=160);plt.close(fig)
    write_json(marker_path,{"status":"completed","binding":binding,"test_exposed":True,
        "report":(output/"summary.json").relative_to(root).as_posix(),"report_sha256":sha256_file(output/"summary.json")})
    return report


class FeedbackV2:
    def __init__(self,root):
        self.root=root;self.directory,self.metadata=checked(root)
        self.models=joblib.load(self.directory/"models.joblib")
        self.original=baseline_model(root)

    def predict(self,texts):
        return predict_candidates(self.root,self.models,texts,self.metadata["fit_priors"],self.original)[self.metadata["selected_classifier"]]
