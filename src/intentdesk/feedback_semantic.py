"""V4: same-budget semantic/ordinal experiments with a fresh selection set."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import re

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY
from intentdesk import feedback_v2, feedback_v3
from intentdesk import feedback_semantic_data as data
from intentdesk import feedback_semantic_encoder as encoder
from intentdesk import feedback_semantic_models as models


CONFIG = {"version": "feedback_semantic_s1", "minimum_macro_gain": .005,
          "maximum_accuracy_loss": .01, "bootstrap_product_repetitions": 200,
          "selection": "Maximum new development macro-F1, then accuracy, then NLL; retain V2 unless gain >= .005 and accuracy loss <= .01",
          "no_target_guarantee": "Five exact star labels, no 90% promise; no final-result reselection"}
CONTROL = "retained_v2_full_fit"
HEADS = {f"{kind}_{view}": {"kind":kind, "view":view}
         for kind in ("nominal", "ordered") for view in encoder.CONFIG["views"]}


def binding(root):
    names = ("feedback_semantic.py", "feedback_semantic_data.py", "feedback_semantic_encoder.py",
             "feedback_semantic_models.py", "feedback_v3.py", "feedback_v2.py", "feedback_data.py",
             "feedback_lab.py", "feedback_models.py", "common.py")
    return {"sources":{n:sha256_file(root/"src/intentdesk"/n) for n in names},
            "runner":sha256_file(root/"scripts/run_feedback_semantic_job.py"),
            "manifest":sha256_file(root/DIRECTORY/"manifest.json"), "uv_lock":sha256_file(root/"uv.lock"),
            "configs":{"experiment":CONFIG,"data":data.CONFIG,"encoder":encoder.CONFIG,"models":models.CONFIG},
            "model_id":encoder.MODEL_ID,"revision":encoder.REVISION,
            "previous_metadata":sha256_file(feedback_v3.current_directory(root)/"metadata.json"),
            "previous_marker":sha256_file(feedback_v3.current_directory(root)/"final_experiment.json"),
            "v2_models":sha256_file(feedback_v2.current_directory(root)/"models.joblib")}


def current_directory(root):
    name = read_json(root/"artifacts/feedback_semantic_latest.json")["run_id"]
    if not re.fullmatch(r"feedback_semantic_[0-9TZ]+",name):
        raise ValueError("Invalid semantic run path.")
    return root/"artifacts"/name


def prepare(root):
    identity = binding(root)
    pointer = root/"artifacts/feedback_semantic_latest.json"
    if pointer.exists():
        directory = current_directory(root)
        if read_json(directory/"protocol.json")["binding"] != identity:
            raise ValueError("Semantic source/data/config changed; preserve run and use a new version.")
        return directory
    feedback_v3.checked(root)
    feedback_v2.checked(root)
    directory = root/"artifacts"/("feedback_semantic_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True,exist_ok=False)
    receipt = data.reserve(root,directory)
    write_json(directory/"protocol.json",{"binding":identity,"data":receipt,
        "hypotheses":["Pretrained semantic features may outperform lexical features on identical training rows",
            "Ordered thresholds may reduce neighboring-rating mistakes", "Beginning/end windows may help updated or long reviews"],
        "fixed_candidates":[CONTROL,"lexical_logistic","semantic_logistic_prefix","semantic_logistic_ends"]+list(HEADS),
        "conditional_candidate":"One last-block fine-tune, only if a frozen neural head beats lexical macro-F1 by .005 and resource profile passes",
        "controls":"V2 uses 2,912,393 fit reviews; pilot candidates share 20,000. The full-fit control is not a same-budget algorithm ablation.",
        "human_validation":"200 fit-only blinded reviews prepared; zero human rows completed; existing workflow pilot still needs people"})
    write_json(pointer,{"run_id":directory.name})
    return directory


def control_probabilities(root, rows):
    value = joblib.load(feedback_v2.current_directory(root)/"models.joblib")["word_char_balanced"]
    output = []
    with threadpool_limits(limits=2):
        for start in range(0,len(rows),256):
            x = feedback_v2.features([r["text"] for r in rows[start:start+256]])
            output.append(value.predict_proba(x))
    if list(value.classes_) != [1,2,3,4,5]:
        raise ValueError("Control class order changed.")
    return models.validate_probabilities(np.concatenate(output))


def feature_arrays(directory, name):
    receipt = read_json(directory/(name+"_features.json"))
    result = {}
    rows = read_json(directory/"protocol.json")["data"]["partitions"][name]["rows"]
    if receipt["offset"] != rows:
        raise ValueError("Features are incomplete.")
    if set(receipt["sha256"]) != set(encoder.CONFIG["views"]):
        raise ValueError("Feature views are incomplete.")
    for view, digest in receipt["sha256"].items():
        path = directory/(name+"_"+view+".npy")
        if sha256_file(path) != digest:
            raise ValueError("Semantic feature bytes changed.")
        value = np.load(path,mmap_mode="r",allow_pickle=False)
        if value.shape != (rows,encoder.CONFIG["dimension"]) or value.dtype!=np.float32:
            raise ValueError("Semantic feature shape/dtype changed.")
        result[view] = value
    return result


def build_features(root,directory,partitions,receipt):
    tokenizer, network, manifest = encoder.load_encoder(root)
    for name in partitions:
        rows = data.load(directory,name,receipt)
        encoder.embed_rows(directory,name,rows,tokenizer,network,manifest)
    del network,tokenizer
    gc.collect()
    return manifest


def selection(results):
    best = sorted(results,key=lambda n:(-results[n]["macro_f1"],-results[n]["accuracy"],results[n]["nll"],n))[0]
    if (results[best]["macro_f1"]-results[CONTROL]["macro_f1"]<CONFIG["minimum_macro_gain"] or
        results[best]["accuracy"]<results[CONTROL]["accuracy"]-CONFIG["maximum_accuracy_loss"]):
        return CONTROL
    return best


def predict(root,directory,name,rows,features):
    if name==CONTROL:
        return control_probabilities(root,rows)
    if name=="partial_finetune":
        state = models.load_torch(directory/"fine_best.pt")
        tokenizer, network, _ = encoder.load_encoder(root)
        network.encoder.layer[-1].load_state_dict(state["block"])
        head = models.make_head(state["kind"])
        head.load_state_dict(state["head"])
        p = models.fine_predict(tokenizer,network,head,rows,state["view"])
        del network,tokenizer,head
        gc.collect()
        return p
    if name in HEADS:
        state = models.load_torch(directory/(name+".pt"))
        head = models.make_head(state["kind"],encoder.CONFIG["dimension"])
        head.load_state_dict(state["head"])
        return models.head_probabilities(head,features[HEADS[name]["view"]])
    linear = joblib.load(directory/"linear_models.joblib")
    with threadpool_limits(limits=2):
        x = linear["tfidf"].transform([r["text"] for r in rows]) if name=="lexical_logistic" else features[name.removeprefix("semantic_logistic_")]
        return models.validate_probabilities(linear["models"][name].predict_proba(x))


def train(root):
    directory = prepare(root)
    if (directory/"metadata.json").exists():
        return checked(root)[1]
    protocol = read_json(directory/"protocol.json")
    receipt = protocol["data"]
    manifest = build_features(root,directory,("fit","development","calibration"),receipt)
    fit_rows = data.load(directory,"fit",receipt)
    dev_rows = data.load(directory,"development",receipt)
    fit = feature_arrays(directory,"fit"); dev = feature_arrays(directory,"development")
    linear = models.train_logistic(directory,fit_rows,dev_rows,fit,dev)
    y = np.array([r["rating"] for r in fit_rows])
    dev_y = np.array([r["rating"] for r in dev_rows])
    result = {CONTROL:models.metrics(dev_y,control_probabilities(root,dev_rows)),**linear["development"]}
    head_results = {}
    for name, specification in HEADS.items():
        head_results[name] = models.train_head(directory,name,specification["kind"],fit[specification["view"]],y,dev[specification["view"]],dev_y)
        result[name] = head_results[name]["development"]
    best_head = sorted(HEADS,key=lambda n:(-result[n]["macro_f1"],-result[n]["accuracy"],result[n]["nll"],n))[0]
    specification = HEADS[best_head]
    if result[best_head]["macro_f1"]-result["lexical_logistic"]["macro_f1"] >= models.CONFIG["fine_minimum_gain_over_lexical"]:
        fine = models.finetune(root,directory,best_head,specification["kind"],specification["view"],fit_rows,dev_rows)
        if fine.get("development") is not None:
            result["partial_finetune"] = fine["development"]
    else:
        fine = {"status":"skipped_development_guard","reason":"No frozen neural head improved lexical macro-F1 by .005"}
    selected = selection(result)
    # Freeze selection before any calibration or final exposure.
    write_json(directory/"selection.json",{"selected_classifier":selected,"development":result,
        "selection":CONFIG["selection"],"test_exposed":False})
    calibration_rows = data.load(directory,"calibration",receipt)
    calibration_p = predict(root,directory,selected,calibration_rows,feature_arrays(directory,"calibration"))
    calibration = models.calibrate(calibration_rows,calibration_p)
    write_json(directory/"calibration.json",calibration)
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != read_json(root/DIRECTORY/"manifest.json")["audit_sha256"]:
        raise ValueError("Audited database changed during semantic training.")
    # Bind only deployment/evaluation inputs, not mutable human worksheets.
    bound = ["linear_models.joblib","selection.json","calibration.json"]+[n+".pt" for n in HEADS]
    bound += [name+"_features.json" for name in ("fit","development","calibration")]
    if "partial_finetune" in result:
        bound.append("fine_best.pt")
    metadata = {"run_id":directory.name,"binding":protocol["binding"],"protocol":protocol,
        "encoder_manifest":manifest,"development":result,"heads":head_results,"fine_tuning":fine,
        "learning_curve":linear["learning_curve"],
        "selected_classifier":selected,"calibration":calibration,"test_exposed":False,
        "automatic_actions":False,"business_value_established":False,
        "human_audit_completed":0,"artifacts":{n:sha256_file(directory/n) for n in bound}}
    write_json(directory/"metadata.json",metadata)
    write_json(root/"artifacts/feedback_semantic_latest.json",{"run_id":directory.name,
        "metadata_sha256":sha256_file(directory/"metadata.json")})
    return metadata


def checked(root):
    directory = current_directory(root)
    pointer = read_json(root/"artifacts/feedback_semantic_latest.json")
    if sha256_file(directory/"metadata.json") != pointer.get("metadata_sha256"):
        raise ValueError("Semantic training incomplete or metadata changed.")
    metadata = read_json(directory/"metadata.json")
    if metadata["binding"] != binding(root):
        raise ValueError("Semantic source/data/config changed.")
    if read_json(root/"models/feedback_minilm/manifest.json") != metadata["encoder_manifest"]:
        raise ValueError("Semantic encoder provenance changed.")
    for name,digest in metadata["encoder_manifest"]["sha256"].items():
        if name!=Path(name).name or sha256_file(root/"models/feedback_minilm"/name)!=digest:
            raise ValueError("Semantic encoder weights/tokenizer changed.")
    for name,digest in metadata["artifacts"].items():
        if name!=Path(name).name or sha256_file(directory/name)!=digest:
            raise ValueError("Semantic model/calibration artifact changed.")
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != read_json(root/DIRECTORY/"manifest.json")["audit_sha256"]:
        raise ValueError("Semantic audited database changed.")
    return directory,metadata


def product_intervals(rows, probabilities, control):
    groups = sorted({r["product"] for r in rows})
    ids = {p:i for i,p in enumerate(groups)}
    actual = np.array([r["rating"]-1 for r in rows])
    matrices = []
    for p in (probabilities,control):
        a = np.zeros((len(groups),5,5),dtype=np.int64)
        np.add.at(a,([ids[r["product"]] for r in rows],actual,p.argmax(1)),1)
        matrices.append(a)
    rng = np.random.default_rng(models.CONFIG["seed"])
    estimates = []
    for _ in range(CONFIG["bootstrap_product_repetitions"]):
        sampled = rng.integers(0,len(groups),len(groups))
        scores = []
        for a in matrices:
            matrix = a[sampled].sum(0)
            tp = matrix.diagonal()
            f1 = np.divide(2*tp,matrix.sum(0)+matrix.sum(1),out=np.zeros(5,dtype=float),where=(matrix.sum(0)+matrix.sum(1))>0).mean()
            scores.append((float(tp.sum()/matrix.sum()),float(f1)))
        estimates.append([*scores[0],scores[0][0]-scores[1][0],scores[0][1]-scores[1][1]])
    lower,upper = np.quantile(estimates,[.025,.975],axis=0)
    return {"method":f"{CONFIG['bootstrap_product_repetitions']} paired product bootstrap resamples; descriptive, never model selection",
            "products":len(groups),"accuracy_95_interval":[float(lower[0]),float(upper[0])],
            "macro_f1_95_interval":[float(lower[1]),float(upper[1])],
            "accuracy_difference_vs_v2_interval":[float(lower[2]),float(upper[2])],
            "macro_f1_difference_vs_v2_interval":[float(lower[3]),float(upper[3])]}


def evaluate(root):
    directory,metadata = checked(root)
    path = directory/"test.jsonl"
    identity = {"metadata":sha256_file(directory/"metadata.json"),"test":sha256_file(path)}
    if identity["test"] != metadata["protocol"]["data"]["partitions"]["test"]["sha256"]:
        raise ValueError("Semantic final sample changed.")
    marker_path = directory/"final_experiment.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["status"]!="completed" or marker["binding"]!=identity:
            raise ValueError("Semantic final consumed/interrupted/changed; never reset to retune.")
        report_path = (root/marker["report"]).resolve()
        if not report_path.is_relative_to((root/"reports/feedback_semantic_runs").resolve()) or sha256_file(report_path)!=marker["report_sha256"]:
            raise ValueError("Completed semantic report changed.")
        return read_json(report_path)
    write_json(marker_path,{"status":"started","binding":identity,"test_exposed":True})
    build_features(root,directory,("test",),metadata["protocol"]["data"])
    rows = data.load(directory,"test",metadata["protocol"]["data"])
    features = feature_arrays(directory,"test")
    y = np.array([r["rating"] for r in rows])
    results = {}; selected_p = None; control_p = None
    for name in metadata["development"]:
        p = predict(root,directory,name,rows,features)
        results[name] = models.metrics(y,p)
        if name==metadata["selected_classifier"]: selected_p=p
        if name==CONTROL: control_p=p
    calibrated = models.temperature_probabilities(selected_p,metadata["calibration"]["temperature"])
    threshold = metadata["calibration"]["threshold"]
    confident = models.confidence_metrics(y,calibrated,threshold) if threshold is not None else {"coverage":0.,"confident_accuracy":None,"reason":"Calibration threshold guard failed","automatic_actions":False}
    counts = read_json(root/DIRECTORY/"manifest.json")["available_by_partition_and_rating"]["fit"]
    majority = int(max(counts,key=counts.get))
    majority_p = np.zeros((len(rows),5)); majority_p[:,majority-1]=1
    report = {"run_id":directory.name,"training":metadata,"classification":results,
        "selected_classifier":metadata["selected_classifier"],"selection_unchanged_after_test":True,
        "majority_baseline":models.metrics(y,majority_p),"confidence_advisory":confident,
        "calibrated_nll":models.metrics(y,calibrated)["nll"],
        "selected_vs_control_product_bootstrap":product_intervals(rows,selected_p,control_p),
        "test_exposed":True,"automatic_actions":False,"business_value_established":False,"human_validation_completed":False,
        "limitations":["New development/calibration/test groups exclude previous exposure and shift population",
            "Pilot versus full-fit V2 uses different training budgets; same-budget linear controls are separate",
            "Ends view averages first/last 126 wordpieces, omitting the middle of long reviews",
            "Frozen encoder versus conditional last-block adaptation; no foundation-model training claim",
            "Pretraining exposure cannot be fully ruled out; exact fingerprints do not ensure semantic independence",
            "Confidence-subset accuracy is not overall accuracy; all outputs still require review",
            "Subjective five-star labels; no 90% guarantee or human/customer benefit established"]}
    report_path = root/"reports/feedback_semantic_runs"/directory.name/"summary.json"
    write_json(report_path,report)
    write_json(marker_path,{"status":"completed","binding":identity,"test_exposed":True,
        "report":report_path.relative_to(root).as_posix(),"report_sha256":sha256_file(report_path)})
    return report


class SemanticModel:
    """Checked reusable inference for a future, development-qualified release."""
    def __init__(self,root):
        self.root = root
        self.directory,self.metadata = checked(root)
        self.name = self.metadata["selected_classifier"]
        self.tokenizer = self.network = self.head = self.linear = None
        if self.name.startswith("semantic_logistic_") or self.name in HEADS or self.name=="partial_finetune":
            self.tokenizer,self.network,_ = encoder.load_encoder(root)
        if self.name in HEADS or self.name=="partial_finetune":
            state = models.load_torch(self.directory/("fine_best.pt" if self.name=="partial_finetune" else self.name+".pt"))
            self.head = models.make_head(state["kind"])
            self.head.load_state_dict(state["head"])
            self.head.eval()
            if self.name=="partial_finetune":
                self.network.encoder.layer[-1].load_state_dict(state["block"])
                self.view = state["view"]
        elif self.name!=CONTROL:
            self.linear = joblib.load(self.directory/"linear_models.joblib")

    def probabilities(self,texts):
        from intentdesk.feedback_data import clean_rating_declarations
        from intentdesk.predict import validate_text
        rows = [{"text":clean_rating_declarations(validate_text(t))} for t in texts]
        if self.name==CONTROL:
            return control_probabilities(self.root,rows)
        if self.name=="partial_finetune":
            return models.fine_predict(self.tokenizer,self.network,self.head,rows,self.view)
        if self.name=="lexical_logistic":
            return self.linear["models"][self.name].predict_proba(self.linear["tfidf"].transform([r["text"] for r in rows]))
        outputs = {v:[] for v in encoder.CONFIG["views"]}
        for start in range(0,len(rows),encoder.CONFIG["batch_size"]):
            batch,_ = encoder.embed_batch(self.tokenizer,self.network,[r["text"] for r in rows[start:start+encoder.CONFIG["batch_size"]]])
            for view in outputs:
                outputs[view].append(batch[view])
        features = {v:np.concatenate(a) for v,a in outputs.items()}
        if self.name in HEADS:
            return models.head_probabilities(self.head,features[HEADS[self.name]["view"]])
        return self.linear["models"][self.name].predict_proba(features[self.name.removeprefix("semantic_logistic_")])

    def predict(self,texts):
        return (self.probabilities(texts).argmax(1)+1).tolist()
