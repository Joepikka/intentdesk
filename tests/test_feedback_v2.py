import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from intentdesk import feedback_v2 as v2
from intentdesk.common import write_json,sha256_file
from intentdesk.feedback_data import DIRECTORY,ingest,product_split,text_key
from intentdesk.feedback_models import streaming_vectorizer


def test_word_character_features_are_sparse_normalized_and_bounded(monkeypatch):
    monkeypatch.setitem(v2.CONFIG,"hash_features_per_view",128)
    x=v2.features(["a useful working product","a usefull working product"])
    assert x.shape==(2,256) and x.dtype==np.float32
    assert np.allclose(np.asarray(x.multiply(x).sum(1)).ravel(),1,atol=1e-6)
    assert x.nnz<512


def test_prior_adjustment_is_normalized_and_zero_preserves_probabilities():
    p=np.array([[.4,.1,.1,.1,.3]])
    priors=np.array([.1,.1,.1,.1,.6])
    assert np.allclose(v2.adjust_prior(p,priors,0),p)
    changed=v2.adjust_prior(p,priors,1)
    assert changed.argmax(1)[0]==4 and np.allclose(changed.sum(1),1)


def test_selection_keeps_original_when_accuracy_rises_but_macro_f1_drops():
    metrics={"original_streaming":{"macro_f1":.55,"accuracy":.70},
             "original_prior_full":{"macro_f1":.48,"accuracy":.82},
             "word_char_balanced":{"macro_f1":.553,"accuracy":.72}}
    assert v2.selection(metrics)=="original_streaming"
    metrics["word_char_balanced"]["macro_f1"]=.56
    assert v2.selection(metrics)=="word_char_balanced"


@pytest.fixture
def toy(tmp_path,monkeypatch):
    monkeypatch.setitem(v2.CONFIG,"hash_features_per_view",128)
    monkeypatch.setitem(v2.CONFIG,"fresh_check_rows",10)
    products={}
    for i in range(1000):
        p=f"v2-fixture-{i}";products.setdefault(product_split(p),[]).append(p)
    adjective={1:"awful broken terrible",2:"poor disappointing weak",3:"ordinary average adequate",
               4:"good useful reliable",5:"wonderful excellent superb"}
    fit=[{"text":f"{adjective[y]} product fit example {i}","rating":y,
          "product":products["fit"][0]} for y in range(1,6) for i in range(8)]
    dev=[{"text":f"{adjective[y]} product development example {i}","rating":y,
          "product":products["development"][0]} for y in range(1,6) for i in range(3)]
    previous=[{"text":"previous consumed product review","rating":5,"product":products["test"][0]}]
    fresh=[{"text":f"{adjective[y]} product untouched check example {i}","rating":y,
            "product":products["test"][1]} for y in range(1,6) for i in range(4)]
    records=fit+dev+previous+fresh
    for row in records:
        row["id"]=text_key(row["text"])
    data=tmp_path/DIRECTORY;data.mkdir(parents=True)
    source=data/"source.jsonl"
    source.write_text("".join(json.dumps({"title":"","text":r["text"],"rating":r["rating"],"parent_asin":r["product"]})+"\n" for r in records))
    ingest([source],data/"audit.sqlite3")
    write_json(data/"manifest.json",{"fixture":True})
    (tmp_path/"uv.lock").write_text("fixture")
    original=tmp_path/"artifacts/original";original.mkdir(parents=True)
    write_json(original/"final_experiment.json",{"status":"completed"})
    metadata={"run_id":"original","data":{"audit_sha256":sha256_file(data/"audit.sqlite3"),
        "available_by_partition_and_rating":{"fit":{str(y):8 for y in range(1,6)}}}}
    write_json(original/"metadata.json",metadata)
    model=SGDClassifier(loss="log_loss",random_state=2).fit(
        streaming_vectorizer().transform([r["text"] for r in fit]),[r["rating"] for r in fit])
    monkeypatch.setattr(v2,"directory_for",lambda root:original)
    monkeypatch.setattr(v2,"checked_metadata",lambda root:metadata)
    monkeypatch.setattr(v2,"baseline_model",lambda root:model)
    monkeypatch.setattr(v2,"load_partition",lambda root,name:{"fit":fit,"development":dev,"test":previous}[name])
    monkeypatch.setattr(v2,"eligible_rows",lambda *args,**kwargs:iter([fit]))
    return tmp_path,previous,fresh,original


def test_fresh_check_excludes_entire_previously_tested_product(toy):
    root,previous,fresh,_=toy
    target=root/"reserved";target.mkdir()
    result=v2.reserve_check(root,target,previous,10)
    rows=[json.loads(line) for line in (target/"fresh_check.jsonl").read_text().splitlines()]
    assert len(rows)==10 and result["previous_product_overlap"]==0
    assert {r["product"] for r in rows}=={fresh[0]["product"]}


def test_full_toy_v2_trains_checks_once_and_preserves_old_release(toy):
    root,_,_,old=toy
    before={p.name:sha256_file(p) for p in old.iterdir()}
    metadata=v2.train(root)
    assert metadata["test_exposed"] is False and metadata["fit_rows"]==40
    assert set(metadata["development"])==set(v2.CANDIDATES)
    report=v2.evaluate(root)
    assert report["selected_classifier"]==metadata["selected_classifier"]
    assert report["selection_unchanged_after_test"] and not report["previous_final_test_reused"]
    assert v2.evaluate(root)==report
    assert {p.name:sha256_file(p) for p in old.iterdir()}==before
    assert v2.FeedbackV2(root).predict(["a good reliable useful product"])[0] in range(1,6)


def test_changed_fresh_check_refuses_reexposure(toy):
    root,*_=toy
    v2.train(root);v2.evaluate(root)
    path=v2.current_directory(root)/"fresh_check.jsonl"
    path.write_text("changed")
    with pytest.raises(ValueError,match="fresh check changed"):
        v2.evaluate(root)
