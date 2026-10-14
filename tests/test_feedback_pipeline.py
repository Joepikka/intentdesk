import json
from pathlib import Path

import pytest
pytest.importorskip("torch")

from intentdesk import feedback_transformer as neural
from intentdesk.common import sha256_file,write_json
from intentdesk.feedback_data import DIRECTORY,ingest,product_split,text_key
from intentdesk.feedback_lab import SOURCES,train,checked_metadata
from intentdesk.feedback_eval import evaluate
from intentdesk.feedback_predict import FeedbackModels


@pytest.fixture
def trained(tmp_path,monkeypatch):
    monkeypatch.setitem(neural.CONFIG,"epochs",1)
    monkeypatch.setitem(neural.CONFIG,"architectures",{
        "scratch_small":{"dimension":8,"heads":2,"layers":1},
        "scratch_large":{"dimension":16,"heads":2,"layers":2}})
    root=tmp_path; actual=Path(__file__).parents[1]/"src/intentdesk"
    for name in SOURCES+("feedback_predict.py",):
        target=root/"src/intentdesk"/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((actual/name).read_bytes())
    (root/"uv.lock").write_text("fixture lock",encoding="utf-8")
    directory=root/DIRECTORY;directory.mkdir(parents=True)
    groups={}
    for i in range(1000):
        name=f"fixture-product-{i}";groups.setdefault(product_split(name),name)
    adjectives={1:"awful broken terrible",2:"poor disappointing weak",3:"average ordinary adequate",4:"good useful reliable",5:"excellent wonderful superb"}
    partitions={}
    source=[]
    for split in ("fit","development","test"):
        rows=[]
        for rating in range(1,6):
            for i in range(12):
                text=f"{adjectives[rating]} product review {split} item variant {i} testing details"
                rows.append({"id":text_key(text),"text":text,"rating":rating,"product":groups[split]})
                source.append({"title":"","text":text,"rating":rating,"parent_asin":groups[split]})
        rows.sort(key=lambda r:r["id"]);partitions[split]=rows
        (directory/(split+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    source_path=directory/"fixture_source.jsonl";source_path.write_text("\n".join(json.dumps(r) for r in source),encoding="utf-8")
    ingest([source_path],directory/"audit.sqlite3")
    write_json(directory/"manifest.json",{"prepared_sha256":{s+".jsonl":sha256_file(directory/(s+".jsonl")) for s in partitions},
        "available_by_partition_and_rating":{s:{str(r):12 for r in range(1,6)} for s in partitions},
        "raw_bytes":source_path.stat().st_size,"audit":{"source_rows":len(source)},
        "audit_sha256":sha256_file(directory/"audit.sqlite3")})
    metadata=train(root)
    return root,metadata


def test_real_toy_pipeline_trains_all_algorithms_without_final_test_exposure(trained):
    root,metadata=trained
    assert set(metadata["development_classification"])=={"natural_training_majority","logistic_regression","linear_svm","decision_tree","random_forest","knn","streaming_sgd","scratch_small","scratch_large"}
    assert metadata["test_exposed"] is False
    assert metadata["unsupervised"]["targets_supplied_to_fit"] is False
    assert metadata["development_classification"]["scratch_large"]["training"]["initialized_from_pretrained"] is False
    assert checked_metadata(root)==metadata
    assert not (root/"artifacts"/metadata["run_id"]/"final_experiment.json").exists()


def test_frozen_final_check_is_reused_and_tampered_input_is_rejected(trained):
    root,metadata=trained
    first=evaluate(root)
    assert first["selected_classifier"]==metadata["selected_classifier"]
    assert first["selection_unchanged_after_test"] is True
    assert all(r["samples"]==60 for r in first["classification"].values())
    assert evaluate(root)==first
    result=FeedbackModels(root).analyze("A fictional product worked very well and arrived promptly.")
    assert result["predicted_stars"] in range(1,6) and result["automatic_actions"] is False
    assert "cluster_keywords" in result
    (root/DIRECTORY/"test.jsonl").write_text("tampered",encoding="utf-8")
    with pytest.raises(ValueError,match="consumed or binding changed"):
        evaluate(root)


def test_extension_preserves_prior_run_and_reuses_fits_but_not_old_final_receipts(trained):
    from intentdesk.feedback_lab import start_extension,directory_for
    from intentdesk.common import read_json
    root,metadata=trained; old=directory_for(root)
    protocol=read_json(old/"protocol.json")
    protocol["transformer_config"]["epochs"]=6
    write_json(old/"protocol.json",protocol)
    original={p.name:sha256_file(p) for p in old.iterdir() if p.is_file()}
    result=start_extension(root); new=directory_for(root)
    assert result["previous_run"]==old.name and new!=old
    assert {p.name:sha256_file(p) for p in old.iterdir() if p.is_file()}==original
    assert sha256_file(new/"random_forest.joblib")==original["random_forest.joblib"]
    assert not (new/"scratch_small_development.json").exists()
    assert (new/"scratch_small_resume.pt").exists()
    resumed=train(root)
    assert resumed["continuation"]["previous_run"]==old.name
    assert start_extension(root)["reused_extension"]


def test_extension_refuses_consumed_final_test(trained):
    from intentdesk.feedback_lab import start_extension,directory_for
    from intentdesk.common import read_json
    root,_=trained; evaluate(root); old=directory_for(root)
    protocol=read_json(old/"protocol.json");protocol["transformer_config"]["epochs"]=6
    write_json(old/"protocol.json",protocol)
    with pytest.raises(ValueError,match="consumed"):
        start_extension(root)


def test_terminal_training_lock_prevents_duplicate_jobs_and_releases(tmp_path):
    from intentdesk.feedback_lab import training_lock
    with training_lock(tmp_path):
        with pytest.raises(ValueError,match="already running"):
            with training_lock(tmp_path):
                pass
    with training_lock(tmp_path):
        pass
