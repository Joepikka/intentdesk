"""One frozen final feedback check after all model and epoch selection is complete."""

from datetime import datetime,timezone
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json,sha256_file,write_json
from intentdesk.feedback_data import DIRECTORY,load_partition
from intentdesk.feedback_lab import directory_for
from intentdesk.feedback_models import classification,regression,predict_batches
from intentdesk.feedback_predict import FeedbackModels


def evaluate(root: Path,model_factory=FeedbackModels):
    models = model_factory(root); metadata = models.metadata; directory = directory_for(root)
    binding = {"metadata_sha256":sha256_file(directory/"metadata.json"),
        "test_sha256":sha256_file(root/DIRECTORY/"test.jsonl"),
        "evaluator_sha256":sha256_file(Path(__file__)),
        "inference_sha256":sha256_file(root/"src/intentdesk/feedback_predict.py"),
        "protocol":"Evaluate every fixed candidate once; selected classifier fixed by development; no test reselection"}
    marker_path = directory/"final_experiment.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["status"]!="completed" or marker["binding"]!=binding:
            raise ValueError("Feedback final check already consumed or binding changed. Preserve it; do not retune against it.")
        report_path = root/marker["report"]
        if sha256_file(report_path)!=marker["report_sha256"]:
            raise ValueError("Completed feedback final report changed.")
        return read_json(report_path)
    marker = {"status":"started","test_exposed":True,"binding":binding}
    write_json(marker_path,marker)
    test = load_partition(root,"test"); texts = [r["text"] for r in test]; y = np.array([r["rating"] for r in test])
    with threadpool_limits(limits=2):
        sparse = models.features["tfidf"].transform(texts)
        dense = models.features["normalizer"].transform(models.features["svd"].transform(sparse))
    results = {}
    for name in metadata["development_classification"]:
        print(f"Frozen test comparison: {name}",flush=True)
        started = time.perf_counter()
        predicted = models.predict_many(name,texts,sparse,dense)
        results[name] = {**classification(y,predicted),"elapsed_seconds":time.perf_counter()-started,
                         "fit_rows":metadata["development_classification"][name]["fit_rows"]}
    reg = regression(y,predict_batches(models.regressor,dense))
    baseline_mean = float(np.mean([r["rating"] for r in load_partition(root,"fit")]))
    reg_baseline = regression(y,np.full(len(y),baseline_mean))
    with threadpool_limits(limits=2):
        assigned = models.clusters.predict(dense)
        silhouette = float(silhouette_score(dense[:1000],assigned[:1000])) if len(set(assigned[:1000]))>1 else None
        inertia = float(np.sum((dense-models.clusters.cluster_centers_[assigned])**2)/len(dense))
    name = metadata["run_id"]+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root/"reports/feedback_lab_runs"/name; output.mkdir(parents=True,exist_ok=False)
    report = {"run_id":name,"binding":binding,"training":metadata,"classification":results,
        "selected_classifier":metadata["selected_classifier"],"selection_unchanged_after_test":True,
        "regression":{"linear_regression":reg,"fit_mean_baseline":reg_baseline},
        "unsupervised":{"targets_supplied_to_fit":False,"test_silhouette_first_1000":silhouette,
            "test_mean_squared_centroid_distance":inertia,"interpretation":"Geometry diagnostics, not semantic topic accuracy"},
        "test_product_groups":len({r["product"] for r in test}),"test_exposed":True,
        "automatic_actions":False,"business_value_established":False,
        "limitations":["Public subjective review ratings, not actual support intents","Product disjoint, not user-disjoint or temporal",
            "Fixed benchmark subsets for costly comparisons; streaming SGD uses all eligible fit rows",
            "Transformer sees the first 63 tokens plus CLS; classical models see the full cleaned review",
            "Pretrained encoder not used in scratch transformers; no foundation-model pretraining claim"]}
    write_json(output/"summary.json",report)
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(15,5))
    names=list(results)
    axes[0].barh(names,[results[n]["macro_f1"] for n in names],color=["#146146" if n==report["selected_classifier"] else "#839c90" for n in names])
    axes[0].set(xlim=(0,1),xlabel="Final macro-F1",title="Fixed product-disjoint review-rating comparison")
    for n in ("scratch_small","scratch_large"):
        history=metadata["development_classification"][n]["training"]["history"]
        axes[1].plot([r["epoch"] for r in history],[r["development_loss"] for r in history],label=n+" development")
        axes[1].plot([r["epoch"] for r in history],[r["fit_loss"] for r in history],linestyle="--",label=n+" fit")
    axes[1].set(xlabel="Epoch",ylabel="Cross-entropy",title="From-scratch transformer learning curves"); axes[1].legend(fontsize=8)
    figure.suptitle("Research benchmark / public subjective ratings / not verified customer benefit")
    figure.tight_layout();figure.savefig(output/"comparison.png",dpi=160);plt.close(figure)
    figure,axis=plt.subplots(figsize=(6,5)); matrix=np.array(results[report["selected_classifier"]]["confusion_matrix"])
    axis.imshow(matrix,cmap="Blues");axis.set(xticks=range(5),xticklabels=range(1,6),yticks=range(5),yticklabels=range(1,6),xlabel="Predicted stars",ylabel="Observed stars",title="Preselected classifier: final confusion matrix")
    for (i,j),value in np.ndenumerate(matrix):
        axis.text(j,i,str(value),ha="center",va="center",color="white" if value>matrix.max()/2 else "black")
    figure.tight_layout();figure.savefig(output/"confusion.png",dpi=160);plt.close(figure)
    write_json(marker_path,{**marker,"status":"completed","report":(output/"summary.json").relative_to(root).as_posix(),"report_sha256":sha256_file(output/"summary.json")})
    return report
