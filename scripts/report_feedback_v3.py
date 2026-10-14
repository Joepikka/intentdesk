"""Format the completed, hash-checked V3 result; never launch new training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from intentdesk.common import read_json, sha256_file
from intentdesk.feedback_v3 import current_directory


def main():
    root=Path(__file__).resolve().parents[1]
    directory=current_directory(root)
    marker=read_json(directory/"final_experiment.json")
    if marker.get("status")!="completed":
        raise ValueError("Complete the declared experiment first.")
    report_path=(root/marker["report"]).resolve()
    if not report_path.is_relative_to((root/"reports/feedback_v3_runs").resolve()) or sha256_file(report_path)!=marker["report_sha256"]:
        raise ValueError("Completed report binding changed.")
    report=read_json(report_path)
    names=list(report["classification"])
    labels=["Retained V2","IDF balanced","IDF moderate"]
    development=report["training"]["development"]
    final=report["classification"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    x=np.arange(3)
    axes[0].bar(x-.18,[development[n]["macro_f1"] for n in names],width=.36,label="Development",color="#879781")
    axes[0].bar(x+.18,[final[n]["macro_f1"] for n in names],width=.36,label="Fresh check",color="#294d3c")
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel="Macro-F1",title="Selection stayed fixed before fresh check")
    axes[0].legend()
    axes[1].bar(labels,[final[n]["accuracy"] for n in names],color="#294d3c")
    axes[1].axhline(report["natural_training_majority"]["accuracy"],linestyle="--",color="#923e32",label="Training-majority predictor")
    axes[1].set(ylim=(0,1),ylabel="Accuracy",title="Accuracy alone can hide class tradeoffs")
    axes[1].legend(fontsize=8)
    matrix=np.array(final[report["selected_classifier"]]["confusion_matrix"])
    recall=matrix/np.maximum(1,matrix.sum(axis=1,keepdims=True))
    image=axes[2].imshow(recall,cmap="Greens",vmin=0,vmax=1)
    axes[2].set(xticks=range(5),yticks=range(5),xticklabels=range(1,6),yticklabels=range(1,6),xlabel="Predicted stars",ylabel="True stars",title="Retained model: row-normalized confusion")
    for i in range(5):
        for j in range(5):
            axes[2].text(j,i,f"{recall[i,j]:.2f}",ha="center",va="center",color="white" if recall[i,j]>.55 else "black",fontsize=8)
    fig.colorbar(image,ax=axes[2],fraction=.046)
    fig.suptitle("IntentDesk V3 · one fresh 20,000-review product-group check",fontsize=14)
    target=root/"reports/feedback_v3_comparison.png"
    fig.savefig(target,dpi=160)
    print(target)


if __name__=="__main__":main()
