"""Plot actual aggregate cluster sizes; never invent semantic topic labels.

Run from the repository root after publishing the feedback summary snapshot:
uv run python learning/plot_feedback_clusters.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    directory=Path.cwd()/"reports/feedback_lab_v1"
    report=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    groups=report["training"]["unsupervised"]["clusters"]
    labels=[f"Group {r['cluster']}: "+", ".join(r["keywords"][:3]) for r in groups]
    figure,axis=plt.subplots(figsize=(12,8))
    axis.barh(labels,[r["fit_members"] for r in groups],color="#146146")
    axis.set(xlabel="Training examples assigned",title="Unsupervised groups: actual sizes and top lexical features")
    figure.suptitle("Rating-balanced training sample / no targets passed to K-means / not verified semantic topics",fontsize=10)
    figure.tight_layout();figure.savefig(directory/"clustering.png",dpi=160);plt.close(figure)
    print(directory/"clustering.png")


if __name__=="__main__":
    main()
