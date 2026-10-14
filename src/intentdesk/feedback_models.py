"""Classical supervised comparisons, ordinal regression and label-free clustering."""

import time
import warnings

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY, eligible_rows


CONFIG = {"seed": 20261013, "tfidf_max_features": 30000, "tfidf_ngrams": [1,2], "min_df": 3,
          "svd_dimensions": 64, "prediction_batch": 128, "threads": 2,
          "decision_tree_depth": 18, "forest_trees": 96, "forest_depth": 20,
          "neighbors": 7, "clusters": 20, "streaming_epochs": 1,
          "selection": "Maximum development macro-F1; alphabetic tie-break; no final-test tuning"}


def classification(y, predicted):
    return {"samples": len(y), "accuracy": float(accuracy_score(y,predicted)),
            "macro_f1": float(f1_score(y,predicted,labels=[1,2,3,4,5],average="macro",zero_division=0)),
            "per_rating": classification_report(y,predicted,labels=[1,2,3,4,5],output_dict=True,zero_division=0),
            "confusion_matrix": confusion_matrix(y,predicted,labels=[1,2,3,4,5]).tolist()}


def regression(y, predicted):
    return {"samples": len(y), "mae_stars": float(mean_absolute_error(y,predicted)),
            "rmse_stars": float(np.sqrt(mean_squared_error(y,predicted))), "r2": float(r2_score(y,predicted)),
            "prediction_clipping": "none; assumes approximately equal spacing between star values"}


def predict_batches(model, features):
    with threadpool_limits(limits=2):
        return np.concatenate([model.predict(features[start:start+CONFIG["prediction_batch"]]) for start in range(0, features.shape[0], CONFIG["prediction_batch"])])


def estimators():
    seed = CONFIG["seed"]
    return {"logistic_regression": (LogisticRegression(C=2., max_iter=600, tol=.001, random_state=seed), "sparse"),
        "linear_svm": (LinearSVC(C=1., dual="auto", max_iter=3000, tol=.001, random_state=seed), "sparse"),
        "decision_tree": (DecisionTreeClassifier(max_depth=18,min_samples_leaf=12,random_state=seed), "dense"),
        "random_forest": (RandomForestClassifier(n_estimators=96,max_depth=20,min_samples_leaf=8,max_features=.5,n_jobs=2,random_state=seed), "dense"),
        "knn": (KNeighborsClassifier(n_neighbors=7,weights="distance",algorithm="brute",metric="euclidean",n_jobs=1), "dense")}


def train_classical(directory, fit, development, available):
    texts, y = [r["text"] for r in fit], np.array([r["rating"] for r in fit])
    dev_texts, dev_y = [r["text"] for r in development], np.array([r["rating"] for r in development])
    feature_path = directory/"features.joblib"
    if feature_path.exists():
        if sha256_file(feature_path) != read_json(directory/"feature_receipt.json")["sha256"]:
            raise ValueError("Feature checkpoint checksum changed.")
        feature = joblib.load(feature_path)
        sparse = feature["tfidf"].transform(texts)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=3,sublinear_tf=True,dtype=np.float32)
        with threadpool_limits(limits=2):
            sparse = tfidf.fit_transform(texts)
            dimensions = min(CONFIG["svd_dimensions"], sparse.shape[1]-1, sparse.shape[0]-1)
            svd = TruncatedSVD(n_components=dimensions,n_iter=5,random_state=CONFIG["seed"])
            svd.fit(sparse)
        feature = {"tfidf": tfidf, "svd": svd, "normalizer": Normalizer()}
        joblib.dump(feature,feature_path)
        write_json(directory/"feature_receipt.json",{"sha256":sha256_file(feature_path)})
    with threadpool_limits(limits=2):
        dense = feature["normalizer"].transform(feature["svd"].transform(sparse))
        dev_sparse = feature["tfidf"].transform(dev_texts)
        dev_dense = feature["normalizer"].transform(feature["svd"].transform(dev_sparse))
    result = {}
    majority = int(max(available, key=available.get))
    result["natural_training_majority"] = {**classification(dev_y,np.full(len(dev_y),majority)),
        "fit_rows": sum(available.values()), "fit_seconds": 0., "representation": "constant", "constant": majority}
    for name, (model, representation) in estimators().items():
        path = directory/(name+".joblib"); receipt = directory/(name+"_development.json")
        if path.exists() and receipt.exists():
            cached = read_json(receipt)
            if cached["sha256"] != sha256_file(path):
                raise ValueError("Classical model checkpoint checksum changed.")
            result[name] = cached; continue
        print(f"Fitting {name} on {len(fit):,} reviews ({representation})...", flush=True)
        started = time.perf_counter()
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(sparse if representation == "sparse" else dense,y)
        fit_seconds = time.perf_counter()-started
        predicted = predict_batches(model,dev_sparse if representation == "sparse" else dev_dense)
        joblib.dump(model,path)
        result[name] = {**classification(dev_y,predicted), "representation": representation,
            "fit_rows": len(fit), "fit_seconds": fit_seconds, "sha256": sha256_file(path)}
        write_json(receipt,result[name])
        print(f"{name}: development macro-F1 {result[name]['macro_f1']:.4f}", flush=True)
    regression_path = directory/"linear_regression.joblib"
    linear = LinearRegression()
    with threadpool_limits(limits=2):
        linear.fit(dense,y)
    joblib.dump(linear,regression_path)
    regression_results = {"linear_regression": regression(dev_y,predict_batches(linear,dev_dense)),
                          "fit_mean_baseline": regression(dev_y,np.full(len(dev_y),y.mean()))}
    # Rating-balanced sampling is disclosed; no targets enter cluster fitting.
    clusters = MiniBatchKMeans(n_clusters=min(20,len(fit)),batch_size=1024,n_init=3,max_iter=50,random_state=CONFIG["seed"])
    with threadpool_limits(limits=2):
        clusters.fit(dense)
        assigned = clusters.predict(dev_dense)
        silhouette = float(silhouette_score(dev_dense[:1000],assigned[:1000])) if len(set(assigned[:1000]))>1 else None
    joblib.dump(clusters,directory/"clusters.joblib")
    training_clusters = clusters.predict(dense)
    terms = feature["tfidf"].get_feature_names_out(); summaries = []
    for k in range(clusters.n_clusters):
        members = np.flatnonzero(training_clusters==k)
        if len(members):
            weights = np.asarray(sparse[members].mean(0)).ravel()
            keywords = [str(terms[i]) for i in np.argsort(weights)[-10:][::-1]]
        else:
            keywords = []
        summaries.append({"cluster": k,"fit_members": len(members),"keywords": keywords})
    unsupervised = {"method": "MiniBatchKMeans on normalized 64D truncated SVD features",
        "targets_supplied_to_fit": False, "training_sample_is_rating_balanced": True,
        "fit_rows": len(fit), "development_silhouette_first_1000": silhouette,
        "clusters": summaries, "interpretation": "Exploratory word groups; not verified support topics or rating classes"}
    return result, regression_results, unsupervised, {"vocabulary": len(terms), "svd_dimensions": dense.shape[1],
        "svd_explained_variance": float(feature["svd"].explained_variance_ratio_.sum()), "sha256": sha256_file(feature_path)}


def streaming_vectorizer():
    return HashingVectorizer(n_features=2**18,alternate_sign=False,ngram_range=(1,2),norm="l2",dtype=np.float32)


def train_streaming(root, directory, development, available):
    path = directory/"streaming_sgd.joblib"; receipt = directory/"streaming_sgd_progress.json"
    model = SGDClassifier(loss="log_loss",alpha=1e-5,average=True,random_state=CONFIG["seed"])
    seen, last = 0, ""
    if receipt.exists():
        state = read_json(receipt)
        if sha256_file(path) != state["sha256"]:
            raise ValueError("Streaming checkpoint changed.")
        model, seen, last = joblib.load(path), state["fit_rows"], state["last_key"]
    vectorizer = streaming_vectorizer(); total = sum(available.values()); started = time.perf_counter()
    for batch_index, batch in enumerate(eligible_rows(root,"fit",after_key=last)):
        y = np.array([r["rating"] for r in batch])
        weights = np.array([total/(5*available[int(label)]) for label in y])
        with threadpool_limits(limits=2):
            model.partial_fit(vectorizer.transform([r["text"] for r in batch]), y,classes=np.arange(1,6),sample_weight=weights)
        seen += len(batch); last = batch[-1]["id"]
        if batch_index % 50 == 0:
            joblib.dump(model,path)
            write_json(receipt,{"sha256": sha256_file(path),"fit_rows": seen,"last_key": last})
            print(f"Streaming supervised training: {seen:,}/{total:,} fit reviews",flush=True)
    if seen != total:
        raise ValueError("Streaming training count differs from the frozen eligible fit population.")
    joblib.dump(model,path); write_json(receipt,{"sha256": sha256_file(path),"fit_rows": seen,"last_key": last,"completed": True})
    predicted = predict_batches(model,vectorizer.transform([r["text"] for r in development]))
    return {**classification([r["rating"] for r in development],predicted),"representation": "hashing",
        "fit_rows": seen,"fit_seconds_this_invocation": time.perf_counter()-started,"sha256": sha256_file(path),
        "epochs": 1,"class_weights": "inverse natural fit frequency"}
