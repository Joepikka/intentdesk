"""Same-budget linear/ordinal comparisons, bounded adaptation and calibration."""
import copy
import gc
import math
import time

import joblib
import numpy as np
import psutil
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.feedback_models import classification
from intentdesk.feedback_semantic_encoder import CONFIG as ENCODER, load_encoder, windows, pooled
from intentdesk.feedback_v3 import clean_markup


CONFIG = {"seed": 20261014, "head_epochs": 30, "head_minimum_epochs": 5,
          "head_patience": 5, "head_batch": 128, "head_lr": .003, "weight_decay": .01,
          "logistic_C": 2., "logistic_max_iter": 600, "tfidf_features": 30000,
          "learning_curve_rows": [5000, 10000, 20000],
          "fine_epochs": 3, "fine_batch": 4, "fine_minimum_gain_over_lexical": .005,
          "fine_budget_seconds": 5400, "fine_encoder_lr": 2e-5, "fine_head_lr": .001,
          "minimum_free_memory_bytes": 268435456, "profile_max_rss_bytes": 1610612736,
          "temperatures": [.5, .75, 1., 1.25, 1.5, 2., 3.],
          "confidence_thresholds": [.5, .6, .7, .8, .85, .9, .95],
          "minimum_confident_accuracy": .85, "minimum_confident_coverage": .2}


class NominalHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 5)

    def forward(self, x):
        return self.linear(x)

    def probabilities(self, x):
        return self(x).softmax(-1)


class OrderedHead(nn.Module):
    """Cumulative-link model inspired by CORAL; explicitly ordered thresholds."""
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Linear(dimension, 1, bias=False)
        self.base = nn.Parameter(torch.tensor(-1.5))
        self.gaps = nn.Parameter(torch.full((3,), math.log(math.expm1(1.))))

    def thresholds(self):
        increments = nn.functional.softplus(self.gaps)+1e-4
        return torch.cat([self.base.reshape(1), self.base+increments.cumsum(0)])

    def forward(self, x):
        return self.score(x)-self.thresholds()

    def probabilities(self, x):
        above = self(x).sigmoid()
        return torch.cat([1-above[:, :1], above[:, :-1]-above[:, 1:], above[:, -1:]], dim=1)


def make_head(kind, dimension=None):
    dimension = ENCODER["dimension"] if dimension is None else dimension
    if kind not in {"nominal", "ordered"}:
        raise ValueError("Unknown rating head.")
    return NominalHead(dimension) if kind == "nominal" else OrderedHead(dimension)


def loss_rows(head, x, y):
    if isinstance(head, OrderedHead):
        targets = (y[:, None] > torch.arange(1, 5)[None]).to(x.dtype)
        return nn.functional.binary_cross_entropy_with_logits(head(x), targets, reduction="none").mean(1)
    return nn.functional.cross_entropy(head(x), y-1, reduction="none")


def rating_weights(y):
    counts = np.bincount(np.asarray(y), minlength=6)[1:]
    if np.any(counts==0):
        raise ValueError("Every training rating must be represented.")
    return torch.tensor(len(y)/(5*counts), dtype=torch.float32)


def validate_probabilities(p):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or np.any(p < -1e-7) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid five-rating probability output.")
    return p


def metrics(y, p):
    p = validate_probabilities(p)
    y = np.asarray(y)
    result = classification(y, p.argmax(1)+1)
    delta = np.abs(y-(p.argmax(1)+1))
    result.update(nll=float(-np.log(np.maximum(p[np.arange(len(y)), y-1], 1e-12)).mean()),
                  mean_absolute_star_error=float(delta.mean()), adjacent_errors=int((delta==1).sum()))
    return result


def score(m):
    return (m["macro_f1"], m["accuracy"], -m["nll"])


def head_probabilities(head, features):
    head.eval()
    with torch.inference_mode():
        return np.concatenate([head.probabilities(torch.from_numpy(np.array(features[i:i+256], copy=True))).numpy()
                               for i in range(0,len(features),256)])


def save_torch(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)
    write_json(path.with_suffix(".json"), {"sha256": sha256_file(path)})


def load_torch(path):
    if sha256_file(path) != read_json(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Semantic checkpoint checksum changed.")
    return torch.load(path, map_location="cpu", weights_only=True)


def train_head(directory, name, kind, fit, y, dev, dev_y):
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(ENCODER["threads"])
    head = make_head(kind, fit.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"])
    targets = torch.tensor(y, dtype=torch.long)
    weights = rating_weights(y)
    resume = directory/(name+"_resume.pt")
    state = {"epoch": 0, "history": [], "best_metrics": None, "stale": 0}
    if resume.exists():
        state = load_torch(resume)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
    while state["epoch"] < CONFIG["head_epochs"]:
        if state["epoch"] >= CONFIG["head_minimum_epochs"] and state["stale"] >= CONFIG["head_patience"]:
            break
        epoch = state["epoch"]+1
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(CONFIG["seed"]+epoch))
        head.train()
        for batch in order.split(CONFIG["head_batch"]):
            x = torch.from_numpy(np.array(fit[batch.numpy()], copy=True))
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_rows(head,x,targets[batch])*weights[targets[batch]-1]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite semantic-head loss.")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite semantic-head gradient.")
            optimizer.step()
        result = metrics(dev_y, head_probabilities(head, dev))
        if state["best_metrics"] is None or score(result) > score(state["best_metrics"]):
            state.update(best_metrics=result, best_epoch=epoch, stale=0,
                         best_head=copy.deepcopy(head.state_dict()))
        else:
            state["stale"] += 1
        state["history"].append({"epoch": epoch, "development": result})
        state.update(epoch=epoch, head=head.state_dict(), optimizer=optimizer.state_dict(), kind=kind)
        save_torch(resume, state)
        print(f"{name} epoch {epoch}: development accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f}", flush=True)
    save_torch(directory/(name+".pt"), {"kind": kind, "head": state["best_head"]})
    head.load_state_dict(state["best_head"])
    return {"kind": kind, "best_epoch": state["best_epoch"], "history": state["history"],
            "development": state["best_metrics"],
            "fit_diagnostic_first_2000":metrics(np.asarray(y)[:2000],head_probabilities(head,fit[:2000])),
            "trainable_parameters": sum(p.numel() for p in head.parameters())}


def train_logistic(directory, fit_rows, dev_rows, fit_features, dev_features):
    path = directory/"linear_models.joblib"
    receipt = directory/"linear_models.json"
    if receipt.exists():
        value = read_json(receipt)
        if sha256_file(path) != value["sha256"]:
            raise ValueError("Linear pilot checkpoint changed.")
        return value
    y = [r["rating"] for r in fit_rows]
    dev_y = [r["rating"] for r in dev_rows]
    tfidf = TfidfVectorizer(preprocessor=clean_markup, ngram_range=(1,2), max_features=CONFIG["tfidf_features"],
                           min_df=2, sublinear_tf=True, dtype=np.float32)
    with threadpool_limits(limits=2):
        x = tfidf.fit_transform([r["text"] for r in fit_rows])
        dx = tfidf.transform([r["text"] for r in dev_rows])
        models, results = {}, {}
        for name, train_x, development_x in [("lexical_logistic", x, dx)]+[
                ("semantic_logistic_"+view, fit_features[view], dev_features[view]) for view in ENCODER["views"]]:
            model = LogisticRegression(C=CONFIG["logistic_C"], class_weight="balanced",
                max_iter=CONFIG["logistic_max_iter"], random_state=CONFIG["seed"])
            model.fit(train_x, y)
            if list(model.classes_) != [1,2,3,4,5] or np.any(model.n_iter_ >= CONFIG["logistic_max_iter"]):
                raise ValueError("Linear pilot class ordering/convergence failed.")
            models[name] = model
            results[name] = metrics(dev_y, model.predict_proba(development_x))
            print(f"{name}: development macro-F1={results[name]['macro_f1']:.4f}", flush=True)
        learning_curve = []
        for size in sorted({min(n,len(y)) for n in CONFIG["learning_curve_rows"]}):
            if size==len(y):
                learning_curve.append({"fit_rows":size,"development":results})
                continue
            small_vectorizer = copy.deepcopy(tfidf)
            sx = small_vectorizer.fit_transform([r["text"] for r in fit_rows[:size]])
            sdx = small_vectorizer.transform([r["text"] for r in dev_rows])
            partial = {}
            for name,tx,vx in [("lexical_logistic",sx,sdx)]+[
                    ("semantic_logistic_"+v,fit_features[v][:size],dev_features[v]) for v in ENCODER["views"]]:
                model = LogisticRegression(C=CONFIG["logistic_C"],class_weight="balanced",
                    max_iter=CONFIG["logistic_max_iter"],random_state=CONFIG["seed"])
                model.fit(tx,y[:size])
                if list(model.classes_) != [1,2,3,4,5] or np.any(model.n_iter_>=CONFIG["logistic_max_iter"]):
                    raise ValueError("Learning-curve class support/convergence failed.")
                partial[name] = metrics(dev_y,model.predict_proba(vx))
            learning_curve.append({"fit_rows":size,"development":partial})
            print(f"Same-budget learning curve: {size:,} fit rows complete",flush=True)
    joblib.dump({"tfidf": tfidf, "models": models}, path)
    value = {"sha256": sha256_file(path), "development": results,"learning_curve":learning_curve}
    write_json(receipt, value)
    return value


def fine_features(tokenizer, encoder, texts, view):
    first, last, long = windows(tokenizer, texts)
    prefix = pooled(encoder, tokenizer.pad({"input_ids": first}, padding=True, return_tensors="pt"))
    if view == "prefix" or not long.any():
        return prefix
    selected = np.flatnonzero(long)
    tail = pooled(encoder, tokenizer.pad({"input_ids": [last[i] for i in selected]}, padding=True, return_tensors="pt"))
    # index_copy is differentiable through both contextual windows.
    return prefix.index_copy(0, torch.tensor(selected), nn.functional.normalize(prefix[selected]+tail,p=2,dim=1))


def fine_predict(tokenizer, encoder, head, rows, view):
    encoder.eval(); head.eval(); result = []
    with torch.inference_mode():
        for start in range(0,len(rows),ENCODER["batch_size"]):
            x = fine_features(tokenizer, encoder, [r["text"] for r in rows[start:start+ENCODER["batch_size"]]], view)
            result.append(head.probabilities(x).numpy())
    return validate_probabilities(np.concatenate(result))


def finetune(root, directory, candidate, kind, view, fit_rows, dev_rows):
    """Adapt only the last encoder block, subject to measured resource limits."""
    finished = directory/"fine_result.json"
    if finished.exists():
        return read_json(finished)
    tokenizer, encoder, _ = load_encoder(root)
    block = encoder.encoder.layer[-1]
    block.requires_grad_(True)
    head = make_head(kind)
    head.load_state_dict(load_torch(directory/(candidate+".pt"))["head"])
    optimizer = torch.optim.AdamW([
        {"params": block.parameters(), "lr": CONFIG["fine_encoder_lr"]},
        {"params": head.parameters(), "lr": CONFIG["fine_head_lr"]}], weight_decay=CONFIG["weight_decay"])
    y = torch.tensor([r["rating"] for r in fit_rows], dtype=torch.long)
    weights = rating_weights(y.numpy())
    resume = directory/"fine_resume.pt"
    state = {"epoch": 1, "offset": 0, "history": [], "best_metrics": None, "elapsed": 0., "stale": 0}
    if resume.exists():
        state = load_torch(resume)
        block.load_state_dict(state["block"]); head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
    else:
        # Profile a real forward/backward pass, restoring all weights before fit.
        before_block = copy.deepcopy(block.state_dict()); before_head = copy.deepcopy(head.state_dict())
        began = time.perf_counter()
        x = fine_features(tokenizer,encoder,[r["text"] for r in fit_rows[:CONFIG["fine_batch"]]],view)
        loss_rows(head,x,y[:len(x)]).mean().backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in list(block.parameters())+list(head.parameters()))
        profile = {"batch_seconds": time.perf_counter()-began, "rss_bytes": psutil.Process().memory_info().rss,
                   "available_bytes": psutil.virtual_memory().available, "finite_gradients": bool(finite),
                   "encoder_trainable_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad)}
        block.load_state_dict(before_block); head.load_state_dict(before_head); optimizer.zero_grad(set_to_none=True)
        del x, before_block, before_head
        write_json(directory/"fine_profile.json",profile)
        if not finite or profile["rss_bytes"]>CONFIG["profile_max_rss_bytes"] or profile["available_bytes"]<CONFIG["minimum_free_memory_bytes"]:
            result = {"status": "skipped_resource_profile", "profile": profile, "completed_epochs": 0}
            write_json(finished,result)
            return result
    torch.manual_seed(CONFIG["seed"])
    if "rng" in state:
        torch.set_rng_state(state["rng"])
    last_tick = time.perf_counter()
    while state["epoch"]<=CONFIG["fine_epochs"] and state["stale"]<1:
        epoch = state["epoch"]
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(CONFIG["seed"]+epoch))
        encoder.eval(); block.train(); head.train()
        for offset in range(state["offset"],len(y),CONFIG["fine_batch"]):
            batch = order[offset:offset+CONFIG["fine_batch"]]
            optimizer.zero_grad(set_to_none=True)
            x = fine_features(tokenizer,encoder,[fit_rows[i]["text"] for i in batch.tolist()],view)
            loss = (loss_rows(head,x,y[batch])*weights[y[batch]-1]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite partial-fine-tuning loss.")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(list(block.parameters())+list(head.parameters()),1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite partial-fine-tuning gradient.")
            optimizer.step()
            state["offset"] = offset+len(batch)
            state["elapsed"] += time.perf_counter()-last_tick; last_tick = time.perf_counter()
            if offset//CONFIG["fine_batch"]%200==0 or state["offset"]==len(y):
                state.update(block=block.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state())
                save_torch(resume,state)
                print(f"Partial MiniLM epoch {epoch}/{CONFIG['fine_epochs']}: {state['offset']:,}/{len(y):,}",flush=True)
            if state["elapsed"]>=CONFIG["fine_budget_seconds"]:
                result = {"status": "budget_stopped", "completed_epochs":len(state["history"]),
                          "history":state["history"],"candidate":candidate,"kind":kind,"view":view}
                if state["best_metrics"] is not None:
                    result.update(development=state["best_metrics"],best_epoch=state["best_epoch"])
                write_json(finished,result)
                return result
        result = metrics([r["rating"] for r in dev_rows],fine_predict(tokenizer,encoder,head,dev_rows,view))
        if state["best_metrics"] is None or score(result)>score(state["best_metrics"]):
            state.update(best_metrics=result,best_epoch=epoch,stale=0)
            save_torch(directory/"fine_best.pt",{"block":block.state_dict(),"head":head.state_dict(),"kind":kind,"view":view})
        else:
            state["stale"] += 1
        state["history"].append({"epoch":epoch,"development":result})
        print(f"Partial MiniLM epoch {epoch}: accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f}",flush=True)
        state.update(epoch=epoch+1,offset=0,block=block.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state())
        save_torch(resume,state)
    result = {"status":"completed","completed_epochs":len(state["history"]),"history":state["history"],
              "development":state["best_metrics"],"best_epoch":state["best_epoch"],"candidate":candidate,"kind":kind,"view":view}
    write_json(finished,result)
    return result


def temperature_probabilities(p, temperature):
    if temperature<=0:
        raise ValueError("Temperature must be positive.")
    logits = np.log(np.maximum(validate_probabilities(p),1e-12))/temperature
    logits -= logits.max(1,keepdims=True)
    probabilities = np.exp(logits)
    return probabilities/probabilities.sum(1,keepdims=True)


def confidence_metrics(y, p, threshold):
    predicted = p.argmax(1)+1
    accepted = p.max(1)>=threshold
    return {"samples":len(y),"confident_samples":int(accepted.sum()),"coverage":float(accepted.mean()),
            "confident_accuracy":float((predicted[accepted]==np.asarray(y)[accepted]).mean()) if accepted.any() else None,
            "overall_accuracy":float((predicted==y).mean()),"automatic_actions":False}


def calibrate(rows, p):
    import hashlib
    fold = np.array([int(hashlib.sha256(("v4-cal:"+r["product"]).encode()).hexdigest()[:8],16)%2 for r in rows])
    if min((fold==0).sum(),(fold==1).sum())<100:
        raise ValueError("Insufficient product-disjoint calibration support.")
    y = np.array([r["rating"] for r in rows])
    temperature = min(CONFIG["temperatures"],key=lambda t:metrics(y[fold==0],temperature_probabilities(p[fold==0],t))["nll"])
    calibrated = temperature_probabilities(p,temperature)
    candidates = {str(t):confidence_metrics(y[fold==1],calibrated[fold==1],t) for t in CONFIG["confidence_thresholds"]}
    eligible = [float(t) for t,m in candidates.items() if m["coverage"]>=CONFIG["minimum_confident_coverage"]
                and m["confident_accuracy"]>=CONFIG["minimum_confident_accuracy"]]
    threshold = min(eligible) if eligible else None
    return {"temperature":temperature,"threshold":threshold,"threshold_candidates":candidates,
            "temperature_rows":int((fold==0).sum()),"threshold_rows":int((fold==1).sum()),
            "product_overlap":0,"target_scope":"85% observed accuracy at >=20% coverage on threshold fold; advisory only, not overall accuracy",
            "before_temperature_nll":metrics(y[fold==0],p[fold==0])["nll"],
            "after_temperature_nll":metrics(y[fold==0],calibrated[fold==0])["nll"]}
