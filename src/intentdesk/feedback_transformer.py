"""Randomly initialized word transformers with explicit multi-head attention.

No pretrained tokenizer, embeddings or transformer weights are loaded here.
"""

from collections import Counter
import copy
import math
from pathlib import Path
import re
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn

from intentdesk.common import read_json, sha256_file, write_json


CONFIG = {"seed": 20261013, "vocabulary_limit": 20000, "length": 64, "batch_size": 64,
          "epochs": 30, "minimum_epochs": 20, "patience": 5, "learning_rate": .0007, "weight_decay": .01,
          "dropout": .1, "threads": 2,
          "architectures": {"scratch_small": {"dimension": 64, "heads": 4, "layers": 1},
                            "scratch_large": {"dimension": 128, "heads": 4, "layers": 3}},
          "selection": "Highest development macro-F1, then accuracy, then lowest loss, earliest tie; final test never selects epochs",
          "token_budget": "CLS plus first 63 regex tokens; truncation counts disclosed"}


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text.casefold())


def vocabulary(texts):
    counts = Counter(word for text in texts for word in words(text))
    ranked = sorted(counts, key=lambda word: (-counts[word], word))[:CONFIG["vocabulary_limit"]-3]
    return {word: i+3 for i, word in enumerate(ranked)}


def tokenize(texts, vocab, length=64):
    output = np.zeros((len(texts), length), dtype=np.int64); truncated = 0
    for i, text in enumerate(texts):
        tokens = words(text); truncated += len(tokens) > length-1
        ids = [2]+[vocab.get(word, 1) for word in tokens[:length-1]]
        output[i, :len(ids)] = ids
    return torch.from_numpy(output), int(truncated)


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimensions must divide into heads.")
        self.heads, self.head_dimension = heads, dimension//heads
        self.qkv = nn.Linear(dimension, dimension*3)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dimension).permute(2,0,3,1,4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2,-1))/math.sqrt(self.head_dimension)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        weights = self.dropout(scores.softmax(-1))
        context = (weights @ value).transpose(1,2).reshape(batch,length,dimension)
        return self.output(context)


class Block(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.attention = MultiHeadAttention(dimension, heads, dropout)
        self.feedforward = nn.Sequential(nn.Linear(dimension, dimension*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension*2, dimension))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        x = x+self.dropout(self.attention(self.norm1(x), valid))
        return x+self.dropout(self.feedforward(self.norm2(x)))


class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, dimension, heads, layers, length=64, dropout=.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.positions = nn.Embedding(length, dimension)
        self.blocks = nn.ModuleList(Block(dimension, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, 5)

    def forward(self, ids):
        valid = ids != 0
        if not valid.any(1).all():
            raise ValueError("Transformer rows must include a CLS token.")
        x = self.embedding(ids)+self.positions(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x)
        pooled = (x*valid.unsqueeze(-1)).sum(1)/valid.sum(1, keepdim=True)
        return self.classifier(pooled)


def configure():
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)


def probabilities(model, tokens):
    model.eval(); result = []
    with torch.inference_mode():
        for batch in tokens.split(CONFIG["batch_size"]):
            result.append(model(batch).softmax(-1).numpy())
    return np.concatenate(result)


def development_result(model, tokens, labels):
    p = probabilities(model, tokens)
    labels = np.asarray(labels)
    predicted = p.argmax(1)+1
    loss = float(-np.log(np.maximum(p[np.arange(len(p)), labels-1], 1e-12)).mean())
    accuracy = float((predicted == labels).mean())
    macro_f1 = float(f1_score(labels, predicted, labels=[1,2,3,4,5], average="macro", zero_division=0))
    return (macro_f1, accuracy, -loss), {"development_loss": loss,
        "development_accuracy": accuracy, "development_macro_f1": macro_f1}


def can_stop(completed_epochs, stale):
    minimum = min(CONFIG["minimum_epochs"], CONFIG["epochs"])
    return completed_epochs >= minimum and stale >= CONFIG["patience"]


def fit_candidate(directory: Path, name, fit_texts, y_fit, development_texts, y_development, vocab):
    configure()
    architecture = CONFIG["architectures"][name]
    model = ScratchTransformer(len(vocab)+3, **architecture, length=CONFIG["length"], dropout=CONFIG["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    fit, truncated_fit = tokenize(fit_texts, vocab, CONFIG["length"])
    development, truncated_development = tokenize(development_texts, vocab, CONFIG["length"])
    targets = torch.as_tensor(np.asarray(y_fit)-1, dtype=torch.long)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    checkpoint = directory/(name+"_resume.pt")
    if checkpoint.exists():
        receipt = read_json(checkpoint.with_suffix(".json"))
        if receipt["sha256"] != sha256_file(checkpoint):
            raise ValueError("Transformer resume checksum changed.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history, best_loss, best_epoch, stale = state["history"], state["best_loss"], state["best_epoch"], state["stale"]
        best_state = state["best_state"]; torch.set_rng_state(state["torch_rng"])
        if state.get("selection_version") == "macro_f1_accuracy_loss_v2":
            best_score = tuple(state["best_score"])
        else:
            # Preserve the old loss-selected and last checkpoints; do not invent
            # unavailable intermediate weights or their macro-F1 measurements.
            current_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            best_score, old_best = development_result(model, development, y_development)
            history[best_epoch-1].update(old_best)
            model.load_state_dict(current_state)
            current_score, current_metrics = development_result(model, development, y_development)
            history[-1].update(current_metrics)
            if current_score > best_score:
                best_score, best_epoch, best_state = current_score, len(history), current_state
            best_loss, stale = -best_score[2], 0
    continued_from_epochs = len(history)
    started = time.perf_counter()
    for epoch in range(len(history)+1, CONFIG["epochs"]+1):
        if can_stop(len(history), stale):
            break
        model.train(); total, correct = 0., 0
        generator = torch.Generator().manual_seed(CONFIG["seed"]+epoch)
        order = torch.randperm(len(fit), generator=generator)
        for index, batch in enumerate(order.split(CONFIG["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit[batch]); loss = nn.functional.cross_entropy(logits, targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite scratch-transformer loss.")
            loss.backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite scratch-transformer gradient.")
            optimizer.step()
            total += float(loss.item())*len(batch); correct += int((logits.argmax(1)==targets[batch]).sum())
            if index and index % 300 == 0:
                print(f"{name} epoch {epoch}: {min((index+1)*CONFIG['batch_size'],len(fit)):,}/{len(fit):,} trained", flush=True)
        score, development_metrics = development_result(model, development, y_development)
        dev_loss = development_metrics["development_loss"]
        history.append({"epoch": epoch, "fit_loss": total/len(fit), "fit_accuracy": correct/len(fit),
            **development_metrics,
            "last_gradient_norm_before_clip": float(norm)})
        print(f"{name} epoch {epoch}: fit loss {total/len(fit):.4f}, development loss {dev_loss:.4f}, accuracy {score[1]:.2%}, macro-F1 {score[0]:.4f}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
        if score > best_score:
            best_score, best_loss, best_epoch, stale = score, dev_loss, epoch, 0; best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            "best_loss": best_loss, "best_epoch": best_epoch, "stale": stale, "best_state": best_state,
            "selection_version": "macro_f1_accuracy_loss_v2", "best_score": list(best_score),
            "torch_rng": torch.get_rng_state()}, temporary)
        temporary.replace(checkpoint); write_json(checkpoint.with_suffix(".json"), {"sha256": sha256_file(checkpoint)})
    model.load_state_dict(best_state); model.eval()
    path = directory/(name+".pt"); torch.save(model.state_dict(), path)
    restored = ScratchTransformer(len(vocab)+3, **architecture, length=CONFIG["length"], dropout=CONFIG["dropout"])
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); restored.eval()
    with torch.inference_mode():
        if not torch.allclose(model(development[:8]), restored(development[:8]), atol=1e-7, rtol=0):
            raise ValueError("Scratch-transformer checkpoint reload mismatch.")
    return model, {"architecture": architecture, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initialized_from_pretrained": False, "history": history, "best_epoch": best_epoch,
        "continued_from_epochs": continued_from_epochs, "minimum_epochs": min(CONFIG["minimum_epochs"],CONFIG["epochs"]),
        "maximum_epochs": CONFIG["epochs"], "checkpoint_selection": CONFIG["selection"],
        "fit_rows": len(fit), "truncated_fit": truncated_fit, "truncated_development": truncated_development,
        "elapsed_this_invocation_seconds": time.perf_counter()-started, "sha256": sha256_file(path)}
