"""Inference for review research; never substitutes review ratings for intent labels."""

from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json
from intentdesk.feedback_data import clean_rating_declarations
from intentdesk.feedback_lab import checked_metadata, directory_for
from intentdesk.feedback_models import predict_batches, streaming_vectorizer
from intentdesk.feedback_transformer import CONFIG, ScratchTransformer, configure, probabilities, tokenize
from intentdesk.predict import validate_text


class FeedbackModels:
    def __init__(self, root: Path):
        self.metadata = checked_metadata(root)
        self.directory = directory_for(root)
        self.features = joblib.load(self.directory/"features.joblib")
        self.regressor = joblib.load(self.directory/"linear_regression.joblib")
        self.clusters = joblib.load(self.directory/"clusters.joblib")
        self.vocab = read_json(self.directory/"vocabulary.json")
        self.models = {}

    def get(self,name):
        if name not in self.models:
            if name in CONFIG["architectures"]:
                configure()
                model = ScratchTransformer(len(self.vocab)+3,**CONFIG["architectures"][name],length=CONFIG["length"],dropout=CONFIG["dropout"])
                model.load_state_dict(torch.load(self.directory/(name+".pt"),map_location="cpu",weights_only=True))
                self.models[name] = model.eval()
            elif name != "natural_training_majority":
                self.models[name] = joblib.load(self.directory/(name+".joblib"))
        return self.models.get(name)

    def predict_many(self,name,texts,sparse=None,dense=None):
        if name == "natural_training_majority":
            value = self.metadata["development_classification"][name]["constant"]
            return np.full(len(texts),value)
        model = self.get(name)
        if name in CONFIG["architectures"]:
            tokens,_ = tokenize(texts,self.vocab)
            return probabilities(model,tokens).argmax(1)+1
        kind = self.metadata["development_classification"][name]["representation"]
        if kind == "hashing":
            features = streaming_vectorizer().transform(texts)
        elif kind == "sparse":
            features = sparse if sparse is not None else self.features["tfidf"].transform(texts)
        else:
            features = dense if dense is not None else self.dense_features(texts)
        return predict_batches(model,features)

    def dense_features(self,texts):
        with threadpool_limits(limits=2):
            return self.features["normalizer"].transform(self.features["svd"].transform(self.features["tfidf"].transform(texts)))

    def analyze(self,text):
        text = clean_rating_declarations(validate_text(text))
        if not text.strip(" .,!?:;"):
            raise ValueError("Describe a review in words rather than only its star rating.")
        dense = self.dense_features([text])
        selected = self.metadata["selected_classifier"]
        predicted = int(self.predict_many(selected,[text],dense=dense)[0])
        scratch_name = sorted(CONFIG["architectures"],key=lambda n:(-self.metadata["development_classification"][n]["macro_f1"],n))[0]
        scratch = int(self.predict_many(scratch_name,[text])[0])
        with threadpool_limits(limits=2):
            cluster = int(self.clusters.predict(dense)[0])
            rating = float(self.regressor.predict(dense)[0])
        _,truncated = tokenize([text],self.vocab)
        return {"experiment":self.metadata["run_id"],"selected_classifier":selected,"predicted_stars":predicted,
            "linear_regression_stars":rating,"regression_is_unclipped":True,
            "scratch_transformer":scratch_name,"scratch_predicted_stars":scratch,"scratch_input_truncated":bool(truncated),
            "scratch_token_budget":64,"cluster":cluster,
            "cluster_keywords":self.metadata["unsupervised"]["clusters"][cluster]["keywords"],
            "purpose":"Product-review research; stars and clusters are not support intents, urgency or verified facts",
            "automatic_actions":False,"human_review_required":True}
