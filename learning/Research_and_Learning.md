# How research and ML concepts become IntentDesk

This is an implementation and learning map, not a claim that every feature exists. Stage 1 built the data audit and lexical baseline. The local expansion adds frozen MPNet features, a linear control, small/large PyTorch heads and a separate MS MARCO corpus acquisition pipeline. See [the neural lesson](Local_Neural_Experiment.md), [the calibration/review lesson](Calibration_and_Review.md), and [the local app/search lesson](Review_App_and_Search.md). Calibration and independently checked review recommendations are implemented; the OOS gate failed and routing remains review-all. Full-corpus lexical retrieval and its development benchmark are now implemented. Final classifier evaluation is consumed; generation remains a future extension. MPNet replaces the originally proposed MiniLM encoder.

## How we find and choose research

Current scope extension: [Milestone 6](../reports/Milestone_06.md) and [Scope Detection](Scope_Detection.md) connect cosine similarity, neighbor reference separation, feature standardization, sigmoid/binary cross-entropy, class weighting, regularization and exact binomial uncertainty to an implemented detector. The original encoder/classifier stay frozen. This adds a research advisory; it does not establish a fresh independent joint test or authorize actions. Its check is now consumed.

Start with the engineering question: for example, how can we detect unsupported requests, or measure whether a confidence score is trustworthy? Search relevant terms in ACL Anthology (NLP), PMLR/ICML, NeurIPS proceedings, OpenReview and arXiv. Scholar or Semantic Scholar can help locate a work; verify the actual paper on its primary publication page. Distinguish peer-reviewed papers from preprints. Follow references and later papers when they address the same problem.

For each adopted method, read the method/evaluation sections, note assumptions and hardware/data requirements, locate official code if available, and design a bounded comparison. Record both supportive and negative findings. An abstract is useful for triage; it is not sufficient to claim a reproduction.

We will not select an architecture because a paper is fashionable or its benchmark score is high. Ask: does it fit this dataset, our CPU budget, the learning objective and the observed failure? Published improvements are hypotheses to test locally, not promised results.

## Verified starting references

The primary pages/model card below were checked on 10 October 2026. This initial map records paper metadata/abstract-level review and the model card; full-method reading accompanies implementation of each later technique. No claim is made that the published experiments have been reproduced.

| Source | Why it matters | Our planned experiment | Current implementation status |
| --- | --- | --- | --- |
| Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019. [Paper](https://aclanthology.org/D19-1131/), [data/code](https://github.com/clinc/oos-eval) | Defines a task with supported intents and unsupported requests | Measure classification and, later, OOS rejection separately | Data acquisition, contract and split audit built |
| Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP 2019. [Paper](https://aclanthology.org/D19-1410/) | Sentence embeddings and similarity | Compare pretrained sentence features against TF-IDF | Frozen feature comparison implemented; no reproduction of SBERT training |
| [all-mpnet-base-v2 official model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) — model artifact, not a new paper | Pinned 768-dimensional sentence representation | Frozen encoder with linear/small/large classifiers | Implemented locally; tokenizer, weights and revision checksummed; 128-token cap documented |
| Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. [Published-paper record](https://arxiv.org/abs/1711.05101) | Distinguishes AdamW weight decay from adaptive-gradient L2 regularization | Fixed head training now; bounded regularization study only if later evidence warrants it | AdamW training implemented; no claim of isolating its effect against another optimizer |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. [Paper](https://proceedings.mlr.press/v70/guo17a.html) | Confidence calibration and temperature scaling | Compare metrics after fitting temperature on separate calibration data | Implemented; method section reviewed. B2 improved on the independent known-class check; N1 slightly worsened. OOS rejection failed. |
| *KNN-Contrastive Learning for Out-of-Domain Intent Classification*, ACL 2022. [Paper](https://aclanthology.org/2022.acl-long.352/) | Research context for neighborhood information in OOS detection | Test a bounded frozen-feature similarity advisory | Metadata/abstract reviewed only. Our six-feature logistic detector does not reproduce the paper's contrastive training or reported experiments. |
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. [Paper](https://proceedings.nips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) | Research background for combining retrieved evidence and generation | Compare evidence-only, template and LLM-drafted answers on a separate query suite | Later extension; our retrieve-then-draft app does not reproduce the trained RAG architecture |

Implementation references are a different kind of evidence: [scikit-learn pipelines/leakage](https://scikit-learn.org/stable/common_pitfalls.html), [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [PyTorch ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Record library versions in the lockfile and each run.

## Concepts, code and observable evidence

| Concept | Where it belongs | Evidence to produce |
| --- | --- | --- |
| Problem definition / supervised learning | A request has a known intent label | Data card, label contract, task boundaries |
| NLP / tokenization | Word and character TF-IDF; later subword tokenization in the encoder | Vocabulary, n-gram explanation, truncation/error examples |
| Linear algebra / sparse matrices | Text becomes a vector; classifier computes weighted scores | Trace dimensions; keep large TF-IDF matrices sparse |
| Optimization | Fit classifier coefficients and minibatch neural updates | Solver convergence, epoch histories and finite-gradient tests implemented |
| Representation / transfer learning | Pretrained encoder supplies frozen sentence features | Compare lexical and embedding models; say exactly which weights changed |
| Neural network / activation | Newly initialized 768 → 128 → 150 and 768 → 1024 → 512 → 150 heads with ReLU | Actual PyTorch training, saved weights and verified reload parity |
| Backpropagation | Compute how each trainable weight contributes to loss | Numerical gradient exercise and a finite-gradient check |
| Regularization / bias and variance | Model complexity, weight decay, dropout if justified | Train/tune curves and a small predeclared comparison |
| Hyperparameter selection | Development-only model comparisons | Trial ledger with bounded search, no final-test selection |
| Generalization / leakage | Separate fit, tune, calibrate, validation and final test | Manifest, overlap checks and provenance hashes |
| Metrics / statistics | Accuracy, macro-F1, loss, Brier, intervals | Counts, conventions, per-class errors; paired uncertainty later |
| Calibration | Adjust confidence after weights freeze | Reliability diagram, Brier/log loss before/after |
| Decision making / abstention | Decide automatic route versus human review | Coverage/error/cost curves; empirical OOS false acceptance |
| Novelty / OOS detection | Separate likely unsupported requests using frozen similarity and confidence features | Versioned binary model, disjoint references, fresh OOS check plus disclosed reused known check; all requests still reviewed |
| Explainability / error analysis | Understand confused classes and important lexical features | Reviewed mistakes, coefficient inspection and failure taxonomy |
| Information retrieval | Search the separate 8.84M-passage research corpus using disk-based BM25 | Fixed 100-query development sample, MRR@10, recall@10, latency and incomplete-judgment limits |
| Generative AI / RAG | Draft an answer using retrieved passages | Manual evidence-support rubric, citations, refusal failures |
| ML engineering | Package inference separately from training | Reload parity, tested local API, versioned artifacts, browser demo; CI/container are future work |
| SQL / feedback | Store and review corrections before possible future training | Queries, schema and review status; no silent online learning |

This is broad applied ML coverage, not every ML concept. Reinforcement learning, computer vision, causal inference, distributed training and recommendation systems need different tasks. Add a method only when the problem or a meaningful learning experiment warrants it.

## The mathematics, in context

### Stage 1: linear text classifier

TF-IDF produces a sparse feature vector `x`. A linear classifier produces one score for each supported intent:

```text
z = W x + b
p_k = exp(z_k) / sum_j exp(z_j)
L = -(1/N) sum_i log(p_i,y_i) + regularization
```

Softmax is computed with a numerically stable shift by `max(z)`. The exact regularization scaling is library-specific; the configured `C` controls the inverse strength for this baseline. Training changes `W` and `b`; fitting TF-IDF also learns its vocabulary/IDF from fit data. Inference must not refit those transformations.

Even an unsupported request receives a highest-scoring known label. Therefore a top score is not proof the request is supported. Stage 1 returns suggestions for review and disables automatic routing.

### Neural milestone: add a nonlinear head

```text
x: frozen MPNet embedding, dimension 768
h = ReLU(W1 x + b1)                 # 128 hidden units
ReLU(a) = max(0, a)
z = W2 h + b2                      # 150 logits
p = softmax(z)                     # displayed class probabilities
L = -log(p_true_class)             # cross-entropy for one example
dL/dz = p - one_hot(true_class)
theta_next = theta - learning_rate * gradient   # elementary SGD example
```

ReLU supplies nonlinearity in the hidden layer. Softmax turns output logits into competing class probabilities; independent sigmoid outputs are not the starting design for this single-intent multiclass task. Pass raw logits into PyTorch CrossEntropyLoss, which performs the required normalization internally.

The encoder remains frozen; we train only the new head. Dropout and AdamW are bounded candidates, not decorations or assured improvements. We will check learning curves and compare with the same embeddings plus a linear classifier to isolate whether the head helps.

### Decision milestone: probabilities are not actions

```text
coverage = accepted_requests / all_requests
selective_error = wrong_accepted_requests / accepted_requests
OOS_false_accept = accepted_OOS_requests / all_OOS_requests
illustrative_cost = (10 * wrong_accepts + 1 * human_referrals) / all_requests
```

Choose a threshold on development validation under the frozen mixture/cost assumptions in the handoff. These costs are simulated units, not actual money. Show uncertainty and counts; zero accepted requests means selective error is undefined, not a successful zero-error automatic system. Use explicit fallback when evidence is insufficient.

## Graphs have jobs

Stage 1 generates a majority-versus-TF-IDF score comparison, fit-request length histogram and common development confusion pairs. Later stages add:

- Training/tuning loss curves: diagnose optimization and overfitting.
- Learning curves versus training-set size: ask whether more examples could help.
- Calibration/reliability plots: compare confidence with observed correctness.
- Coverage versus selective error and OOS acceptance: show the routing tradeoff.
- Latency/model-size comparisons: assess the practical cost of complexity.

Do not invent neural learning curves for a linear-only milestone. Label every figure with its data split and model version. Final-test figures come only after freezing the evaluated system.

## How you demonstrate understanding

Each milestone includes a short explanation, executable code, measured result and a learning exercise. Begin with `math_exercise.py`: it checks the softmax/cross-entropy gradient numerically and shows a tiny SGD step lowering a toy loss. This is a three-class educational exercise, not IntentDesk neural training.

You should be able to modify a preprocessing step, write a small NumPy or SQL operation, explain a loss curve, and diagnose an incorrect prediction yourself. Keep an honest log of AI assistance and your verification. A source citation or generated equation alone does not establish understanding.

