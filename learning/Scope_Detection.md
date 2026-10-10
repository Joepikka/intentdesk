# Why a classifier also needs a scope detector

A 150-class softmax always assigns its probability mass among those 150 labels. It cannot tell you, by itself, that an entirely different request is unsupported. A motorcycle-shopping request can receive `order` as its nearest label even though that category is inappropriate. Scope detection asks a separate binary question before treating a known category as plausible.

## The implemented model

Let `e(x)` be the frozen, normalized 768-dimensional MPNet embedding. For normalized vectors, cosine similarity is their dot product:

```text
cos(e, r) = e^T r
```

The reference contains 80% of each original fit class. The other 20% supply known training queries for the scope detector. This separation prevents a training query from using itself as a perfect nearest neighbor. Class prototypes and the original classifier remain those learned from the original fit allocation.

Build a six-dimensional vector:

```text
phi(x) = [max_reference_cosine,
          mean_of_top_5_reference_cosines,
          max_class_prototype_cosine,
          max_known_class_probability,
          top_probability - second_probability,
          -sum_k(p_k * log(p_k)) / log(150)]
```

The last feature is normalized entropy: near zero for a concentrated known-label distribution and near one for a uniform distribution. All six features can still be misleading on particular unknown requests; they are signals to evaluate, not universal rules.

StandardScaler learns its mean and scale from detector training rows only. A balanced logistic regression then computes:

```text
u_j = (phi_j - training_mean_j) / training_scale_j
z = w^T u + b
s = sigmoid(z) = 1 / (1 + exp(-z))

weighted_loss = -sum_i weight[y_i] *
                [y_i*log(s_i) + (1-y_i)*log(1-s_i)] + L2_penalty
weight[class] = N / (2 * N_class)
```

The exact objective normalization is library-specific. `C=1` fixes inverse regularization strength. The classes are `1 = known` and `0 = OOS`. Class balancing prevents the many known rows from overwhelming 100 OOS training rows. It also changes the effective class prior: the returned `s` is an **uncalibrated support score**, not a real-world probability that a request is supported. Its sigmoid is different from the original 150-class softmax.

## Thresholds and evidence

The likely-OOS boundary is fixed by a 5th-percentile rule on known selection scores. The supported-candidate threshold is selected from a fixed grid, floored at that boundary, using known retention, OOS acceptance and a declared review-cost scenario. If no eligible threshold exists, the supported threshold is 1.1, which no sigmoid score can reach. The middle band is uncertain; in this run the two selected thresholds happen to coincide.

The model trains on 2,393 known fit queries plus the original unused 100 OOS training queries. Additional published OOS data supply 51 selection examples and 100 new check examples. Fingerprints exclude overlap with all original partitions, including the consumed final set, using the existing row manifest without reading final examples again. Case/spacing/punctuation grouping does not establish semantic independence.

The known check reuses 1,496 calibration rows. Because these rows already fitted the original model's temperature, the combined check must not be described as a fresh independent joint evaluation. Published crowd-authored OOS labels provide external provenance; they do not replace new semantic adjudication against a real product's scope.

The check found 3 supported suggestions among 100 fresh OOS queries. A 3% observed rate is not proof of a true rate below 5%: the one-sided 95% exact upper bound is approximately 7.57%. The new advisory also incorrectly flags 78/1,496 known queries. Both facts belong in the portfolio explanation.

## Research relationship

[Larson et al., EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1131/) defines the CLINC known/OOS task and highlights the gap between intent accuracy and OOS detection. The source extension is pinned to [the original repository revision](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de).

[Zhou, Liu and Qiu, ACL 2022](https://aclanthology.org/2022.acl-long.352/) motivates studying neighborhood structure for OOD intent detection. This project uses a simple frozen-feature nearest-neighbor baseline plus a binary classifier. It does **not** reproduce their contrastive training or density-based method. Only the publication record/abstract was consulted here; no paper-level reproduction claim is made.

## Exercises for your portfolio

1. Trace one request from tokenization to the six features. Explain every vector shape.
2. Explain why including a query in its own reference would corrupt the training signal.
3. Derive the sigmoid gradient and compare binary cross-entropy with multiclass cross-entropy.
4. Explain the class weights and why the raw score is not calibrated.
5. Calculate the observed OOS acceptance and explain why its confidence bound matters.
6. Inspect errors only as diagnostic evidence. Propose a new experiment with a fresh evaluation allocation; never silently retune this consumed check.

The user makes the final decision. An `oos` field prefill is a review suggestion, not an automatically saved label. Old records remain intact when Analyze again creates a new assessment. Future human corrections need consent, scope guidelines, adjudication and a separately versioned training/evaluation plan.
