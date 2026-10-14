# Learn the separate V3 feedback experiment

This explains the existing frozen experiment implemented in `src/intentdesk/feedback_v3.py`. It does not change its source, settings, labels or evaluation boundaries. Read its actual result in the completed run report; new feature engineering is not automatically an improvement.

## Task and representation

Predict one of five review-star classes from review text. This is supervised classification, distinct from Lilly’s numerical sales regression. Stars are ordered, but this experiment uses a nominal classifier rather than an ordinal loss. The lab's existing numeric regression, transformers and unsupervised outputs remain separate comparisons.

The raw Amazon subsets contain 4,629,544 reviews, while each V3 fitting pass uses 2,912,393 audited eligible fit reviews. Product-group partitioning and normalized exact-text checks precede training. Training counts must never be substituted for raw download size in a portfolio claim.

V3 decodes HTML entities, removes markup/video markers and normalizes case and whitespace. It represents full text through word 1–2-grams and character-within-word 3–5-grams. Word sequences capture short phrases; character sequences can help with morphology or spelling variation. Neither creates deep semantic understanding.

A hash maps each n-gram to one of \(2^{18}\) buckets per view. Concatenating the views gives 524,288 columns. Sparse matrices store only nonzero counts. There is no learned vocabulary requiring all review texts in memory, but different terms can collide in the same bucket. This trades interpretability and collisions against predictable memory. See the official [HashingVectorizer documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html).

## Fit-only inverse document frequency

Let \(N\) be the number of fitting documents, \(df_j\) the number containing hash bucket j, and \(c_{ij}\) its within-document count. The experiment uses

\[
idf_j=\log\frac{1+N}{1+df_j}+1,\qquad
v_{ij}=\begin{cases}(1+\log c_{ij})idf_j&c_{ij}>0\\0&c_{ij}=0.\end{cases}
\]

Document frequency counts a bucket once per document, not once per occurrence. Hash collisions are already combined before counting. The two views receive joint L2 normalization:

\[
x_i=\frac{v_i}{\|v_i\|_2}.
\]

Sublinear term frequency prevents repeated words from dominating as quickly as raw counts; IDF reduces the relative weight of common buckets; normalization controls document magnitude. IDF is fitted only on the training partition. It is unsupervised preprocessing, yet fitting it on future evaluation texts would still violate this experiment's inductive data contract.

## Classification, gradients and averaging

`SGDClassifier(loss="log_loss")` learns regularized linear classifiers incrementally through `partial_fit`. Its multiclass construction is one-versus-rest; it is not a transformer or a jointly fitted softmax network. For a binary subproblem, a linear score a and sigmoid probability p give

\[
a=w^Tx+b,\quad p=\frac1{1+e^{-a}},\quad
\ell=-[y\log p+(1-y)\log(1-p)],\quad
\nabla_w\ell=(p-y)x.
\]

Gradient updates move weights to reduce weighted log loss while an L2 penalty constrains their magnitude. `average=True` maintains averaged SGD weights. This can stabilize noisy incremental optimization; it does not guarantee calibrated probabilities or generalization. The declared alpha is \(10^{-6}\), batches contain 256 reviews, and CPU work is bounded to two threads. Consult the official [SGDClassifier documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.SGDClassifier.html).

## Class imbalance and objective choice

If class c has fitting count \(n_c\), define its fitting prior \(\pi_c=n_c/N\). The experiment compares two weight exponents:

\[
u_c=\left(\frac{N}{5n_c}\right)^\gamma,\quad
w_c=\frac{u_c}{\sum_k\pi_k u_k},\quad
\gamma\in\{1,\tfrac12\}.
\]

Normalizing ensures expected fitting weight is one, so changing the exponent does not silently change overall loss scale relative to regularization. Full balancing emphasizes rare stars more; moderate balancing is a compromise. The unbalanced development distribution is retained to reveal the actual tradeoff.

Accuracy counts correct predictions, while macro-F1 gives each star class equal importance. A model predicting the most frequent star can achieve seemingly strong accuracy while failing minority stars. The declared selector uses development macro-F1, with accuracy as a tie-break, and retains V2 unless the gain is at least .005. A more accurate but lower-macro-F1 candidate cannot replace it merely because the headline looks better.

Ordinal diagnostics also report mean absolute star error \(N^{-1}\sum|y_i-\hat y_i|\) and neighboring-star errors. A one-star error and a four-star error are different operationally, although nominal accuracy counts both as wrong.

## Epochs, checkpoints and final evaluation

An epoch is one pass through eligible fitting reviews. Maximum three, minimum two, patience one and earliest-best selection were fixed before fresh evaluation. Repeated training can overfit; “more epochs” is not a universal improvement. Checkpoints support resumption, while an OS lock prevents simultaneous jobs from writing the same artifacts.

The fresh 20,000-review check excludes every product appearing in either consumed earlier check. This creates a new held-out population and can shift class/product difficulty. Compare all fixed candidates **on that same check**, rather than comparing a new score to an old score on different products. Development was already exposed and is not described as untouched.

Source, configuration, dependency lock, audited dataset and earlier experiment bindings are checked. The selected classifier is fixed before evaluation; the final marker records exposure and prevents resetting a consumed check to chase a desired result. No 90% result or business benefit is guaranteed.

An ordinal successor such as [CORAL](https://arxiv.org/abs/1901.07884) is a legitimate future research direction, not an implemented V3 claim. It needs a separately declared experiment and new evaluation products. The same applies to semantic encoders, new labels or longer training.

## Explain this without reading

Why is IDF fit-only even though it uses no star labels? Why normalize class weights? Why is the moderate-weight model's accuracy not sufficient to promote it? What information is lost by hashing? Why are source hashes and test-exposure markers useful? Why can a negative experiment result be good engineering evidence?
