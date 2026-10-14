# Explain and modify the feedback lab yourself

Implemented code is project evidence, not proof that its author mastered every concept. Complete these exercises yourself and keep your own answers. Disclose AI assistance.

1. Trace a review through product split, rating cleanup, duplicate fingerprint, vector, prediction and confusion-matrix cell. Explain why target ratings and product IDs are not model features.
2. Calculate TF-IDF for three short texts. Explain why fitting vocabulary or SVD on test text contaminates evaluation.
3. Explain why numeric rating regression is meaningful under an equal-spacing assumption, while regression on arbitrary intent IDs is not. Calculate MAE.
4. Explain logistic regression, SVM, trees, forests and KNN: objective/decision rule, representation, complexity and failure modes.
5. Derive attention shapes: batch×heads×tokens×head-dimension, then batch×heads×tokens×tokens. Explain square-root scaling and padding masks.
6. Run the uniform-attention and padding-invariance tests. On a separate learning copy, break the mask and explain the resulting error. Preserve frozen experiment sources.
7. Explain which scratch-transformer weights train and how this differs from frozen MPNet. Interpret fit/development curves without selecting another checkpoint using final results.
8. Explain KNN distance concentration/latency and why trees use reduced rather than dense vocabulary features here.
9. Explain how K-means fits without rating targets even though the sample was balanced using ratings. Discuss that sampling bias and why silhouette is not topic accuracy.
10. Explain why full-fit streaming and bounded-sample model scores are not isolated algorithm comparisons. Design a fair same-budget follow-up with a new protocol.
11. Interpret class supports, errors and remaining leakage limits before quoting aggregate scores.
12. Make a new-version change with a hypothesis, training-only implementation, development selection, frozen choice and untouched final evaluation. Record negative findings too.

A useful demonstration includes your own explanation and a small working modification. Dataset bytes, model size and passing tests do not substitute for understanding.
