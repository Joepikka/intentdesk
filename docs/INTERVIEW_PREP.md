# Understand and defend the project

This preparation pack is not evidence that the author has mastered its content. Complete the exercises yourself, save your answers in a local learning journal, and explain the project without reading this document.

## Trace one prediction

1. Run `intentdesk predict "I lost my credit card"`. Trace validation and lexical features in `predict.py` and `train.py`.
2. Trace frozen token embeddings, attention-mask mean pooling and normalization in `neural.py`.
3. Follow logits through softmax, temperature calibration and thresholds in `policy.py`.
4. Follow similarity/confidence features through the binary advisory in `scope.py`.
5. Follow HTTP input and explicit decision storage through `app.py` and `reviews.py`.
6. Explain the six-category projection and manual/assisted protocol in `pilot.py`.

## Equations you should explain

For logits z and temperature T > 0:

```text
p_i = exp(z_i / T) / sum_j exp(z_j / T)
cross_entropy = -log(p_true)
d(cross_entropy)/d(z_i) = p_i - 1[i = true]     # when T = 1
ReLU(x) = max(0, x)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Explain dimensions, numerical stability, what gets averaged in macro-F1, and why temperature scaling preserves the largest logit. Discuss why it does not fix all unsupported-request errors.

Run `uv run --locked python learning/math_exercise.py`. Explain the analytic and finite-difference gradients, then change the toy learning rate and describe the effect. This is a toy mathematical check, not a new benchmark experiment.

## Questions and answer checkpoints

| Question | Your explanation should include |
| --- | --- |
| Why split before fitting TF-IDF? | Vocabulary/IDF leakage; fit-only statistics |
| What did you train? | Lexical/embedding classifiers and heads; encoder frozen |
| Why did bigger not win? | Observed outcome versus possible explanations; capacity, regularization and limited labels; no causal proof from one run |
| What do calibration and abstention change? | Score reliability versus decision policy; separate fitting, selection and check partitions |
| Why is 96.24% insufficient? | Known-intent benchmark excludes the open-world support problem; OOS and distribution shift |
| Why keep final results frozen? | Selection after exposure invalidates independent evaluation |
| What does 3/100 OOS acceptance mean? | Small sample, 7.57% one-sided upper bound; reused known check |
| Why not say the pilot saves time? | No completed human comparison at publication; fictional cases and non-independent labels |
| Why is retrieval separate? | Different corpus/task/terms; a ranked passage is not verified support guidance |
| What must change before hosting? | Authentication, authorization, privacy, capacity, monitoring and operations; current app is loopback-only |

## Demonstrate your own contribution

- [ ] Run the tests and explain three meaningful checks.
- [ ] Draw the data partitions and mark every consumed evaluation set.
- [ ] Explain one correct and one incorrect development prediction without tuning against final examples.
- [ ] Complete one genuine human pilot session with interruptions marked.
- [ ] Make a small, separately tested change you can defend, such as a report display improvement; preserve frozen model sources and study bindings.
- [ ] Record the five-minute demo with your own explanation and AI-assistance disclosure.

Leave these unchecked until completed. The assistant cannot complete your learning or a human study on your behalf. Existing [lessons and research map](../learning/Research_and_Learning.md) supply deeper material.
