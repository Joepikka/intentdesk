# Lesson 3: confidence is a decision input

Read [the measured milestone](../reports/Milestone_03.md) beside `src/intentdesk/policy.py`. Everything in this lesson corresponds to implemented code, except the explicitly stated next OOS-detector experiment.

## Why accuracy was insufficient

The classifier always scores 150 known labels. An unrelated request still receives one of those labels. Even a perfectly calibrated model on known-label requests can fail on unsupported requests. The CLINC [dataset paper](https://aclanthology.org/D19-1131/) treats out-of-scope prediction as a separate challenge; our 9/80 false-acceptance result demonstrates the distinction locally.

## Fit one temperature, freeze the weights

For logits `z`, use a positive scalar `T`:

```text
p_k(T) = exp(z_k/T) / sum_j exp(z_j/T)
T* = argmin_(T > 0) mean_i[-log(p_yi(T))]
```

We optimize `log(T)` between `log(0.05)` and `log(20)` on calibration rows only. If the fitted loss does not improve on the identity temperature, retain `T=1`. The model weights and class ordering stay fixed. Positive scaling preserves the largest logit, so top-label accuracy remains unchanged. `T<1` sharpens probabilities; `T>1` softens them.

This follows the scalar-temperature method in Guo et al., [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html), section 4.2. We reviewed the method section and applied it to our frozen candidates; we did not reproduce that paper's training experiments.

B2 was underconfident on known requests. Its fitted `T=0.4913` reduced independent-check log loss from 0.2844 to 0.1408. N1's small temperature adjustment slightly worsened its check loss. Fitting calibration loss does not guarantee improvement on another sample or on unknown requests.

## Semantic support is a different feature

From fit embeddings only, compute one normalized mean per intent:

```text
mu_c = mean of x_i whose fit label is c
v_c = mu_c / ||mu_c||_2
support(x) = x dot v_predicted_intent
```

All sentence embeddings and prototypes are normalized, so the dot product is cosine similarity. Its range is -1 to 1. It is a heuristic for resemblance to training requests, not a probability. It may confuse related unsupported questions with supported requests.

The proposed rule requires both calibrated maximum probability ≥ a confidence threshold and cosine support ≥ a similarity threshold. Selection chose 0.90 and 0.30 for B2. Its independent OOS check failed, so these thresholds are **not activated** for automatic recommendations.

## Prevent selection leakage

The existing fit/tune partitions trained and shortlisted models. The calibration partition fits temperatures. A fixed stratified 20/80 split of official validation gives policy selection and a separate check. Save the selected model, temperatures and thresholds before checking outcomes. Never revise them to make that same check pass.

The code writes `selection_freeze.json` before evaluating check outcomes. It records check exposure before evaluation so a crash cannot accidentally authorize another selection attempt. Completed reruns return the frozen result. This is experiment lifecycle control, not proof that public benchmark data was absent from upstream pretraining.

## Rates and the assumed workload

Our validation data has fewer than 20% OOS requests, so use explicit scenario weights rather than silently treating that dataset mixture as deployment traffic:

```text
w = 0.20                              # assumed OOS prevalence
c = (1-w)*accepted_IS/n_IS + w*accepted_OOS/n_OOS
e = (1-w)*wrong_accepted_IS/n_IS + w*accepted_OOS/n_OOS
selective_error = e/c                 # undefined if c=0
cost = 10*e + 1*(1-c)
```

Every accepted OOS request counts as wrong. A rejected request counts as human review. The constants are illustrative requirements, not measured prices. Unknown real-world prevalence can change the result. Review-all has coverage zero and illustrative cost one.

## Why zero observed mistakes is not zero risk

For `k` observed events among `n` requests, a one-sided binomial upper bound uses a beta-distribution quantile:

```text
upper(k,n,alpha) = BetaInverseCDF(1-alpha; k+1, n-k)
upper(0,n,alpha) = 1 - alpha^(1/n)
```

Our code uses `alpha=0.05/3` for each of three bounds. The union bound allocates an overall 5% error budget under the binomial-model assumptions. Combine an upper bound on error among accepted known requests, an upper bound on OOS acceptance, and a lower bound on known acceptance. A conservative mixed selective-error bound is:

```text
coverage_lower = (1-w)*known_acceptance_lower
mixed_error_upper <= known_conditional_error_upper
                     + w*OOS_acceptance_upper/coverage_lower
```

With insufficient accepted examples or zero coverage, fail the gate instead of dividing by zero or declaring perfect performance. The class-stratified benchmark and real distribution shift limit interpretation; this is not a distribution-free guarantee. The [SciPy beta distribution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html) describes the numerical quantile API used here.

## Exercises to do yourself

1. Explain why `softmax(z/0.5)` can increase confidence without changing the predicted class.
2. Calculate the proposed weighted coverage from 2,181/2,396 known accepts and 9/80 OOS accepts. Recover about 75.07% for the assumed 20% OOS mixture.
3. Calculate the proposed OOS false-acceptance rate: 9/80 = 11.25%. Explain why that fails a 5% target even when known-class accuracy is high.
4. Explain why a hypothetical low cost cannot override a failed rejection requirement.
5. Run `route` on a supported question, an unrelated question and a request longer than 128 tokens. All require review with the current failed gate; no external action occurs.
6. Explain why repeatedly changing thresholds using the 80 checked OOS examples would invalidate their role as an independent check.

The next model experiment needs audited negative training data and a genuinely fresh reviewed validation set. Keep the existing failed result, thresholds and final-test boundary intact.
