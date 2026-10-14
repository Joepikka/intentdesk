# Present IntentDesk accurately

IntentDesk demonstrates an applied NLP/ML workflow plus a local inference application. It is suitable as a student portfolio project when you can explain and modify it. It does not make you experienced in every ML concept, establish production reliability or guarantee an AI job. Your June 2027 graduation and employers' degree, internship, location and work-authorization requirements still matter.

## Evidence an employer can inspect

| Skill | Concrete project evidence |
| --- | --- |
| Python and software fundamentals | Typed validation models, CLI commands, modules, exceptions, tests and resource lifecycles |
| Data work | Pinned datasets, hashes, duplicate/leakage audit, manifests, protected split allocation, separate feedback records |
| NLP and feature engineering | Word/character TF-IDF, n-grams, tokenization, pretrained MPNet mean pooling and normalized embeddings |
| Classical ML | Logistic regression and a majority baseline, meaningful comparison with neural heads |
| Deep learning | PyTorch heads, ReLU, dropout, cross-entropy, AdamW, learning curves, early stopping and checkpoint reload parity |
| Statistical evaluation | Accuracy, macro-F1, log loss, Brier, reliability bins, temperature scaling, grouped bootstrap and independently checked abstention |
| Decision-making | Explicit review cost assumptions, threshold selection separated from checking, failed gate and review fallback |
| Unsupported-request detection | Six frozen-embedding/confidence features, balanced binary logistic regression, disjoint neighbor references, pre-check artifact binding, measured false positives and exact uncertainty bound |
| Retrieval | A complete 8.84M-passage SQLite FTS5/BM25 index, official development queries/qrels, MRR@10/recall@10 and timeouts |
| ML application engineering | FastAPI inference, reusable model, SQLite review history, optimistic concurrency, local browser UI and versioned artifacts |
| Workflow evaluation | A six-category card-support pilot, local category-to-team handoffs, blinded manual trials, assisted trials, fixed timing exclusions and reviewer-level exploratory comparisons |
| Robustness/error analysis | Sixty versioned synthetic cases, negation/typo/ambiguity slices, confusion matrix, per-class metrics, separate reviewer-declared labels and recorded failures |
| Iterative ML improvement | Preserved lexical-candidate failure, separate hybrid features/classifier, declared synthetic augmentation, fit-only training and versioned consumed checks |
| Classical model comparison | Logistic regression, linear SVM, decision tree, random forest and KNN on fixed product-review partitions, with disclosed representation differences |
| Regression and unsupervised learning | Numeric star-rating regression with MAE/RMSE/R²; training-fitted SVD and MiniBatchKMeans with geometry diagnostics rather than invented topic accuracy |
| Large-data engineering | Two pinned review files totaling 1.98 GB, disk-backed duplicate/label audit, product-group splits, bounded batches and full-fit streaming learning |
| Transformer implementation | Random embeddings, custom scaled dot-product multi-head attention, masking, residuals, layer normalization, GELU and supervised scratch training with checkpoint selection |
| Responsible claims | Frozen encoder disclosure, source/terms preservation, unsupported-request limitation, no production/real-customer savings claim |

The project does **not** currently prove full-transformer fine-tuning, distributed training, generative-model alignment, computer vision, reinforcement learning, production MLOps or customer-facing RAG quality. Add future work only when you have implemented, evaluated and understood it.

The [feedback lab](FEEDBACK_ML_LAB.md) adds supervised training of compact transformers from random initialization. This differs from fine-tuning MPNet or pretraining a foundation language model. Use its actual run reports, fit/development/final allocations and parameter counts when discussing it. The larger streaming allocation and differing token/feature budgets prevent an isolated algorithm-only interpretation. Review-rating results do not establish card-intent accuracy or customer savings.

## Resume wording you can support

Use these only after you have personally run the project, understood the code and completed the exercises. Do not turn AI-generated implementation into a claim that you wrote every line unaided.

The [card-support demonstration](CARD_SUPPORT_PILOT.md) gives the project a concrete workflow and a measurement instrument. You may claim you implemented that instrument after understanding it. Do not claim it improved human speed, reduced customer errors or saved money: there are currently no completed human comparisons and the dataset is synthetic. Add actual study counts and measured limitations only after running and exporting a human comparison.

> Built a local NLP intent-review application using Python, PyTorch, scikit-learn, FastAPI and SQLite; compared lexical and frozen-MPNet classifiers with reproducible, leakage-audited data partitions.

> Evaluated a preselected calibrated classifier at 96.24% accuracy and 0.9624 macro-F1 on 4,500 known-intent test requests; retained human review after unsupported-request evaluation failed.

> Implemented resumable BM25 search across 8.84 million MS MARCO passages with bounded database caching, integrity checks and a fixed development retrieval benchmark.

> Added a versioned unsupported-request advisory with explicit human review; reduced supported suggestions from 13 to 3 on a separate 100-request OOS check while preserving the original final evaluation.

For that last claim, show [Milestone 6](../reports/Milestone_06.md): the known check is reused calibration data, with 78 false OOS flags out of 1,496. The OOS false-acceptance upper bound is 7.57% at one-sided 95% confidence. This is not an independently validated automatic router. The motorcycle example is a development demonstration, not held-out evidence.

After personally running and explaining the feedback lab, an additional supported claim is:

> Implemented a local 1.98 GB review-data pipeline with product-group splits and duplicate/label audits; compared classical classifiers, numeric regression, unlabeled clustering and custom scratch transformers (1,318,021 and 2,966,533 trainable parameters). Trained a streaming classifier on 2.91 million eligible fit reviews and reported all fixed candidates on a 20,000-review product-disjoint test.

[Milestone 10](../reports/Milestone_10.md) supports the counts and comparisons. The preselected streaming_sgd scored 70.37% accuracy and 0.5427 macro-F1 on subjective review ratings. These scores do not measure card-intent quality or customer savings.

You can also demonstrate the [robustness tool](ROBUSTNESS.md): it found 22 disagreements across 60 synthetic development cases, including all 12 clarification cases. Explain why its 63.3% provisional agreement is not independent accuracy and cannot be compared directly with the 96.24% known-intent benchmark. Show the blank human label sheet and the distinction between annotation quality, model error and runtime failure. This is useful evidence of ML evaluation judgment; it does not establish a stronger model.

Add the measured retrieval score from Milestone 4 only with its 100-query development-sample qualification. Do not describe it as a full official leaderboard result. Do not claim you trained the 109M-parameter encoder or used GB of intent labels. Match a job description's terms only when these artifacts and your knowledge support them; keyword stuffing and hidden text do not demonstrate competence.

## A five-minute demo

The [workflow update](CARD_WORKFLOW_UPDATE.md) adds a lexical candidate that generalized poorly, followed by a frozen-embedding/character hybrid. On a reserved 32-case synthetic check, agreement was 29/32 versus original 21/32. Describe the same-author, unreviewed-label limitation and remaining failures. Do not claim 90.6% real-customer accuracy, a trained foundation model, fully solved negation or independent human benefit. Explain why inspecting the V2 check made it development data for V3 and why later tuning needs a new check.

1. Explain the user problem: help review a short English request, with uncertainty and a person making the decision.
2. Submit a supported example, inspect the top labels and save a review. Show history and a corrected decision.
3. Submit an unsupported request such as the motorcycle example. Show Likely unsupported, hidden known-label suggestions and editable `oos`; explain the detector can also misclassify known requests. Use Analyze again on an older record to demonstrate preserved history. Human review remains required.
4. Search the separate research corpus. Explain BM25, source passages and what the ranking score means.
5. Show final/development results, the weaker large head and one data-quality limitation. Name a next experiment requiring fresh evaluation data.

Use fictional inputs. A screen recording and a public source repository can demonstrate the project without paid hosting. Raw data, trained weights, caches and local review records stay out of the repository. No publication or job application is performed by this document.

## Questions to answer without reading a script

- Derive softmax and explain how cross-entropy changes the weights through backpropagation.
- Why does positive temperature scaling preserve the highest-scoring class?
- Why might a linear classifier on frozen embeddings beat a larger neural head?
- Why does the current robustness suite expose ambiguity errors despite a strong known-intent benchmark? What would a fair follow-up evaluation require?
- What information leaks if TF-IDF is fitted before splitting the data?
- Why are calibration data, threshold-selection data and an independent check separate?
- What does 135 accepted unsupported requests out of 1,000 tell you, and why can you not fix the threshold using that final test?
- How do the six scope features differ from known-class probabilities? Why separate a query from its neighbor reference, and why does 3/100 with a 7.57% upper bound not prove a 5% acceptance target?
- How does SQLite commit both a review decision and its version/history consistently?
- What changes for a hosted multi-user application? Discuss authentication, privacy, throughput, observability, data retention and measured drift.

Read the learning documents and write your own answers in a separate learning journal. The goal is demonstrable understanding and a defensible project, not a list claiming every AI topic.

## Completed feedback improvement evidence

[Milestone 11](../reports/Milestone_11.md) adds a bounded model-improvement comparison, class-prior/imbalance analysis, new held-out product groups and a checked release wrapper. The selected word/character classifier improved fresh-check accuracy from 67.955% to 69.13% and macro-F1 from 0.5487 to 0.5591 on the same sample. A higher-accuracy natural-weight candidate had worse macro-F1 and was not substituted after checking. Explain that tradeoff and why the original 70.37% came from a different population. Do not claim 90% accuracy.

After personally understanding and demonstrating it, you can describe versioned hypothesis testing, preservation of negative/overfit outcomes, development-only selection and separate fresh evaluation. This is evidence of applied ML engineering judgment, not proof of every ML concept or real business benefit.
