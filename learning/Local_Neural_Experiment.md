# Your next lesson: a larger neural network on this PC

The next executable milestone is a frozen MPNet encoder plus supervised intent heads. Everything runs locally, with CPU execution as the reproducible default. The MX450 is not used by this experiment. More waiting does not remove a GPU memory limit, so we cache small batches of features and train the heads separately.

## What is trained here?

The official [all-mpnet-base-v2 model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) describes a 768-dimensional sentence encoder, with Apache-2.0 licensing. We pin revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Its roughly 109 million pretrained parameters are frozen. We did not pretrain that transformer or reproduce its billion-pair training run.

The background paper is Song et al., *MPNet: Masked and Permuted Pre-training for Language Understanding*, NeurIPS 2020 ([published paper](https://proceedings.nips.cc/paper_files/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-Abstract.html)). The sentence embedding artifact includes additional training beyond that paper. Using its features is transfer learning, not reproducing the paper.

We train these fixed comparisons, using the same audited CLINC fit/tune rows:

| Experiment | Representation | Trained classifier |
| --- | --- | --- |
| Existing B1 | Word/character TF-IDF | Logistic regression |
| B2 | Frozen MPNet, 768 features | Logistic regression, C=4 |
| N1 | Same MPNet features | 768 → 128 → 150; 117,782 parameters |
| N2 | Same MPNet features | 768 → 1024 → 512 → 150; 1,389,206 parameters |

N2 has about 11.8 times as many trainable parameters as N1. Width and depth both change, so this is a small architecture comparison, not an isolated study of width. One seed and one fixed optimizer configuration do not establish statistical superiority. No automatic model promotion follows from a high tuning score.

## Follow one request through the math

1. Subword tokenization maps text into token IDs and an attention mask.
2. The frozen transformer computes contextual token vectors `H`.
3. Masked mean pooling ignores padding, then L2 normalization gives a sentence vector `x`:

```text
u = sum_t(mask_t * H_t) / sum_t(mask_t)
x = u / ||u||_2                         shape: 768
h1 = dropout(ReLU(W1 x + b1))           shape: 1024
h2 = dropout(ReLU(W2 h1 + b2))          shape: 512
z  = W3 h2 + b3                        shape: 150
ReLU(a) = max(0, a)
p_k = exp(z_k - max(z)) / sum_j exp(z_j - max(z))
L = -(1/B) sum_i log(p_i,y_i)
```

PyTorch `CrossEntropyLoss` takes logits `z` directly, combining stable log-softmax and negative log likelihood. Do not apply softmax first when computing that loss. Probabilities are used later for reports and predictions.

Backpropagation applies the chain rule through the head. For softmax cross-entropy, the logit derivative is `p - one_hot(y)`, divided by the minibatch size. ReLU passes gradients for positive inputs and blocks them for negative inputs. Dropout randomly masks activations during training and is disabled during evaluation.

AdamW maintains moving averages of gradients and squared gradients, then applies decoupled weight decay:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
theta_t = (1-lr*weight_decay)*theta_(t-1)
          - lr*m_hat_t/(sqrt(v_hat_t)+epsilon)
```

We use PyTorch AdamW defaults for betas/epsilon, learning rate 0.001, weight decay 0.01, dropout 0.2, batches of 128 and gradient norm clipping at 5. The per-epoch log records the last minibatch's gradient norm before clipping, not an average across batches.

## Training and the graphs

`embed` reads only fit/tune requests. It encodes CPU batches of eight, uses a 128-token cap, caches float32 arrays on disk, and checkpoints progress every 256 requests. The manifest records any truncated development requests. The model card's inference default is 384 tokens; our cap is an explicit local resource choice.

`train-neural` runs at most 40 epochs per head, restoring the checkpoint with the lowest tune cross-entropy. Six successive epochs without a lower loss stop training early. The curve's fit loss includes dropout while its tune loss does not, which can affect their relative heights. Check training loss falling while tune loss rises: that is a warning that the head is fitting development-specific details.

The plot compares macro-F1 on the same tune partition. Also read log loss: a model can classify correctly while assigning less useful probabilities. Scores remain uncalibrated. Calibration and OOS rejection require their separate, reserved data.

The reporting function casts all probability arrays to float64 before computing losses, so scikit-learn uses the same clipping precision for neural and linear outputs. Training itself uses float32 logits and PyTorch cross-entropy. Small numerical differences between training loss and reported log loss can remain; accuracy is independent of this clipping convention.

## The gigabyte dataset has a separate purpose

The official [MS MARCO dataset registry](https://github.com/microsoft/msmarco/blob/21b4ece9f4fc10fa4cec2028bb74287629945df6/Datasets.md) lists 8,841,823 passages for v1 passage retrieval. The archive is 1,035,009,698 bytes; the uncompressed collection size is measured after extraction. It contains passage IDs and text, not the 150 intent labels. Repeating CLINC examples or assigning arbitrary labels would not create additional supervised evidence.

Use it to learn streaming data engineering, disk-backed indexing, lexical retrieval, relevance judgments, Recall@k/MRR, and later dense retrieval. Its [terms](https://microsoft.github.io/msmarco/) permit non-commercial research and do not extend the rights to underlying content. Keep corpus files out of the public repository. This is a retrieval research benchmark, not a licensed banking-support knowledge base.

The MPNet model card lists MS MARCO pairs in its upstream training mixture. Disclose that exposure in any later retrieval results; a held-out query file does not establish that the encoder never saw related benchmark content.

## What you should do yourself next

1. Open `src/intentdesk/neural.py` and explain the dimensions of every linear layer. Calculate `a*b+b` for its weights and biases and recover the reported parameter count.
2. Run `predict-neural` on supported requests and unsupported requests. Explain why a high raw score is not a guarantee that routing is safe.
3. Inspect ten mistakes from each head. Write your own explanations in a separate error-review document; do not silently change labels to make the model right.
4. Read the loss curves and explain why the selected epoch can precede the last epoch.
5. Rework the existing numerical-gradient exercise yourself. Then identify the actual `backward()` and `optimizer.step()` calls in this experiment.

The subsequent calibrated-abstention milestone is now implemented: see [Lesson 3](Calibration_and_Review.md). Its independently checked OOS rejection failed, so every request stays in review. The next model experiment needs audited negative training data and fresh independent validation. In the separate retrieval track, acquire official queries/relevance judgments and build a disk-backed BM25 index before adding dense reranking. A review-first UI can use the current suggestions; no API key or hosted model is needed for the current stages.
