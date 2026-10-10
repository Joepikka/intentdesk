# First lesson: teach a computer to recognize a request

Start with the input: a sentence such as “I lost my credit card.” The target is a category called an **intent**, in this case `report_lost_card`. The category is not a generated answer. Password reset, an earlier illustrative example, is not in the actual CLINC150 intent list; it is an example of a request this benchmark classifier does not support.

## Follow the data through the code

1. `src/intentdesk/data.py` validates the dataset and assigns partitions. Fit rows are examples the model can learn from; tune rows help us check development performance. Calibration, validation and final-test data have separate later purposes.
2. `src/intentdesk/train.py` builds word and character n-grams. TF-IDF converts text into numbers, giving a sparse vector for each request. The vocabulary and IDF must be learned from fit examples only.
3. Logistic regression learns a weight for each feature/class combination. Despite its name, it is a classifier here. It produces scores for all 150 intents and converts them to probabilities.
4. The majority baseline ignores the request and predicts the most frequent fit label. It checks whether the learned model adds useful signal.
5. `src/intentdesk/predict.py` loads the saved model and produces suggestions. It never refits the vocabulary on your request and does not log the text to disk.

## Read the results honestly

Accuracy asks what fraction of labels were correct. Macro-F1 calculates a precision/recall balance per class and averages classes equally. A raw confidence score is not a guarantee of correctness or support for an unknown request.

Open the saved `summary.md` for the latest completed run. Check the **split**, number of examples, metrics and limitations before quoting a number. Results on tune data are development results. At this first milestone the final test was unused; it has since been consumed under the frozen protocol in [Milestone 5](../reports/Milestone_05.md).

Look at `development_errors.json`, choose five mistakes, and write your own explanation: ambiguous wording, missing signal, confusing categories, or a possible label problem. The generated file explicitly marks errors as not yet manually reviewed. Do not change labels just to make the score better.

## Try these exercises

1. Run a prediction for your own lost-card request and paraphrase it three ways. Does the top category change? Explain why word and character features might respond differently. Then try a password-reset request and explain why its top known-label suggestion cannot establish support.
2. Find the line that fits the pipeline. Explain why it uses `x_fit`, rather than all requests combined.
3. Run `uv run python learning/math_exercise.py`. Explain the input/weight shapes, why probabilities sum to one, and why subtracting the gradient can lower loss.
4. Change the toy exercise's learning rate and observe what happens. Keep that separate from baseline hyperparameter selection.
5. Explain why the classifier can return a category for a request it was never designed to support.

## Questions to answer without an assistant

- What is a label, a feature and a model weight?
- Why do we need a baseline and held-out examples?
- Which parts were trained here, and which neural features have not been built yet?
- What would make a reported accuracy misleading?

Record your answers and any mistakes you corrected. This is evidence of learning, not a memorization test or a claim you already know every ML technique.

