# Start here when you return

## Try the concrete support workflow

Open **http://127.0.0.1:8765/pilot**, read the category guide, enter `reviewer01`, and complete the 14-ticket manual/assisted comparison yourself. It prepares local team handoffs and records your timing/corrections. Tickets are fictional; customer value is not established. Read [the short pilot guide](CARD_SUPPORT_PILOT.md). This is an additional workflow; the original classifier demo remains at `/`.

IntentDesk is a working local research portfolio: enter a fictional request, inspect intent suggestions and save a review. It also searches a separate large research corpus. It runs without paid APIs or cloud training.

## Run it yourself

Open PowerShell:

```powershell
cd 'D:\ml model forex\intentdesk'
uv sync --locked --extra dev --extra neural --extra web
uv run --locked --extra neural --extra web intentdesk serve
```

Open http://127.0.0.1:8765 and keep the terminal running. If the app is already running, open the URL directly. Ctrl+C in the server terminal stops it. The source lives in this directory; the old forex bot is a separate project.

## Your first session

1. Enter `I lost my credit card`. Inspect the three suggestions, choose a decision, add a note and save. Find it under Reviewed and open Decision history.
2. Enter `Write a sonnet about the moon`, or your motorcycle example. Inspect the scope advisory; it can flag likely unsupported requests and suggest `oos`, while still requiring you to confirm. It can make mistakes. These are learning examples, not new evaluation evidence.
3. Search `photosynthesis chlorophyll` in the separate corpus section. Explain why a matching passage is not a verified answer.
4. Read [Milestone 5](../reports/Milestone_05.md). Explain why 96.24% known-intent accuracy and poor unsupported-request detection can coexist.

## Learn it in this order

Spend as many sessions as needed on each step; speed is not the goal.

| Step | Read and do | Explain without a script |
| --- | --- | --- |
| Python and data | [First lesson](../learning/First_Lesson.md); trace `data.py`, a JSON row and a manifest entry | Functions, dictionaries, exceptions, train/development/test separation and leakage |
| Classical ML and math | Run `uv run --locked python learning/math_exercise.py`; trace `train.py` | TF-IDF, vector dimensions, logistic regression, softmax, cross-entropy and gradients |
| Neural networks | [Neural lesson](../learning/Local_Neural_Experiment.md); inspect saved training curves | Frozen transfer learning, ReLU, backpropagation, dropout, AdamW and why larger did not win |
| Reliable decisions | [Calibration lesson](../learning/Calibration_and_Review.md); compare calibration and policy reports | Macro-F1, loss, calibration, confidence intervals, abstention and consumed evaluation data |
| Application and retrieval | [App/search lesson](../learning/Review_App_and_Search.md); trace a request through API and SQLite | HTTP validation, SQL transactions, stale versions, BM25, MRR and recall |
| Portfolio presentation | [Portfolio guide](PORTFOLIO.md); record your own five-minute demo | What you implemented, what AI helped with, measured failures and a justified next experiment |

Write your own explanations in a learning journal and make one small, tested change you can defend. The existing application is useful practice; your ability to explain, debug and extend it is what demonstrates your skill.

The final classifier evaluation, failed policy check and new scope-advisory check are frozen. Do not train against or tune to their results. Read [the new scope lesson](../learning/Scope_Detection.md) and [Milestone 6](../reports/Milestone_06.md). A future detector version needs fresh, independently reviewed known/OOS joint evaluation examples. Generation, permissioned support documents and a hosted service are later extensions, not completed features.
