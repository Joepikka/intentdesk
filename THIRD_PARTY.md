# Dependencies, data and cost

IntentDesk runs on the owner's PC using Python, NumPy, SciPy, scikit-learn, PyTorch CPU, Transformers, FastAPI, Uvicorn and the standard-library SQLite database. No API key, cloud GPU, subscription or paid inference service is required by the application. Normal electricity, internet and disk costs remain. The Codex/ChatGPT account used to develop the project is separate from the application's runtime requirements.

- **CLINC150**: pinned upstream dataset and CC BY 3.0 license are recorded in `data/README.md` and `data/CLINC150_LICENSE.txt`. Preserve attribution. It is crowdsourced English assistant-intent data, not private customer support records. [Upstream repository](https://github.com/clinc/oos-eval).
- **MPNet model**: `sentence-transformers/all-mpnet-base-v2`, pinned revision and checksums in the local model manifest. Model card declares Apache 2.0. Pretrained encoder remains frozen; only downstream heads were trained here. [Model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
- **MS MARCO**: separate research passage collection, queries and relevance judgments. Non-commercial research use only; underlying passage rights are not granted by the dataset distributor. Raw files and indexes are excluded from the portfolio repository. [Official terms](https://microsoft.github.io/msmarco/), [pinned dataset registry](https://github.com/microsoft/msmarco/blob/21b4ece9f4fc10fa4cec2028bb74287629945df6/Datasets.md).
- **FastAPI**: MIT-licensed Python API framework. [Official documentation and license](https://fastapi.tiangolo.com/).
- **SQLite**: local database/full-text search; [public-domain statement](https://www.sqlite.org/copyright.html), [FTS5 documentation](https://www.sqlite.org/fts5.html).

Free access does not remove license obligations. A future commercial customer product should use permissioned support documents or appropriately licensed replacement data. No corpus, weights, user-review database or credentials should be published accidentally.
