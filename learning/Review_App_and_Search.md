# From predictions to a usable ML application

Everything in this milestone runs locally. The browser talks to a Python API, which reuses one frozen model instance. SQLite records the request, its model/version/score output, and your review decisions. The separate passage-search index uses the standard-library SQLite FTS5 extension; it needs no vector database subscription.

## Try the workflow

From the project directory in PowerShell:

```powershell
uv sync --locked --extra dev --extra neural --extra web
uv run --locked --extra neural --extra web intentdesk serve
```

Open http://127.0.0.1:8765. Submit a fictional request, inspect the suggestions and save a decision. Click a suggestion to fill its intent label. You may also choose any supported intent, `oos` (unsupported) or `uncertain` (insufficient context).

The app listens on loopback, uses one model process and performs no banking/ticket actions. Keep the terminal running; Ctrl+C stops it. Restarting preserves `local/reviews.sqlite3`. That folder is excluded from version control. To preserve your work, copy the entire `local` folder after stopping the server; the database uses a WAL companion while open. Do not publish personal requests.

## Why reviews have versions

Suppose two tabs open request version 0. The first decision updates the version to 1 and appends its history entry in one database transaction. The second tab's update requires version 0, so it fails with HTTP 409 instead of replacing the first review. Refresh before deciding again. SQL values are bound parameters, so request text never becomes executable SQL.

Feedback is a candidate annotation, not automatically correct ground truth. Before training on it, check consent, labeling guidelines, adjudication, duplicates, provenance and split allocation. Our browser smoke examples are explicitly assistant-authored demo reviews, not independent human validation. The consumed policy check and final benchmark must never enter new training by accident.

## Search mathematics

For a query term t and passage d, BM25 combines rarity, term frequency and length normalization:

\[
\operatorname{BM25}(d,q)=\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
\]

SQLite FTS5 defaults use k1=1.2 and b=0.75, with a floor on IDF and a negative internal score so smaller values rank first. Our displayed score negates that value. It is a ranking score, not a probability. The tokenizer applies Unicode word boundaries and Porter stemming. Query construction removes a fixed English stop-word list, keeps at most 12 unique terms and combines literal terms with OR. A phrase like `machine learning` can match either word, with ranking favoring stronger matches. Search is lexical: it can miss synonyms and match irrelevant shared terms.

Read [the SQLite BM25 definition](https://www.sqlite.org/fts5.html#the_bm25_function) and [the original probabilistic retrieval review](https://www.nowpublishers.com/article/Details/INR-019). Our implementation uses the SQLite variant, not a claimed reproduction of another search engine.

## Why the index fits this PC

The corpus remains on disk. The builder inserts 5,000 passages per transaction with a 64 MiB SQLite page cache and disk-backed temporary work. Each committed transaction records the next byte offset and passage ID; a restart resumes only the same corpus/configuration. Corpus and finished index checksums, row counts and database integrity checks are recorded. A partial index is never exposed as finished. The full index needs additional disk space beyond the 3.06 GB text corpus. A cache bound does not bound the entire Python/OS working set.

During search, only the top passages are returned. A two-second interactive query budget stops costly searches and asks for more specific terms. Benchmark queries have a predeclared ten-second budget and timeouts count as zero retrieval success. Model inference and search each accept one simultaneous operation, returning a retry message for competing work. This is a small local demo, not a distributed serving system.

## Measure search honestly

The fixed retrieval experiment samples 100 official development query IDs by SHA-256 ordering, before ranking them against all 8,841,823 passages. It reports:

\[
\operatorname{MRR@10}=\frac{1}{Q}\sum_q\frac{1}{\operatorname{rank}_q},\qquad
\operatorname{Recall@10}=\frac{1}{Q}\sum_q\frac{|R_q\cap\operatorname{top10}_q|}{|R_q|}.
\]

Reciprocal rank is zero when no judged relevant passage appears in the first ten results. Relevance judgments are incomplete: an unjudged retrieved passage is not automatically irrelevant. The 100-query result is exploratory development evidence, not a leaderboard score or end-to-end answer-quality score. This stage uses no neural reranker and generates no answers. MPNet's upstream MS MARCO exposure remains relevant to future dense-retrieval comparisons, but this lexical index does not use MPNet embeddings.

## Explain it yourself

1. Why can a classifier be very accurate on known intents while failing on unsupported requests?
2. Trace one submission through HTML/JavaScript, FastAPI validation, the shared Router, SQLite and the review screen.
3. Show why a review conflict must roll back the decision-history insert too.
4. Work out MRR@10 for first-relevant ranks 1, 2, absent and 5. Answer: 0.425.
5. Explain why increasing dataset GB cannot add missing intent labels.
6. Find a synonym query that lexical search handles poorly, and describe how you would design a fair dense-retrieval comparison without tuning the final classifier benchmark.

The portfolio is strongest when you can modify and explain this code yourself. Commit your own exercises and written reasoning, and disclose AI assistance in your project presentation.
