# Milestone 4: free local app and full-corpus search

Completed 10 October 2026. The app provides a Python API, browser review workbench, SQLite request/decision history and local BM25 search. No paid API or cloud is required. The classifier's failed development gate remains active: all requests require review, and no external actions occur.

## Retrieval evidence

- Indexed **8,841,823 passages**, from the separately audited **3.062 GB** MS MARCO corpus.
- Finished index: **4.700 GB** on disk; indexing/integrity/hash time **1064.9 seconds** across this run's committed work.
- A 64 MiB SQLite page cache, 5,000-row transactions, resumable byte offsets and disk-based merges keep the corpus out of a giant Python/RAM structure. This is not a claim that the whole process uses only 64 MiB.
- Row count, FTS integrity, SQLite quick-check and finished index SHA-256 passed.
- Fixed sample: **100 official development queries**, chosen by hashed ID order before retrieval, searched against the complete corpus. Query/relevance source revisions and checksums are preserved.
- **MRR@10: 0.2195**; **mean recall@10: 0.4350**; hit rate@10: 0.4400.
- Query-bootstrap MRR interval: [0.1511, 0.2937]. Small development sample and incomplete relevance judgments limit its interpretation.
- Sequential benchmark median **738.4 ms**, p95 **3418.3 ms**; **0** ten-second timeouts counted as zero retrieval success. These include per-query read-only SQLite connection work on this PC, not API/concurrent performance.

This is a lexical retrieval baseline. It does not use the MPNet encoder, neural reranking, a support knowledge base or generated answers. Search results are passages, not verified facts. MS MARCO is non-commercial research data and is not included in the source repository.

## Review application

Run `uv run --locked --extra neural --extra web intentdesk serve`, then open http://127.0.0.1:8765. Submit a fictional request, compare known-label scores, choose an intent/`oos`/`uncertain`, save a decision and inspect its history. SQLite persists these records in the ignored `local/` folder. Feedback never automatically retrains a model. Optimistic versions prevent stale-tab overwrites. Model/search operations return retry messages when busy; long searches have a two-second interactive budget.

The application binds only to loopback and uses a single reusable model process. Request validation, body-size limits, host/origin checks, a required mutation header, CSP and text-only rendering cover this local-demo boundary. It has no multi-user authentication or internet deployment design. Local review records are unencrypted files accessible to the PC owner.

[Search evidence](retrieval_evaluation.json), [index metadata](retrieval_index.json), [learning walkthrough](../learning/Review_App_and_Search.md). Browser screenshots and verification evidence accompany the final handoff. Demo reviews written by the assistant are labeled as such and are not independent human validation.
