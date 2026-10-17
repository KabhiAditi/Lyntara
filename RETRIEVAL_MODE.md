# Branch: `experiment/without-embeddings`

SQL generation places the **entire database schema** in the prompt. No vector
database is involved in answering a question.

    SCHEMA_MODE=full

## Requires

PostgreSQL and Ollama. **Qdrant is not needed to run queries.**

The enriched `View_Selection/{db}_context.json` is still required — it is what
the prompt is built from — but the indexing run that produces it now skips the
embedding/upsert stage entirely, and `check_pipeline_done()` no longer demands
a Qdrant collection.

## Why this mode

Nothing can be missed: every table and column the model might need is in front
of it, so retrieval recall is not a failure mode. The cost is that prompt size
grows with the schema. Chinook's 11 tables serialise to roughly 580 tokens,
which is comfortable — but a 200-table warehouse would not fit, and this mode
does not scale to one.

## Measured on Chinook (11 tables, 20 questions)

| model | execution accuracy | mean latency |
|---|---|---|
| 1.7B step-200  | 60% | 5.6s |
| 1.7B step-1150 | 60% | 5.2s |
| 9B ckpt200     | 70% | 45.7s |

Full schema was **5 points worse than retrieval for the 1.7B** and **identical
for the 9B**. The likely reason is distraction: the smaller model has to
discount six irrelevant tables that retrieval would have filtered out.

See `evaluation/results/` and the master report.
