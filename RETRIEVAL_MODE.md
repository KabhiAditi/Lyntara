# Branch: `experiment/with-embeddings`

SQL generation selects schema by **dense retrieval**.

A question is embedded with `Alibaba-NLP/gte-large-en-v1.5` (1024-dim) and
matched against per-table description vectors in Qdrant. The top `SCHEMA_TOP_K`
tables (default 5) are serialised into the prompt.

    SCHEMA_MODE=embeddings
    SCHEMA_TOP_K=5

## Requires

PostgreSQL, Qdrant (with `{db}-table` collections built by the indexing
pipeline), and Ollama.

## Why this mode

Prompt size stays roughly constant as the schema grows, so it is the only
option that scales to databases with hundreds of tables. The cost is a
recall risk: a join is answerable only if *every* table it needs is retrieved.

## Measured on Chinook (11 tables, 20 questions)

| model | execution accuracy | mean latency |
|---|---|---|
| 1.7B step-200  | 65% | 6.5s |
| 1.7B step-1150 | 65% | 7.7s |
| 9B ckpt200     | 70% | 43.0s |

Retrieval was worth **+5 points to the 1.7B** over full-schema and was
**neutral for the 9B**. See `evaluation/results/` and the master report.

## Note on a fixed defect

`SchemaRetriever.get_top_context_object()` previously discarded all but the
single best-matching table, so the model saw one table no matter what
`top_k` was set to, making every join unanswerable. `schema_service` now
keeps all top-k tables.
