# Lyntara LLM provider layer

SQL generation runs on a **local** Qwen3.5-9B fine-tune instead of the Gemini API.
Everything below the call sites is provider-agnostic, so changing model or runtime
is a `.env` edit.

## Roles

| Role | Default | Used by |
|---|---|---|
| `sql` | `ollama` / `lyntara-sql:ckpt200` | `NL2SQL.generate` — the Spider fine-tune |
| `sqlfix` | `ollama` / `qwen3.5:9b` | `fix_sql_with_llm` — repair, stock base model |
| `context` | `gemini` / `gemini-2.5-flash` | per-table JSON enrichment in `pipeline_service` |

`sqlfix` deliberately uses the **base** model: the Spider LoRA never saw
error-correction examples.

## Configuration

Every role reads `LLM_<ROLE>_*` from `.env`:

```
LLM_SQL_PROVIDER=ollama        # ollama | openai | llamacpp | lmstudio | vllm | gemini
LLM_SQL_BASE_URL=http://localhost:11434
LLM_SQL_MODEL=lyntara-sql:ckpt200
LLM_SQL_TEMPERATURE=0.1
LLM_SQL_MAX_TOKENS=512
LLM_SQL_NUM_CTX=8192
LLM_SQL_TIMEOUT=120
LLM_SQL_RETRIES=1
```

### Swapping models

Same runtime, different model — change one line:

```
LLM_SQL_MODEL=qwen3.5:9b        # stock base, to A/B against the fine-tune
```

Different runtime — e.g. llama.cpp's `llama-server` on :8080:

```
LLM_SQL_PROVIDER=llamacpp
LLM_SQL_BASE_URL=http://localhost:8080
LLM_SQL_MODEL=whatever-the-server-serves
```

Back to a hosted API:

```
LLM_SQL_PROVIDER=gemini
LLM_SQL_MODEL=gemini-2.5-flash
```

Adding a backend means one entry in `_BUILDERS` in `factory.py`. Anything speaking
the OpenAI HTTP shape needs no new code at all.

## Files

- `base.py` — the `LLMProvider` protocol and prompt-style constants.
- `openai_compat.py` — `/v1/chat/completions` + `/v1/completions`; covers Ollama,
  llama.cpp, LM Studio, vLLM, OpenAI.
- `ollama.py` — Ollama's native `/api/generate` with `raw: true`.
- `gemini.py` — google-genai behind the same protocol.
- `factory.py` — role → configured provider.
- `prompts.py` — **the training contract** (see below).

## Why `raw: true` matters

`prompts.py` encodes the exact format the adapter was trained on
(`SLM/V3/QwenV3.ipynb`, cells 17 and 19):

```
<|im_start|>system
You are an expert SQL assistant. ...<|im_end|>
<|im_start|>user
Database: {db_id}

Schema:
CREATE TABLE ... ;

Question: {question}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Two details are load-bearing:

1. Schema is rendered as **CREATE TABLE statements**, not the old
   `Table:/Columns:` format. PostgreSQL types are folded into Spider's four
   buckets (TEXT / INTEGER / DATETIME / BOOLEAN).
2. The assistant turn is prefilled with an **empty `<think></think>` block**,
   because every training target had one.

Sending this through a chat endpoint would wrap it in a second ChatML envelope,
so the SQL path uses raw completion and the provider sends the string untouched.

## Known limitation: dialect

The adapter was trained on Spider, which is **SQLite**. Lyntara targets
**PostgreSQL**. Expect occasional dialect errors (quoting, casting, `LIMIT`).
Mitigations already in place: `sqlglot` validation against the schema in
`NL2SQL._validate`, and the `/api/execute` self-repair path. Measure this with
`scripts/smoke_local_llm.py` before trusting it on a new database.

## Performance

A 9B Q4_K_M on an M2 Air (16 GB) runs ~10 tok/s and the model emits reasoning
before the SQL, so budget **20-30s per query** versus ~2s from Gemini. Hence
`LLM_SQL_RETRIES=1` by default and a 120s timeout — the old hard-coded
`retries=2` would double the worst case for a deterministic repeat.
