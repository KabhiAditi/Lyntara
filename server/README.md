## NL2SQL FastAPI server (search-only)

### Install

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### Run

```bash
uvicorn server.main:app --reload --port 8000
```

### Endpoints

- `GET /api/health`
- `POST /api/connect` (added next)
- `POST /api/query` (added next)

### Notes

- This phase is **read-only**: the API will generate SQL but will not execute it.
- You’ll need `GEMINI_API_KEY` and (for MRD resolution) `GROQ_API_KEY` in your environment for full functionality.

