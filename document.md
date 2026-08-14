# Lyntara - NL2SQL Agent

## 1. Overview
Lyntara is an AI-powered Natural Language to SQL (NL2SQL) system designed to act as a testing and reporting ground for a proposed AI agent. It takes user queries in natural language, classifies their intent/complexity, retrieves relevant database schemas using vector search, and generates executable SQL statements. The architecture consists of a Python-based ML engine, a FastAPI backend, and a modern web frontend.

---

## 2. Module-Wise Explanation

### `Engine/` (Core AI & Generation Logic)
This module handles the natural language processing, classification, and SQL generation.
- **`classifier.py` / `classifierv1.py`**: Executes the machine learning pipeline. It uses pre-trained `.pkl` models (`srd_mrd_classifier`) to categorize incoming queries (e.g., determining if a query requires Single-Relational Data or Multi-Relational Data).
- **`sql.py`**: Interacts with Large Language Models (like Gemini and Groq) to generate accurate SQL based on the classified query and retrieved database schemas.
- **`main.py`**: A CLI script for testing the NLP and SQL generation pipeline directly in the terminal without spinning up the web server.

### `server/` (FastAPI Backend)
The RESTful API layer that connects the frontend with the `Engine` and external databases.
- **`main.py`**: Exposes the API endpoints (`/api/connect`, `/api/query`, `/api/execute`, `/api/schema`, etc.) and handles Server-Sent Events (SSE) for real-time pipeline status updates.
- **`config.py`**: Manages application configurations, environment variables, and path definitions.
- **`schemas.py`**: Contains Pydantic models for strict request and response validation.
- **`services/`**: Sub-module containing the business logic for database connectivity (`db_service.py`), engine interaction (`engine_service.py`), and Qdrant indexing (`pipeline_service.py`).

### `frontend-app/` (Modern Web Interface)
A modern, component-based frontend application built with Vite. 
- Contains the UI for users to connect to their databases, initialize the Qdrant schema pipeline, and input natural language queries to get SQL results.

### `frontend/` (Legacy UI)
- Contains a standalone HTML file (`querynl2sql (1).html`). It appears to be an older or alternative monolithic user interface that relies on vanilla HTML/CSS/JS.

---

## 3. How to Run and Use It

### Backend (Server)
1. Open a terminal in the root of the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: 
   - Mac/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r server/requirements.txt`
5. Run the FastAPI server: `uvicorn server.main:app --reload --port 8000`

### Frontend (Vite App)
1. Open a new terminal and navigate to the frontend directory: `cd frontend-app`
2. Install node modules: `npm install`
3. Start the development server: `npm run dev`

---

## 4. Runnability Check & Missing Components

**Is the project properly runnable out-of-the-box?**  
**No.** While the code is structurally sound, several critical components and configurations are missing or require manual setup before the system can run successfully.

### What is Missing / Required to Run:

1. **Missing `.env` File**: 
   The project requires environment variables to function properly. You must manually create a `.env` file at the repository root containing at least:
   - `GEMINI_API_KEY` (Required for LLM SQL generation)
   - `GROQ_API_KEY` (Required for Multi-Relational Data resolution)
   - Database credentials (`PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`)
   - Qdrant configuration (`QDRANT_HOST`, `QDRANT_PORT`)

2. **Missing `View_Selection/context.json`**:
   The `server/config.py` file explicitly expects a directory and file at `repo_root / "View_Selection" / "context.json"` (Line 46). Additionally, the `/api/schema` endpoint attempts to read from this directory. **This directory and file do not exist in the current repository**, which will likely cause `FileNotFound` errors during schema retrieval.

3. **External Services**:
   - **Qdrant Vector Database**: A running instance of Qdrant (defaulting to `localhost:6333`) must be active to support the vector-based schema search.
   - **PostgreSQL Database**: A running instance of PostgreSQL is required to test the database connections and to execute the generated SQL queries against actual data.

4. **Uninstalled Dependencies**:
   - Python dependencies must be installed via `requirements.txt`.
   - Node dependencies must be installed via `npm install` inside `frontend-app`.
