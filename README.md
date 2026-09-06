# RBAC Security Layer Implementation Walkthrough

The PostgreSQL Role-Based Access Control (RBAC) security layer has been implemented according to requirements.

## 1. Summary of Modified & Created Files

- **`server/services/rbac_service.py`** [NEW]: Core RBAC service containing AST parsing (`parse_sql_permissions`), PostgreSQL privilege verification (`validate_sql_security`), and audit logging.
- **`server/services/db_service.py`** [MODIFY]: Integrated `validate_sql_security()` inside `execute_sql()` **before** opening cursor or executing `cur.execute(sql)`.
- **`server/schemas.py`** [MODIFY]: Extended `ExecuteResponse` with `permission_denied` and `denied_details` fields.
- **`server/main.py`** [MODIFY]: Updated `POST /api/execute` endpoint handler to return structured permission denied details.
- **`server/tests/test_rbac_service.py`** [NEW]: Comprehensive test suite covering 16 security check scenarios.

---

## 2. Architectural Changes & Data Flow

### Sequence of Operations
1. **Request Reception**: Client calls `POST /api/execute` with PostgreSQL credentials and SQL query.
2. **Database Connection**: `execute_sql()` connects to PostgreSQL using the user's authentic credentials (`_connect_postgres`).
3. **Pre-Execution RBAC Validation (`validate_sql_security`)**:
   - **AST Parsing (`sqlglot`)**: The SQL string is parsed to extract the operation (`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`, `TRUNCATE`), target tables, CTEs, subqueries, and read tables.
   - **Database Privilege Check**: Checks `has_database_privilege(username, dbname, 'CONNECT')`.
   - **Schema Privilege Check**: Checks `has_schema_privilege(username, schema, 'USAGE')` (and `'CREATE'` for schema DDL).
   - **Table Privilege Check**: Checks `has_table_privilege(username, f'"{schema}"."{table}"', privilege)` (or table ownership check via `to_regclass` and `pg_has_role` for `ALTER`/`DROP`).
4. **Execution Decision**:
   - **If Denied**: `cur.execute(sql)` is **NEVER** called. Audit log logs `RBAC DENY`, and `execute_sql()` returns `(columns=[], rows=[], error=message, permission_denied=True, denied_details={...})`.
   - **If Allowed**: `cur.execute(sql)` runs normally, transaction commits, and results are returned.

---

## 3. Test Verification Results

All 16 unit tests passed successfully:

```text
Ran 16 tests in 0.017s

OK
```

### Test Coverage Breakdown
- `SELECT` allowed & denied checks
- `INSERT` denied checks
- `UPDATE` denied checks
- `DELETE` denied checks
- `CREATE` denied checks
- `ALTER` denied checks
- `DROP` denied checks
- `TRUNCATE` denied checks
- Multi-table JOINs & CTE handling
- Quoted identifiers & schema-qualified table names
- Fail-closed behavior on SQL parsing failure

---

## 4. Confirmation of Pre-Execution RBAC Checking

The RBAC validation check is invoked inside `execute_sql()` in `server/services/db_service.py` **prior** to initializing `cur.execute(sql)`:

```python
# 1. RBAC validation BEFORE SQL execution
rbac_result = validate_sql_security(conn, user, database, sql)

# 2. Block execution if authorization check fails
if not rbac_result.allowed:
    ...
    return [], [], rbac_result.message, True, denied_details

# 3. Execute SQL query ONLY if RBAC check passes
with conn.cursor() as cur:
    cur.execute(sql)
```

No unauthorized query reaches PostgreSQL query execution.








HOW TO RUN

Viewed README.md:1-4
Viewed README.md:1-30
Viewed config.py:1-69
Viewed package.json:1-15

Here are the detailed steps to run the complete NL2SQL project with the PostgreSQL RBAC security layer.

---

### Prerequisites

1. **Python 3.10+** installed.
2. **PostgreSQL** running locally or remotely (e.g. port `5432`).
3. **Qdrant** vector database running (e.g. port `6333` via Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
4. **Node.js** (v18+) for running the Vite frontend interface.

---

### Step 1: Environment Configuration

Create a `.env` file in the root of the repository (`NLSQL-research--main/.env`):

```env
# Gemini API Key (Required for LLM SQL generation & context enrichment)
GEMINI_API_KEY=your_gemini_api_key_here

# Enable Engine pipeline features
ENABLE_CLASSIFIER=1
ENABLE_SQL_GENERATION=1

# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=table

# Optional: PostgreSQL defaults
PGHOST=localhost
PGPORT=5432
PGDATABASE=your_db_name
PGUSER=your_postgres_user
PGPASSWORD=your_postgres_password
```

---

### Step 2: Install Python Dependencies

From the repository root, set up your Python virtual environment and install the required dependencies:

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r server/requirements.txt
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

---

### Step 3: Run the RBAC Unit Tests

To verify that the RBAC security layer, AST parser, and PostgreSQL permission pre-checks are working properly:

```bash
python -m unittest server/tests/test_rbac_service.py
```

---

### Step 4: Start the FastAPI Backend Server

Start the FastAPI backend server using `uvicorn`:

```bash
uvicorn server.main:app --reload --port 8000
```

The API will start at `http://localhost:8000`. You can inspect interactive API documentation at `http://localhost:8000/docs`.

---

### Step 5: Start the Frontend UI

In a new terminal window, navigate to the `frontend-app` directory and start the Vite development server:

```bash
cd frontend-app
npm install
npm run dev
```

The UI will be accessible in your web browser (typically at `http://localhost:5173`).

---

### How the Workflow Operates

1. **Database Indexing Pipeline**:
   - In the Web UI (or via `POST /api/pipeline/run`), connect your database credentials.
   - The server extracts table schemas, enriches them with Gemini metadata, and embeds them into Qdrant.
2. **NL2SQL Query Generation**:
   - Enter a natural language query in the chat input.
   - The engine searches Qdrant for relevant tables and uses Gemini to generate PostgreSQL SQL.
3. **Execution with RBAC Enforcement (`POST /api/execute`)**:
   - Clicking **Execute** sends the SQL query along with your PostgreSQL user credentials.
   - **`rbac_service` Intercepts the Query BEFORE Execution**:
     - Parses statement operations (`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`, `TRUNCATE`).
     - Queries PostgreSQL system functions (`has_database_privilege`, `has_schema_privilege`, `has_table_privilege`).
     - **If unauthorized**: Blocks `cur.execute()` and returns a structured `permission_denied` response detailing missing privileges.
     - **If authorized**: Executes the SQL query and returns the results.
