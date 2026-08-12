# System Architecture & Diagrams

Below are the conceptual breakdowns and Mermaid.js visualizations for your NL2SQL pipeline, assuming a fully integrated production environment where schemas are actively indexed and queries are generated and executed.

---

## 1. Data Flow Diagrams (DFD)

### Level 0 DFD (Context Diagram)
The Level 0 DFD shows the system as a single high-level process interacting with external entities.

*   **External Entities:** User (Frontend), External Database (PostgreSQL/Target DB), LLM APIs (Gemini, Groq).
*   **System:** NL2SQL System.
*   **Flow:** 
    * User inputs Database Credentials -> System.
    * System retrieves Schema -> Target DB.
    * User inputs Natural Language Query -> System.
    * System sends/receives context and prompts -> LLM APIs.
    * System executes SQL -> Target DB.
    * System returns SQL and Data Results -> User.

```mermaid
graph TD
    User([User]) -- "DB Credentials & NL Query" --> Sys((NL2SQL System))
    Sys -- "SQL Results & UI State" --> User
    Sys -- "Connect & Fetch Schema" --> DB[(Target Database)]
    DB -- "Schema Metadata & Query Data" --> Sys
    Sys -- "Prompts (Resolution/Generation)" --> APIs[LLM APIs: Gemini/Groq]
    APIs -- "Resolved Query & Generated SQL" --> Sys
```

### Level 1 DFD (Logical Sub-Systems)
Breaks down the main system into primary functional modules (Data Ingestion, ML Pipeline, Execution).

*   **Processes:** 
    1. Schema Collector & Indexer
    2. Query Classifier & Context Resolver
    3. Schema Retriever & SQL Generator
    4. Query Executor
*   **Data Stores:** Conversation History (Memory), Qdrant Vector DB (Schema Embeddings).

```mermaid
graph TD
    %% Entities
    User([User UI])
    DB[(Target DB)]
    APIs[LLM APIs]
    Groq[Groq API]
    
    %% Data Stores
    Qdrant[(Qdrant Vector DB)]
    History[(Conversation History)]

    %% Processes
    P1((1. Schema Indexer))
    P2((2. Query Classifier \n& Resolver))
    P3((3. Schema Retriever \n& SQL Generator))
    P4((4. Query Executor))

    %% Flows
    DB -- Schema Data --> P1
    P1 -- Vector Embeddings --> Qdrant
    
    User -- NL Query --> P2
    P2 <--> History
    P2 -- Context Resolution --> Groq
    Groq -- Resolved Query --> P2
    P2 -- Resolved Query --> P3
    
    Qdrant -- Top-K Context --> P3
    P3 -- Prompt + Context --> APIs
    APIs -- Generated SQL --> P3
    
    P3 -- Validated SQL --> P4
    P4 -- Execute SQL --> DB
    DB -- Result Rows --> P4
    
    P4 -- Display Data --> User
    P3 -- Display SQL --> User
```

### Level 2 DFD (ML Pipeline Breakdown - Process 2 & 3 Expansion)
Shows the exact mechanics inside the inference layers.

```mermaid
graph TD
    Input((Resolved Query)) --> S1(SentenceTransformer Encoder)
    S1 -- Vector --> S2(Qdrant Similarity Search)
    Q[(Context JSON/Qdrant)] --> S2
    S2 -- "Top-K Table Schemas" --> S3(Prompt Builder)
    S3 -- "Structured Prompt" --> Gemini[Gemini API]
    Gemini -- "Raw Output" --> S4(Regex Extractor & Validator)
    
    S4 -- Invalid --> Err(Error Handler)
    S4 -- Valid --> DBExec(Database execution thread)
```

---

## 2. UML Component Diagram

This models the static architecture of your production deployments, outlining how the fastAPI backend orchestrates the inner ML components and external persistence stores.

```mermaid
componentDiagram
    package "Client Layer" {
        [Frontend App (HTML/JS)]
    }

    package "API Layer (FastAPI)" {
        [Main Server]
        [db_service]
        [engine_service]
    }

    package "Engine (ML Core)" {
        [Classifier (scikit-learn)]
        [Context Resolver (Groq)]
        [Schema Retriever (HuggingFace)]
        [NL2SQL Generator (Gemini)]
    }

    package "Infrastructure & APIs" {
        database "Qdrant Vector DB" {
            [Schema Sub-Collections]
        }
        database "Target Database" {
            [PostgreSQL]
        }
        cloud "External LLMs" {
            [Groq Server]
            [Google GenAI]
        }
    }

    [Frontend App (HTML/JS)] ..> [Main Server] : HTTP/REST
    [Main Server] --> [db_service] : Validates & Executes
    [Main Server] --> [engine_service] : Handles inference routes
    
    [engine_service] --> [Classifier (scikit-learn)] : Predict SRD/MRD
    [engine_service] --> [Context Resolver (Groq)] : Generate standalone query
    [engine_service] --> [Schema Retriever (HuggingFace)]
    [engine_service] --> [NL2SQL Generator (Gemini)]
    
    [Context Resolver (Groq)] ..> [Groq Server] : REST
    [NL2SQL Generator (Gemini)] ..> [Google GenAI] : REST
    [Schema Retriever (HuggingFace)] ..> [Schema Sub-Collections] : gRPC/HTTP
    [db_service] ..> [PostgreSQL] : TCP (psycopg)
```

---

## 3. Sequence Flow (Inference Execution Diagram)

The sequence diagram reflects the exact synchronous nature of a user pressing "Run Query" in the UI.

```mermaid
sequenceDiagram
    actor User
    participant UI as Frontend
    participant API as FastAPI Backend
    participant Class as Classifier Model
    participant Res as Context Resolver (Groq)
    participant Qdr as Qdrant DB
    participant Gen as SQL Generator (Gemini)
    participant DB as Target Database

    %% Connection Phase
    User->>UI: Input DB Credentials
    UI->>API: POST /api/connect
    API->>Qdr: Verify Qdrant Healthy
    API->>DB: Verify PostgreSQL Access
    API-->>UI: 200 OK (Connected)
    UI->>API: GET /api/schema
    API-->>UI: Returning indexed schema layout

    %% Inference Phase
    User->>UI: Type NL Query ("Break it down by region")
    UI->>API: POST /api/query
    
    API->>Class: Predict SRD/MRD
    Class-->>API: Returns MRD (0.95 Confidence)
    
    API->>Res: Fetch History + Groq Prompt
    Res-->>API: "Break total revenue down by region"
    
    API->>Qdr: Search embeddings for tables
    Qdr-->>API: Returns 'orders', 'payments' schema
    
    API->>Gen: Generate SQL with context context
    Gen-->>API: SELECT region, SUM...
    
    API-->>UI: Query Response (SQL payload, status)

    %% Execution Phase
    User->>UI: Click "Execute"
    UI->>API: POST /api/execute (SQL + DB Config)
    API->>DB: psycopg.execute(SQL)
    DB-->>API: Row Data
    API-->>UI: Table rows returned
    UI-->>User: Visualizes metrics
```
