# VALORANT Tactical & Patch Coach

End-to-end LLM-powered Capstone Application for **HW8 — Project: Build & Evaluate an LLM-Powered Application (200 pts)**.

## 1. Problem & Target Users
Competitive VALORANT players, team IGLs (In-Game Leaders), and esports analysts frequently struggle to reconcile recent patch balance adjustments (such as Patch 9.08 changes to Cypher, Phoenix, Gekko, Yoru, and the Sunset map) with exact numerical weapon and ability economy statistics. Standard base LLMs hallucinate specific credit costs, damage falloffs, or recent patch windup timings.

**VALORANT Tactical & Patch Coach** solves this by combining:
1. **Hybrid Retrieval / RAG with Inline Citations (`[patch_9_08.md]`, `[agent_guides.md]`)**: Searches curated Patch 9.08 notes and tactical playbooks using 120-word overlapping chunks, BM25 lexical scoring, character/synonym-enriched Vector Cosine Similarity, Reciprocal Rank Fusion (`RRF`, `k=60`), and a `min_score` floor.
2. **ReAct Agent Loop with Tool Calling & Loop Guard**: Calls `model="gemma-4-E4B-it"` at `temperature=0` through the course proxy (`from openai import OpenAI; client = OpenAI()`), dynamically selecting between `search_patches` and `query_valorant_db` for up to `max_steps=5` with duplicate tool-call prevention and Pydantic schema validation.
3. **Safe Agentic Text-to-SQL (`data/valorant.db`)**: Queries a local SQLite database containing 12 weapons and 16 agent abilities using `is_safe_select(sql)` (blocking stacked queries `;`, SQL comments `--`/`/*`, and all non-`SELECT` DDL/DML keywords) over a strict read-only SQLite URI (`?mode=ro`).
4. **Served FastAPI Backend & Interactive Web UI**: Exposes `GET /` and `POST /api/chat` serving `static/index.html`, displaying both grounded answers with citation badges and the step-by-step ReAct tool trace.
5. **Reproducible Evaluation (`eval/test_set.json` & `eval/run_eval.py`)**: Evaluates 30 curated test questions comparing a **Baseline (Vector-Only Top-1)** against our **Improved System (Hybrid BM25 + Vector RRF Top-5)** using `Recall@5`, `MRR`, and SQL execution accuracy.

## 2. Project Structure
```text
valorant-tactical-coach/
├── requirements.txt
├── README.md
├── data/
│   ├── valorant.db
│   └── patches/
│       ├── patch_9_08.md
│       └── agent_guides.md
├── scripts/
│   └── init_db.py
├── app/
│   ├── rag.py
│   ├── sql_tool.py
│   ├── agent.py
│   └── main.py
├── static/
│   └── index.html
└── eval/
    ├── test_set.json
    └── run_eval.py
```

## 3. Setup & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (No Hardcoded Secrets)
```bash
export OPENAI_API_KEY="your-class-proxy-api-key"
export OPENAI_BASE_URL="https://your-class-proxy-url/v1"
```

### Step 3: Initialize the SQLite Database
```bash
python3 scripts/init_db.py
```

## 4. Running the Web Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` in your browser to use the interactive chat interface and inspect real-time ReAct tool traces.

## 5. Running the Reproducible Evaluation
```bash
python3 eval/run_eval.py
```
This script runs all 30 benchmark examples in `eval/test_set.json` and prints the Before vs. After comparison table (`Recall@5` and `MRR`) along with SQL execution and safety verification checks.
