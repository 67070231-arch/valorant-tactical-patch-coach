import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import MODEL_NAME, run_agent
from app.rag import get_rag_engine
from scripts.init_db import DB_PATH, init_database

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="VALORANT Tactical & Patch Coach API",
    description="End-to-End LLM Capstone Application with Hybrid RAG, Agentic Text-to-SQL, and Reproducible Evaluation",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Player or coach question about VALORANT patches, tactics, or weapon/ability stats.")


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    tool_trace: List[Dict[str, Any]]
    steps_used: int
    model: str


@app.on_event("startup")
def startup_initialize() -> None:
    if not os.path.exists(DB_PATH):
        init_database()
    get_rag_engine()


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index_path)


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        result = run_agent(payload.message)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
