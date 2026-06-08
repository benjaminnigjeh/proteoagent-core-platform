#!/usr/bin/env python3
"""
api.py — ProteoAgent Core Platform
FastAPI service exposing the supervisor + auxiliary agents over HTTP.
"""

import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from auxiliary_agents import classify_error, generate_session_pdf, run_guardrail, run_qc
from supervisor_agent import build_llm, run_supervisor

DB_PATH = os.environ.get("SESSION_DB_PATH", "/data/sessions.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id        TEXT PRIMARY KEY,
                history   TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    yield


app = FastAPI(title="ProteoAgent API", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str
    llm_choice: str = "ollama"
    anthropic_api_key: Optional[str] = None


class RunResponse(BaseModel):
    session_id: str
    decision: dict
    qc: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_history(session_id: str) -> list:
    with get_db() as conn:
        row = conn.execute(
            "SELECT history FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        return []
    items = json.loads(row["history"])
    history = []
    for item in items:
        if item["type"] == "human":
            history.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            history.append(AIMessage(content=item["content"]))
    return history


def _save_history(session_id: str, history: list) -> None:
    raw = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            raw.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            raw.append({"type": "ai", "content": msg.content})
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, history) VALUES (?, ?)",
            (session_id, json.dumps(raw)),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/run", response_model=RunResponse)
def run_agent(req: RunRequest):
    guardrail = run_guardrail(req.user_input)
    if not guardrail["allowed"]:
        raise HTTPException(status_code=400, detail=guardrail["reason"])

    try:
        llm = build_llm(req.llm_choice, req.anthropic_api_key)
    except Exception as exc:
        err = classify_error(exc)
        raise HTTPException(status_code=400, detail=err["user_message"])

    session_id = req.session_id or str(uuid.uuid4())
    history = _load_history(session_id)

    try:
        decision, updated_history = run_supervisor(guardrail["cleaned_input"], llm, history)
    except Exception as exc:
        err = classify_error(exc)
        raise HTTPException(status_code=500, detail=err["user_message"])

    _save_history(session_id, updated_history)
    qc = run_qc(decision)

    return RunResponse(session_id=session_id, decision=decision, qc=qc)


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, history, created_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": row["id"], "history": json.loads(row["history"]), "created_at": row["created_at"]}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"deleted": session_id}


@app.post("/api/session/{session_id}/report")
def export_pdf(session_id: str):
    history = _load_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty")
    try:
        path = generate_session_pdf(session_id, history, "/data/reports")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"report_path": path}
