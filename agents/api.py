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
from fastapi.responses import FileResponse
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
                id         TEXT PRIMARY KEY,
                history    TEXT NOT NULL DEFAULT '[]',
                messages   TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # migration: add messages column to pre-existing sessions tables
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN messages TEXT NOT NULL DEFAULT '[]'")
        except Exception:
            pass
    yield


app = FastAPI(title="ProteoAgent API", version="1.0.0", lifespan=lifespan)


# ── Models ────────────────────────────────────────────────────────────────────

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
        row = conn.execute("SELECT history FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        return []
    history = []
    for item in json.loads(row["history"]):
        if item["type"] == "human":
            history.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            history.append(AIMessage(content=item["content"]))
    return history


def _save_turn(session_id: str, history: list, user_text: str, decision: dict, qc: dict) -> None:
    raw_history = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            raw_history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            raw_history.append({"type": "ai", "content": msg.content})

    with get_db() as conn:
        row = conn.execute("SELECT messages FROM sessions WHERE id = ?", (session_id,)).fetchone()
        ui_msgs = json.loads(row["messages"]) if row and row["messages"] else []

    ui_msgs.append({"role": "user", "content": user_text})
    ui_msgs.append({"role": "agent", "content": decision.get("message", ""), "decision": decision, "qc": qc})

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, history, messages) VALUES (?, ?, ?)",
            (session_id, json.dumps(raw_history), json.dumps(ui_msgs)),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions")
def list_sessions():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, messages, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        msgs = json.loads(r["messages"] or "[]")
        turns = sum(1 for m in msgs if m["role"] == "user")
        preview = next((m["content"] for m in msgs if m["role"] == "user"), "")
        result.append({
            "session_id": r["id"],
            "created_at": r["created_at"],
            "turns": turns,
            "preview": preview[:80],
        })
    return result


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

    qc = run_qc(decision)
    _save_turn(session_id, updated_history, guardrail["cleaned_input"], decision, qc)

    return RunResponse(session_id=session_id, decision=decision, qc=qc)


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, messages, created_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = json.loads(row["messages"] or "[]")
    return {
        "session_id": row["id"],
        "messages": msgs,
        "turns": sum(1 for m in msgs if m["role"] == "user"),
        "created_at": row["created_at"],
    }


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
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"proteoagent-{session_id[:8]}.pdf",
    )
