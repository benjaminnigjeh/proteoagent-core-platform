#!/usr/bin/env python3
"""
auxiliary_agents.py — v0
Four auxiliary agents that wrap the supervisor pipeline.

  GuardRailing   — validates user input before it reaches the supervisor
  QC             — quality-checks the supervisor's routing decision
  ErrorHandling  — classifies exceptions into user-friendly messages
  Reporting      — exports a session's conversation history to a PDF file
"""

import json
import os
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage


# ── GuardRailing ───────────────────────────────────────────────────────────────

_BLOCK_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"forget (all|previous|your) instructions",
    r"(system|assistant)\s*:",
    r"<\|.*?\|>",
    r"#{3,}\s*(system|prompt|instruction)",
]


def run_guardrail(user_input: str) -> dict:
    """Return {"allowed": bool, "reason": str, "cleaned_input": str}."""
    if not user_input or not user_input.strip():
        return {"allowed": False, "reason": "Input is empty.", "cleaned_input": ""}

    cleaned = user_input.strip()

    if len(cleaned) < 5:
        return {"allowed": False, "reason": "Input is too short to process.", "cleaned_input": cleaned}

    for pattern in _BLOCK_PATTERNS:
        if re.search(pattern, cleaned, re.I):
            return {
                "allowed": False,
                "reason": "Input contains disallowed patterns.",
                "cleaned_input": "",
            }

    return {"allowed": True, "reason": "Input passed guardrail checks.", "cleaned_input": cleaned}


# ── QC ────────────────────────────────────────────────────────────────────────

def run_qc(decision: dict) -> dict:
    """Return {"passed": bool, "warnings": [...], "suggestions": [...]}."""
    warnings    = []
    suggestions = []
    passed      = True

    agents = decision.get("agents_needed", [])
    tasks  = decision.get("tasks", {})

    if not agents:
        warnings.append("No agents selected — the supervisor could not route this request.")
        passed = False

    for agent in agents:
        if agent not in tasks:
            warnings.append(f"'{agent}' is selected but has no task description.")

    if (
        "bioinformatics" in agents
        and "databank" not in agents
        and "deconvolution" not in agents
    ):
        warnings.append(
            "bioinformatics_agent requires upstream output from databank or deconvolution."
        )
        suggestions.append(
            "Add databank_agent and/or deconvolution_agent before bioinformatics_agent."
        )

    if not decision.get("reasoning"):
        suggestions.append("The supervisor provided no reasoning for this routing.")

    return {"passed": passed, "warnings": warnings, "suggestions": suggestions}


# ── Error Handling ─────────────────────────────────────────────────────────────

_ERROR_PATTERNS = [
    (
        re.compile(r"api.?key|authentication|unauthorized|401", re.I),
        "api_key",
        "API key is missing or invalid. Please check your Anthropic API key.",
        True,
    ),
    (
        re.compile(r"timeout|timed.?out|connection.?refused|connection.?error", re.I),
        "timeout",
        "The request timed out or the server is unreachable. Please try again.",
        True,
    ),
    (
        re.compile(r"rate.?limit|too.?many.?requests|429", re.I),
        "rate_limit",
        "Rate limit reached. Please wait a moment before retrying.",
        True,
    ),
    (
        re.compile(r"model.*not.?found|invalid.?model|model.*unavailable", re.I),
        "model",
        "The selected model is unavailable. Try switching the LLM backend.",
        False,
    ),
]


def classify_error(exc: Exception) -> dict:
    """Return {"error_type", "user_message", "recoverable", "raw"}."""
    raw = f"{type(exc).__name__}: {exc}"
    for pattern, error_type, message, recoverable in _ERROR_PATTERNS:
        if pattern.search(raw):
            return {
                "error_type":   error_type,
                "user_message": message,
                "recoverable":  recoverable,
                "raw":          raw,
            }
    return {
        "error_type":   "unknown",
        "user_message": "An unexpected error occurred. See the details panel for more information.",
        "recoverable":  False,
        "raw":          raw,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_session_pdf(session_id: str, history: list, output_folder: str) -> str:
    """
    Generate a PDF report of the session history and return the saved file path.
    Requires fpdf2: pip install fpdf2
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 is not installed. Run: pip install fpdf2")

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"session_{session_id[:8]}_{timestamp}.pdf"
    filepath  = os.path.join(output_folder, filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ProteoAgent Server  -  Session Report", ln=True)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"Session ID : {session_id}", ln=True)
    pdf.cell(0, 6, f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Turns      : {len(history) // 2}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ── Conversation turns ────────────────────────────────────────────────────
    for i, msg in enumerate(history):
        if isinstance(msg, HumanMessage):
            turn_num = i // 2 + 1

            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(235, 241, 255)
            pdf.set_text_color(30, 60, 140)
            pdf.cell(0, 8, f"  Turn {turn_num}  -  User", ln=True, fill=True)

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", size=10)
            content = msg.content
            # Strip the routing template wrapper — show only the original query
            m = re.search(r"User request:\s*(.*?)\n\nRespond", content, re.DOTALL)
            if m:
                content = m.group(1).strip()
            pdf.multi_cell(0, 6, content)
            pdf.ln(3)

        elif isinstance(msg, AIMessage):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(235, 255, 241)
            pdf.set_text_color(20, 110, 55)
            pdf.cell(0, 8, "  Supervisor  -  Routing Decision", ln=True, fill=True)

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", size=10)
            content = msg.content
            # Pretty-print the JSON routing decision if present
            try:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    parsed  = json.loads(m.group())
                    content = json.dumps(parsed, indent=2)
            except Exception:
                pass
            pdf.multi_cell(0, 6, content)
            pdf.ln(5)

    pdf.output(filepath)
    return filepath
