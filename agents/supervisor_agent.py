#!/usr/bin/env python3
"""
supervisor_agent.py — v0
Supervisor agent for ProteoAgent Server.
Analyzes user requests and produces a routing plan.
In v0, no specialists are wired up; the supervisor returns its routing analysis only.
"""

import json
import os
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are the Supervisor Agent of the ProteoAgent Server multi-agent system.
You coordinate four specialist agents:

  databank       → Steps 1-8: RAW file processing, MS1/MS2 matrices, TDPortal integration
  deconvolution  → Proteoform deconvolution from m/z CSV (7 algorithms, consensus masses)
  proxai         → Neural-network gradient feature discovery across retention bins
  bioinformatics → Signal annotations, quantification, protein identification

Routing rules:
- RAW files, NPZ, HDF5, TDPortal, PFR        → databank
- m/z CSV, proteoform masses, deconvolution   → deconvolution
- cast_* columns, gradient, feature discovery → proxai
- charge assignment, quantification, ID       → bioinformatics

Parallel execution:
- Deconvolution and ProXAI are independent and can run IN PARALLEL.
- Bioinformatics must wait for Databank + Deconvolution.

If the request is ambiguous, ask one clarifying question.
If the user asks for the full pipeline, route to all agents in the correct order."""


_ROUTING_TEMPLATE = """Analyze this user request. Write a conversational reply to the user AND decide which agents are needed.

User request: {user_input}

Respond in this exact JSON format (no extra text):
{{
  "message": "Conversational reply directly to the user (1-3 sentences). If this is a task, confirm what you are routing. If conversational, answer naturally. If ambiguous, ask one clarifying question.",
  "agents_needed": [],
  "parallel_pairs": [],
  "run_parallel": false,
  "tasks": {{}},
  "reasoning": "Internal reasoning for your routing decision",
  "next_step": "What should happen after these agents complete (or what the user should do next)"
}}

Only include agents relevant to the request. If the request is conversational or informational, leave agents_needed as []."""


def build_llm(llm_choice: str, anthropic_api_key: Optional[str] = None):
    if llm_choice == "claude":
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key required for llm_choice=claude")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-opus-4-7", api_key=api_key, max_tokens=4096)
    if llm_choice == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model="qwen2.5:3b", base_url=base_url)
    raise ValueError(f"Unknown LLM '{llm_choice}'. Use 'claude' or 'ollama'.")


def run_supervisor(user_input: str, llm, history: list | None = None) -> tuple[dict, list]:
    """Call the LLM supervisor and return (decision, updated_history)."""
    prompt = _ROUTING_TEMPLATE.format(user_input=user_input)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )

    updated_history = list(history or [])
    updated_history.append(HumanMessage(content=prompt))
    updated_history.append(AIMessage(content=content))

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), updated_history
        except json.JSONDecodeError:
            pass

    return _fallback(user_input), updated_history


def _fallback(user_input: str) -> dict:
    return {
        "message": "I received your request but had trouble parsing my routing decision. I've defaulted to the databank agent — please review its output and re-submit if needed.",
        "agents_needed": ["databank"],
        "parallel_pairs": [],
        "run_parallel": False,
        "tasks": {"databank": user_input},
        "reasoning": "Could not parse routing response; defaulting to databank agent.",
        "next_step": "Review databank agent output then re-submit.",
    }
