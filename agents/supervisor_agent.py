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

  databank_agent       → Steps 1-8: RAW file processing, MS1/MS2 matrices, TDPortal integration
  deconvolution_agent  → Proteoform deconvolution from m/z CSV (7 algorithms, consensus masses)
  proxai_agent         → Neural-network gradient feature discovery across retention bins
  bioinformatics_agent → Signal annotations, quantification, protein identification

Routing rules:
- RAW files, NPZ, HDF5, TDPortal, PFR        → databank_agent
- m/z CSV, proteoform masses, deconvolution   → deconvolution_agent
- cast_* columns, gradient, feature discovery → proxai_agent
- charge assignment, quantification, ID       → bioinformatics_agent

Parallel execution:
- Deconvolution and ProXAI are independent and can run IN PARALLEL.
- Bioinformatics must wait for Databank + Deconvolution.

If the request is ambiguous, ask one clarifying question.
If the user asks for the full pipeline, route to all agents in the correct order."""


_ROUTING_TEMPLATE = """Analyze this user request and decide which agents are needed.

User request: {user_input}

Respond in this exact JSON format (no extra text):
{{
  "agents_needed": ["databank"],
  "parallel_pairs": [],
  "run_parallel": false,
  "tasks": {{
    "databank": "specific task description"
  }},
  "reasoning": "brief explanation",
  "next_step": "suggested next step after these agents complete"
}}

Only include agents that are relevant to the request."""


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
        "agents_needed": ["databank"],
        "parallel_pairs": [],
        "run_parallel": False,
        "tasks": {"databank": user_input},
        "reasoning": "Could not parse routing response; defaulting to databank agent.",
        "next_step": "Review databank agent output then re-submit.",
    }
