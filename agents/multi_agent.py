#!/usr/bin/env python3
"""
multi_agent.py
ProteoAgent Server Multi-Agent Framework — main entry point.

Architecture:
  You → Supervisor → routes to specialist agents (parallel where possible)
         ├── Databank Agent        (Steps 1-8)
         ├── Deconvolution Agent   (7 algorithms)   ┐ parallel
         ├── ProXAI Agent          (gradient XAI)   ┘ parallel
         └── Bioinformatics Agent  (annotate/quant/id)

Parallel execution:
  - Deconvolution + ProXAI run concurrently (independent of each other)
  - Bioinformatics waits for Databank + Deconvolution outputs

Setup:
    conda activate claude_databank
    pip install langgraph langchain-anthropic langchain-ollama

Usage (Claude API):
    set ANTHROPIC_API_KEY=sk-ant-...
    python multi_agent.py --llm claude

Usage (Ollama / local):
    python multi_agent.py --llm ollama
"""

import os, sys, json, argparse
from typing import Annotated, TypedDict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import databank_specialist    as db_spec
import deconvolution_specialist as deconv_spec
import proxai_specialist       as proxai_spec
import bioinformatic_specialist as bio_spec
import supervisor_agent        as sup

# ── Config ────────────────────────────────────────────────────────────────────
MEMORY_DB    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_agent_memory.db")
OLLAMA_MODEL = "qwen2.5:3b"
SESSION_ID   = "proxai_multi_agent"


# ══════════════════════════════════════════════════════════════════════════════
# Graph state
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:              Annotated[list[BaseMessage], add_messages]
    user_input:            str
    supervisor_decision:   str          # which agents to invoke
    databank_result:       str
    deconvolution_result:  str
    proxai_result:         str
    bioinformatics_result: str
    final_summary:         str
    run_parallel:          bool         # whether to run deconv+proxai in parallel


# ══════════════════════════════════════════════════════════════════════════════
# Build LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_llm(llm_choice: str, anthropic_api_key: Optional[str] = None):
    if llm_choice == "claude":
        api_key = anthropic_api_key if anthropic_api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key required for llm_choice=claude")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-opus-4-7", api_key=api_key, max_tokens=4096)
    elif llm_choice == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url="http://localhost:11434")
    else:
        raise ValueError(f"Unknown LLM '{llm_choice}'. Use 'claude' or 'ollama'.")


# ══════════════════════════════════════════════════════════════════════════════
# Agent node builders
# ══════════════════════════════════════════════════════════════════════════════

def make_specialist_node(llm, tools, system_prompt, agent_name: str):
    """Create a LangGraph node that runs a specialist react agent."""
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    def node_fn(state: AgentState) -> dict:
        print(f"\n  [{agent_name}] Starting...")
        user_msg = state.get("user_input", "")
        supervisor_ctx = state.get("supervisor_decision", "")

        # Build context message for specialist
        context = f"Task from supervisor: {supervisor_ctx}\n\nUser original request: {user_msg}"

        # Add any results from prior agents as context
        if agent_name == "Bioinformatics Agent":
            db_res   = state.get("databank_result", "")
            dec_res  = state.get("deconvolution_result", "")
            if db_res:   context += f"\n\nDatabank Agent completed:\n{db_res}"
            if dec_res:  context += f"\n\nDeconvolution Agent completed:\n{dec_res}"

        try:
            response = agent.invoke({"messages": [HumanMessage(content=context)]})
            msgs = response.get("messages", [])
            answer = ""
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    content = m.content
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content)
                    if content.strip():
                        answer = content.strip()
                        break
            print(f"  [{agent_name}] Done.")
            return {f"{agent_name.lower().replace(' ', '_')}_result": answer}
        except Exception as e:
            error = f"[{agent_name} ERROR] {e}"
            print(f"  {error}")
            return {f"{agent_name.lower().replace(' ', '_')}_result": error}

    return node_fn


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor node
# ══════════════════════════════════════════════════════════════════════════════

def make_supervisor_node(llm):
    """Create the supervisor node that decides routing."""

    def supervisor_node(state: AgentState) -> dict:
        # If the API already computed a routing decision, skip the LLM call.
        existing = state.get("supervisor_decision", "")
        if existing:
            try:
                decision = json.loads(existing)
                print(f"\n[Supervisor] Using pre-computed routing: {decision.get('agents_needed', [])}")
                return {
                    "supervisor_decision": existing,
                    "run_parallel": decision.get("run_parallel", False),
                }
            except Exception:
                pass

        user_input = state.get("user_input", "")
        print(f"\n[Supervisor] Analyzing request...")

        # Ask LLM to decide which agents are needed and if parallel is possible
        routing_prompt = f"""You are the ProteoAgent Server Supervisor. Analyze this user request and decide:
1. Which specialist agents are needed (databank, deconvolution, proxai, bioinformatics)?
2. Can any run in parallel? (deconvolution + proxai are always parallel-safe)
3. What is the task for each agent?

User request: {user_input}

Respond in this exact JSON format:
{{
  "agents_needed": ["databank", "deconvolution", "proxai", "bioinformatics"],
  "parallel_pairs": [["deconvolution", "proxai"]],
  "run_parallel": true,
  "tasks": {{
    "databank": "specific task for databank agent",
    "deconvolution": "specific task for deconvolution agent",
    "proxai": "specific task for proxai agent",
    "bioinformatics": "specific task for bioinformatics agent"
  }},
  "reasoning": "brief explanation of routing decision"
}}

Only include agents that are actually needed. Omit agents not relevant to this request."""

        try:
            response = llm.invoke([HumanMessage(content=routing_prompt)])
            content = response.content
            if isinstance(content, list):
                content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c)
                                   for c in content)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = {"agents_needed": ["databank"], "run_parallel": False,
                            "tasks": {"databank": user_input}, "reasoning": "fallback"}
        except Exception as e:
            print(f"  [Supervisor] Routing error: {e}, defaulting to sequential")
            decision = {"agents_needed": ["databank"], "run_parallel": False,
                        "tasks": {"databank": user_input}, "reasoning": "error fallback"}

        print(f"  [Supervisor] Agents needed: {decision.get('agents_needed', [])}")
        print(f"  [Supervisor] Parallel: {decision.get('run_parallel', False)}")
        if decision.get("reasoning"):
            print(f"  [Supervisor] Reasoning: {decision['reasoning']}")

        return {
            "supervisor_decision": json.dumps(decision),
            "run_parallel": decision.get("run_parallel", False),
        }

    return supervisor_node


# ══════════════════════════════════════════════════════════════════════════════
# Parallel execution node
# ══════════════════════════════════════════════════════════════════════════════

def make_parallel_node(llm):
    """Run deconvolution and ProXAI in parallel using ThreadPoolExecutor."""

    deconv_agent = create_react_agent(
        model=llm, tools=deconv_spec.build_tools(),
        prompt=deconv_spec.SYSTEM_PROMPT)
    proxai_agent = create_react_agent(
        model=llm, tools=proxai_spec.build_tools(),
        prompt=proxai_spec.SYSTEM_PROMPT)

    def run_agent(agent, task, name):
        try:
            response = agent.invoke({"messages": [HumanMessage(content=task)]})
            msgs = response.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    content = m.content
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content)
                    if content.strip():
                        return name, content.strip()
            return name, "No response."
        except Exception as e:
            return name, f"[{name} ERROR] {e}"

    def parallel_node(state: AgentState) -> dict:
        decision_str = state.get("supervisor_decision", "{}")
        try:
            decision = json.loads(decision_str)
        except Exception:
            decision = {}

        tasks     = decision.get("tasks", {})
        needed    = decision.get("agents_needed", [])
        user_input = state.get("user_input", "")

        deconv_task = tasks.get("deconvolution", user_input)
        proxai_task = tasks.get("proxai", user_input)

        run_deconv = "deconvolution" in needed
        run_proxai = "proxai" in needed

        if not run_deconv and not run_proxai:
            return {"deconvolution_result": "", "proxai_result": ""}

        results = {"deconvolution_result": "", "proxai_result": ""}
        jobs = []
        if run_deconv: jobs.append((deconv_agent, deconv_task, "Deconvolution Agent"))
        if run_proxai: jobs.append((proxai_agent, proxai_task, "ProXAI Agent"))

        print(f"\n  [Parallel] Running {[j[2] for j in jobs]} concurrently...")

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {executor.submit(run_agent, j[0], j[1], j[2]): j[2] for j in jobs}
            for future in as_completed(futures):
                name, result = future.result()
                print(f"  [{name}] Completed.")
                key = "deconvolution_result" if "Deconv" in name else "proxai_result"
                results[key] = result

        return results

    return parallel_node


# ══════════════════════════════════════════════════════════════════════════════
# Final summary node
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_node(llm):
    def summary_node(state: AgentState) -> dict:
        print("\n[Supervisor] Generating final summary...")

        db_res    = state.get("databank_result", "")
        deconv_res = state.get("deconvolution_result", "")
        proxai_res = state.get("proxai_result", "")
        bio_res   = state.get("bioinformatics_result", "")

        parts = []
        if db_res:    parts.append(f"Databank Agent:\n{db_res}")
        if deconv_res: parts.append(f"Deconvolution Agent:\n{deconv_res}")
        if proxai_res: parts.append(f"ProXAI Agent:\n{proxai_res}")
        if bio_res:    parts.append(f"Bioinformatics Agent:\n{bio_res}")

        if not parts:
            return {"final_summary": "No agents produced output.",
                    "messages": [AIMessage(content="No agents produced output.")]}

        combined = "\n\n---\n\n".join(parts)
        summary_prompt = f"""You are the ProteoAgent Server Supervisor. Summarize the results from all specialist agents into a clear, concise final report for the user.

Agent results:
{combined}

Provide:
1. What was accomplished by each agent
2. All output files produced (with paths)
3. The logical next step in the pipeline (if any)

Keep it concise and actionable."""

        try:
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            content = response.content
            if isinstance(content, list):
                content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c)
                                   for c in content)
            summary = content.strip()
        except Exception as e:
            summary = f"Summary generation failed: {e}\n\nRaw results:\n{combined}"

        return {
            "final_summary": summary,
            "messages": [AIMessage(content=summary)],
        }

    return summary_node


# ══════════════════════════════════════════════════════════════════════════════
# Graph routing conditions
# ══════════════════════════════════════════════════════════════════════════════

def should_run_databank(state: AgentState) -> Literal["databank", "parallel_or_skip"]:
    try:
        decision = json.loads(state.get("supervisor_decision", "{}"))
        if "databank" in decision.get("agents_needed", []):
            return "databank"
    except Exception:
        pass
    return "parallel_or_skip"


def should_run_parallel(state: AgentState) -> Literal["parallel", "bioinformatics", "summary"]:
    try:
        decision = json.loads(state.get("supervisor_decision", "{}"))
        needed   = decision.get("agents_needed", [])
        if "deconvolution" in needed or "proxai" in needed:
            return "parallel"
        if "bioinformatics" in needed:
            return "bioinformatics"
    except Exception:
        pass
    return "summary"


def should_run_bioinformatics(state: AgentState) -> Literal["bioinformatics", "summary"]:
    try:
        decision = json.loads(state.get("supervisor_decision", "{}"))
        if "bioinformatics" in decision.get("agents_needed", []):
            return "bioinformatics"
    except Exception:
        pass
    return "summary"


# ══════════════════════════════════════════════════════════════════════════════
# Build the full graph
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(llm, checkpointer):
    # Build all specialist agents as nodes
    db_node      = make_specialist_node(llm, db_spec.build_tools(),
                                        db_spec.SYSTEM_PROMPT, "Databank Agent")
    parallel_node = make_parallel_node(llm)
    bio_node     = make_specialist_node(llm, bio_spec.build_tools(),
                                        bio_spec.SYSTEM_PROMPT, "Bioinformatics Agent")
    supervisor   = make_supervisor_node(llm)
    summary      = make_summary_node(llm)

    # Wrap node results to match state keys
    def databank_node(state):
        result = db_node(state)
        return {"databank_result": result.get("databank_agent_result", "")}

    def bioinformatics_node(state):
        result = bio_node(state)
        return {"bioinformatics_result": result.get("bioinformatics_agent_result", "")}

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("supervisor",      supervisor)
    graph.add_node("databank",        databank_node)
    graph.add_node("parallel",        parallel_node)
    graph.add_node("bioinformatics",  bioinformatics_node)
    graph.add_node("summary",         summary)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → databank or skip
    graph.add_conditional_edges(
        "supervisor",
        should_run_databank,
        {"databank": "databank", "parallel_or_skip": "parallel"},
    )

    # Databank → parallel or bio or summary
    graph.add_conditional_edges(
        "databank",
        should_run_parallel,
        {"parallel": "parallel", "bioinformatics": "bioinformatics", "summary": "summary"},
    )

    # Parallel (deconv + proxai) → bio or summary
    graph.add_conditional_edges(
        "parallel",
        should_run_bioinformatics,
        {"bioinformatics": "bioinformatics", "summary": "summary"},
    )

    # Bioinformatics → summary
    graph.add_edge("bioinformatics", "summary")

    # Summary → END
    graph.add_edge("summary", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_multi_agent_request(user_input: str, llm_choice: str = "ollama", anthropic_api_key: Optional[str] = None) -> dict:
    llm = build_llm(llm_choice, anthropic_api_key)

    conn = sqlite3.connect(MEMORY_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(llm, checkpointer)
    config = {"configurable": {"thread_id": SESSION_ID}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "supervisor_decision": "",
            "databank_result": "",
            "deconvolution_result": "",
            "proxai_result": "",
            "bioinformatics_result": "",
            "final_summary": "",
            "run_parallel": False,
        },
        config=config,
    )

    conn.close()
    return result


class MultiAgentSession:
    """Persistent session for multi-agent requests."""

    def __init__(self, llm_choice: str, anthropic_api_key: Optional[str] = None):
        self.llm = build_llm(llm_choice, anthropic_api_key)
        self.conn = sqlite3.connect(MEMORY_DB, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        self.graph = build_graph(self.llm, self.checkpointer)
        self.config = {"configurable": {"thread_id": SESSION_ID}}

    def invoke_request(self, user_input: str) -> dict:
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_input": user_input,
                "supervisor_decision": "",
                "databank_result": "",
                "deconvolution_result": "",
                "proxai_result": "",
                "bioinformatics_result": "",
                "final_summary": "",
                "run_parallel": False,
            },
            config=self.config,
        )
        return result

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_agent(llm_choice: str):
    llm = build_llm(llm_choice)

    conn         = sqlite3.connect(MEMORY_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph        = build_graph(llm, checkpointer)
    config       = {"configurable": {"thread_id": SESSION_ID}}

    llm_label = "Claude API" if llm_choice == "claude" else f"{OLLAMA_MODEL} via Ollama"

    print("=" * 64)
    print(f"  ProteoAgent Server  —  Multi-Agent Framework  —  {llm_label}")
    print("  4 specialist agents  ·  parallel deconv + ProXAI")
    print("  Memory:", MEMORY_DB)
    print("=" * 64)
    print()
    print("  Agents:")
    print("    Databank Agent       → Steps 1-8 (RAW → NPZ → HDF5 → CSV)")
    print("    Deconvolution Agent  → Proteoform deconvolution (7 algorithms)")
    print("    ProXAI Agent         → Neural-network gradient feature discovery")
    print("    Bioinformatics Agent → Annotations · Quantification · Identification")
    print()
    print("  Deconvolution + ProXAI run in PARALLEL when both are needed.")
    print()
    print("  Type your request and press Enter. Type 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break

        if not user_input: continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye."); break

        print()
        try:
            result = graph.invoke(
                {
                    "messages":              [HumanMessage(content=user_input)],
                    "user_input":            user_input,
                    "supervisor_decision":   "",
                    "databank_result":       "",
                    "deconvolution_result":  "",
                    "proxai_result":         "",
                    "bioinformatics_result": "",
                    "final_summary":         "",
                    "run_parallel":          False,
                },
                config=config,
            )
            summary = result.get("final_summary", "")
            if summary:
                print(f"\nSupervisor: {summary}\n")
            else:
                print("\n[No summary produced]\n")
        except Exception as e:
            import traceback
            print(f"\n[Multi-Agent Error] {e}")
            print(traceback.format_exc())
            print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProXAI Suite Multi-Agent Framework")
    parser.add_argument(
        "--llm", choices=["claude", "ollama"], default="ollama",
        help="LLM backend: 'claude' (Anthropic API) or 'ollama' (local, default)")
    args = parser.parse_args()
    run_multi_agent(args.llm)
