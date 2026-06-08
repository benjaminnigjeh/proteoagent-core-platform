import json
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from supervisor_agent import _fallback, build_llm, run_supervisor


# ── _fallback ─────────────────────────────────────────────────────────────────

def test_fallback_has_required_keys():
    r = _fallback("test input")
    for key in ("message", "agents_needed", "parallel_pairs", "run_parallel", "tasks", "reasoning", "next_step"):
        assert key in r

def test_fallback_agents_is_list():
    assert isinstance(_fallback("x")["agents_needed"], list)

def test_fallback_message_is_string():
    assert isinstance(_fallback("x")["message"], str)


# ── build_llm ─────────────────────────────────────────────────────────────────

def test_build_llm_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM"):
        build_llm("gpt4")

def test_build_llm_claude_no_key_raises():
    with pytest.raises(ValueError, match="API key"):
        build_llm("claude", anthropic_api_key="")

def test_build_llm_claude_empty_env_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_llm("claude")


# ── run_supervisor ────────────────────────────────────────────────────────────

def _make_llm(json_payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=json.dumps(json_payload))
    return llm


def test_run_supervisor_returns_decision_and_history():
    payload = {
        "message": "Routing to databank.",
        "agents_needed": ["databank"],
        "parallel_pairs": [],
        "run_parallel": False,
        "tasks": {"databank": "Process RAW file"},
        "reasoning": "RAW file detected",
        "next_step": "Run deconvolution",
    }
    decision, history = run_supervisor("Process my RAW file", _make_llm(payload))
    assert decision["agents_needed"] == ["databank"]
    assert decision["message"] == "Routing to databank."
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)

def test_run_supervisor_history_is_extended():
    payload = {"message": "ok", "agents_needed": [], "parallel_pairs": [], "run_parallel": False, "tasks": {}, "reasoning": "chat", "next_step": ""}
    prior = [HumanMessage(content="prior"), AIMessage(content="reply")]
    _, history = run_supervisor("follow-up", _make_llm(payload), history=prior)
    assert len(history) == 4

def test_run_supervisor_fallback_on_bad_json():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="this is not json at all !!!")
    decision, history = run_supervisor("something", llm)
    assert "agents_needed" in decision
    assert len(history) == 2

def test_run_supervisor_list_content_joined():
    llm = MagicMock()
    payload = {"message": "ok", "agents_needed": ["proxai"], "parallel_pairs": [], "run_parallel": False, "tasks": {"proxai": "gradient"}, "reasoning": "r", "next_step": "n"}
    llm.invoke.return_value = AIMessage(content=[{"text": json.dumps(payload)}])
    decision, _ = run_supervisor("feature discovery", llm)
    assert decision["agents_needed"] == ["proxai"]
