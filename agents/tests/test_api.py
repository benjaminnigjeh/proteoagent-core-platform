import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

_GOOD_DECISION = {
    "message": "Routing to databank agent.",
    "agents_needed": ["databank"],
    "parallel_pairs": [],
    "run_parallel": False,
    "tasks": {"databank": "Process RAW file"},
    "reasoning": "RAW file detected",
    "next_step": "Run deconvolution",
}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Sessions list ─────────────────────────────────────────────────────────────

def test_list_sessions_empty():
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == []

def test_list_sessions_after_run():
    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [])), \
         patch("api._execute_agents", return_value="Agent result."):
        client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})

    r = client.get("/api/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) == 1
    assert sessions[0]["turns"] == 1
    assert "Process my RAW" in sessions[0]["preview"]


# ── Session CRUD ──────────────────────────────────────────────────────────────

def test_get_session_not_found():
    r = client.get("/api/session/does-not-exist")
    assert r.status_code == 404

def test_get_session_after_run():
    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [])), \
         patch("api._execute_agents", return_value="Agent result."):
        run_r = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})
    sid = run_r.json()["session_id"]

    r = client.get(f"/api/session/{sid}")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == sid
    assert data["turns"] == 1
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "agent"

def test_delete_session():
    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [])), \
         patch("api._execute_agents", return_value="Agent result."):
        run_r = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})
    sid = run_r.json()["session_id"]

    client.delete(f"/api/session/{sid}")
    assert client.get(f"/api/session/{sid}").status_code == 404

def test_delete_nonexistent_session_ok():
    r = client.delete("/api/session/nonexistent")
    assert r.status_code == 200


# ── /api/run ──────────────────────────────────────────────────────────────────

def test_run_empty_input_blocked():
    r = client.post("/api/run", json={"user_input": "", "llm_choice": "ollama"})
    assert r.status_code == 400

def test_run_too_short_blocked():
    r = client.post("/api/run", json={"user_input": "hi", "llm_choice": "ollama"})
    assert r.status_code == 400

def test_run_unknown_llm_blocked():
    r = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "unknown"})
    assert r.status_code == 400

def test_run_success_returns_expected_fields():
    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [])), \
         patch("api._execute_agents", return_value="Agent result."):
        r = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})

    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["decision"]["agents_needed"] == ["databank"]
    assert "qc" in data
    assert "passed" in data["qc"]

def test_run_continues_existing_session():
    sid = None
    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [])), \
         patch("api._execute_agents", return_value="Agent result."):
        r1 = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})
        sid = r1.json()["session_id"]
        r2 = client.post(
            "/api/run",
            json={"session_id": sid, "user_input": "Now run deconvolution please", "llm_choice": "ollama"},
        )

    assert r2.json()["session_id"] == sid
    session = client.get(f"/api/session/{sid}").json()
    assert session["turns"] == 2


# ── PDF report ────────────────────────────────────────────────────────────────

def test_report_not_found():
    r = client.post("/api/session/nonexistent/report")
    assert r.status_code == 404

def test_report_returns_pdf():
    from langchain_core.messages import HumanMessage, AIMessage

    with patch("api.build_llm", return_value=MagicMock()), \
         patch("api.run_supervisor", return_value=(_GOOD_DECISION, [
             HumanMessage(content="test"), AIMessage(content=json.dumps(_GOOD_DECISION))
         ])), \
         patch("api._execute_agents", return_value="Agent result."):
        run_r = client.post("/api/run", json={"user_input": "Process my RAW file please", "llm_choice": "ollama"})
    sid = run_r.json()["session_id"]

    with patch("api.generate_session_pdf") as mock_pdf:
        import os
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"%PDF-1.4 fake pdf content")
        tmp.close()
        mock_pdf.return_value = tmp.name

        r = client.post(f"/api/session/{sid}/report")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        os.unlink(tmp.name)
