from auxiliary_agents import run_qc

_GOOD = {
    "agents_needed": ["databank"],
    "tasks": {"databank": "Process RAW file"},
    "reasoning": "User submitted a RAW file",
    "next_step": "Run deconvolution",
}


def test_passing_qc():
    r = run_qc(_GOOD)
    assert r["passed"] is True
    assert r["warnings"] == []

def test_no_agents_fails():
    r = run_qc({**_GOOD, "agents_needed": [], "tasks": {}})
    assert r["passed"] is False
    assert any("No agents" in w for w in r["warnings"])

def test_agent_missing_task_warns():
    r = run_qc({**_GOOD, "tasks": {}})
    assert any("databank" in w for w in r["warnings"])

def test_bioinformatics_without_upstream_warns():
    r = run_qc({
        "agents_needed": ["bioinformatics"],
        "tasks": {"bioinformatics": "Identify proteins"},
        "reasoning": "ID requested",
        "next_step": "",
    })
    assert any("bioinformatics" in w for w in r["warnings"])

def test_bioinformatics_with_databank_passes():
    r = run_qc({
        "agents_needed": ["databank", "bioinformatics"],
        "tasks": {
            "databank": "Process RAW",
            "bioinformatics": "Identify proteins",
        },
        "reasoning": "Full pipeline",
        "next_step": "Done",
    })
    assert r["passed"] is True

def test_missing_reasoning_suggests():
    r = run_qc({**_GOOD, "reasoning": ""})
    assert any("reasoning" in s.lower() for s in r["suggestions"])

def test_multi_agent_passes():
    r = run_qc({
        "agents_needed": ["databank", "deconvolution"],
        "tasks": {
            "databank": "Process RAW",
            "deconvolution": "Deconvolve m/z",
        },
        "reasoning": "Both needed",
        "next_step": "Review results",
    })
    assert r["passed"] is True
