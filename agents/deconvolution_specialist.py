#!/usr/bin/env python3
"""
deconvolution_specialist.py
Deconvolution specialist agent — owns proteoform deconvolution.
Called by the supervisor in the multi-agent framework.
"""

import json
from langchain_core.tools import StructuredTool
import deconvolution_backend as deconv

SYSTEM_PROMPT = """You are the Deconvolution Specialist Agent in the ProteoAgent Server multi-agent system.
You are responsible for proteoform deconvolution using up to 7 algorithms:
  UniDec, ms_deisotope, pyOpenMS, FLASHDeconv, TopFD, Sparse_NNLS, Peak_Charge_Collapse

Your one tool is run_deconvolution — it reads an m/z intensity CSV and outputs:
  - Per-algorithm proteoform mass CSVs
  - A combined long-table CSV
  - A CONSENSUS proteoform masses CSV (most important)
  - An algorithm summary CSV
  - Spectrum plots (if make_plots=True)

Rules:
- The input CSV must have an m/z column and an intensity column.
- Always report the consensus CSV path and how many consensus proteoforms were found.
- If an algorithm is unavailable (no executable), it is skipped gracefully.
- Return a clear summary of all output files when done."""


def _logs():
    logs = []; return logs, logs.append


def run_deconvolution_tool(
    input_csv: str,
    out_dir: str = None,
    mz_col: str = "m/z",
    intensity_col: str = None,
    min_charge: int = 2,
    max_charge: int = 30,
    min_mass: float = 5000.0,
    max_mass: float = 100000.0,
    consensus_ppm: float = 30.0,
    make_plots: bool = True,
    enabled_algorithms: list = None,
    unidec_exe: str = None,
    flashdeconv_exe: str = None,
    topfd_exe: str = None,
) -> str:
    """Run proteoform deconvolution on an m/z intensity CSV using up to 7 algorithms."""
    logs, log = _logs()
    try:
        r = deconv.run_deconvolution(
            input_csv=input_csv, out_dir=out_dir,
            mz_col=mz_col, intensity_col=intensity_col,
            min_charge=min_charge, max_charge=max_charge,
            min_mass=min_mass, max_mass=max_mass,
            consensus_ppm=consensus_ppm, make_plots=make_plots,
            enabled_algorithms=enabled_algorithms,
            unidec_exe=unidec_exe, flashdeconv_exe=flashdeconv_exe,
            topfd_exe=topfd_exe, log=log,
        )
        n_consensus = len(r.get("consensus", [])) if r.get("consensus") is not None else 0
        return json.dumps({
            "status": "ok",
            "out_dir": r["out_dir"],
            "consensus_csv": r.get("consensus_csv", ""),
            "summary_csv": r.get("summary_csv", ""),
            "all_csv": r.get("all_csv", ""),
            "n_consensus_proteoforms": n_consensus,
            "log": logs,
        })
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e),
                           "traceback": traceback.format_exc(), "log": logs})


def build_tools():
    return [
        StructuredTool.from_function(
            func=run_deconvolution_tool,
            name="run_deconvolution",
            description=(
                "Run proteoform deconvolution on an m/z intensity CSV using up to 7 algorithms "
                "(UniDec, ms_deisotope, pyOpenMS, FLASHDeconv, TopFD, Sparse_NNLS, Peak_Charge_Collapse). "
                "Outputs consensus proteoform masses CSV and per-algorithm results."
            ),
        ),
    ]
