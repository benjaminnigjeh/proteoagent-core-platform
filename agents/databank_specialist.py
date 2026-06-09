#!/usr/bin/env python3
"""
databank_specialist.py
Databank specialist agent — owns Steps 1-8 of the databank pipeline.
Called by the supervisor in the multi-agent framework.
"""

import json, os, subprocess
from langchain_core.tools import StructuredTool
import databank_backend as db
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PY_CASTING = os.path.join(_HERE, "casting", "python.exe")
RAW_WORKER = os.path.join(_HERE, "raw_worker.py")

SYSTEM_PROMPT = """You are the Databank Specialist Agent in the ProteoAgent Server multi-agent system.
You are responsible for Steps 1-8 of the Databank Generation Workflow:

  generate_ms_matrices      (Steps 1-2) → reads .raw files, generates MS1/MS2/meta NPZ files
  combine_ms2_to_hdf5       (Step 3)    → merges MS2 NPZ files into HDF5 spectral library
  combine_ms2_to_csv        (Step 4)    → exports MS2 scan metadata to CSV index
  bin_ms1                   (Step 5)    → bins MS1 scans into sliding RT windows
  combine_tdportal_reports  (Step 6)    → merges TDPortal identification CSVs
  import_tdportal_ids       (Step 7)    → matches TDPortal IDs to databank scans
  remove_missing_pfr        (Step 8)    → removes rows without PFR identification

Rules:
- Always confirm folder paths and group names before running Steps 1-2 (they are slow).
- After every tool call, report what was saved.
- Return a clear summary of all output file paths when done.
- If a step fails, explain the error and suggest a fix."""


def _logs():
    logs = []; return logs, logs.append


# ── Tool functions ─────────────────────────────────────────────────────────────

def run_generate_ms_matrices(
    folder_paths: list, out_dir: str, groups: list,
    ignore_keywords: list = None,
    ms1_mz_min: float = 600.0, ms1_mz_max: float = 1969.0, ms1_bin_da: float = 0.1,
    ms2_mz_min: float = 400.0, ms2_mz_max: float = 1999.0, ms2_bin_da: float = 1.0,
) -> str:
    """Steps 1-2: Read Thermo .raw files and generate MS1/MS2/meta NPZ matrix files per group."""
    job = {"folder_paths": folder_paths, "out_dir": out_dir, "groups": groups,
           "ignore_keywords": ignore_keywords or [],
           "ms1_mz_min": ms1_mz_min, "ms1_mz_max": ms1_mz_max, "ms1_bin_da": ms1_bin_da,
           "ms2_mz_min": ms2_mz_min, "ms2_mz_max": ms2_mz_max, "ms2_bin_da": ms2_bin_da}
    proc = subprocess.run([PY_CASTING, RAW_WORKER], input=json.dumps(job),
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return json.dumps({"status": "error", "message": proc.stderr.strip()})
    try:
        return proc.stdout
    except Exception:
        return json.dumps({"status": "error", "message": f"Unexpected: {proc.stdout[:300]}"})

def run_combine_ms2_to_hdf5(npz_paths: list, h5_out_path: str) -> str:
    """Step 3: Combine MS2 NPZ files into a single HDF5 spectral library."""
    logs, log = _logs()
    try:
        db.combine_ms2_npz_to_hdf5(npz_paths, h5_out_path, log=log)
        return json.dumps({"status": "ok", "file": h5_out_path, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})

def run_combine_ms2_to_csv(npz_paths: list, csv_out_path: str) -> str:
    """Step 4: Export MS2 scan metadata from NPZ files into a CSV index."""
    logs, log = _logs()
    try:
        db.combine_ms2_npz_to_csv(npz_paths, csv_out_path, log=log)
        return json.dumps({"status": "ok", "file": csv_out_path, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})

def run_bin_ms1(npz_path: str, drift_path: str, out_csv: str,
                rt_start: float = 10.0, rt_end: float = 80.0,
                bin_w: float = 10.0, bin_step: float = 5.0) -> str:
    """Step 5: Bin MS1 scans into sliding RT windows with drift correction."""
    logs, log = _logs()
    try:
        db.bin_ms1_npz(npz_path=npz_path, drift_path=drift_path, out_csv=out_csv,
                       rt_start=rt_start, rt_end=rt_end, bin_w=bin_w, bin_step=bin_step, log=log)
        return json.dumps({"status": "ok", "file": out_csv, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})

def run_combine_tdportal_reports(csv_paths: list, out_path: str) -> str:
    """Step 6: Merge multiple TDPortal CSV reports into one combined CSV."""
    logs, log = _logs()
    try:
        db.combine_tdportal_reports(csv_paths, out_path, log=log)
        return json.dumps({"status": "ok", "file": out_path, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})

def run_import_tdportal_ids(tdportal_csv: str, databank_csv: str, out_csv: str) -> str:
    """Step 7: Match TDPortal protein IDs to databank scans by scan number."""
    logs, log = _logs()
    try:
        tdf = pd.read_csv(tdportal_csv); ddf = pd.read_csv(databank_csv)
        log(f"TDPortal: {len(tdf):,} rows  Databank: {len(ddf):,} rows")
        db.add_tdportal_ids(tdf, ddf, out_csv, log=log)
        return json.dumps({"status": "ok", "file": out_csv, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})

def run_remove_missing_pfr(input_csv: str, output_csv: str) -> str:
    """Step 8: Remove rows with no PFR identification from a databank CSV."""
    logs, log = _logs()
    try:
        db.remove_missing_pfr(input_csv, output_csv, log=log)
        return json.dumps({"status": "ok", "file": output_csv, "log": logs})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "log": logs})


# ── Build tool list ────────────────────────────────────────────────────────────

def build_tools():
    return [
        StructuredTool.from_function(func=run_generate_ms_matrices, name="generate_ms_matrices",
            description="Steps 1-2: Read Thermo .raw files, generate MS1/MS2/meta NPZ files per group. Confirm paths first."),
        StructuredTool.from_function(func=run_combine_ms2_to_hdf5, name="combine_ms2_to_hdf5",
            description="Step 3: Combine MS2 NPZ files into a single HDF5 spectral library."),
        StructuredTool.from_function(func=run_combine_ms2_to_csv, name="combine_ms2_to_csv",
            description="Step 4: Export MS2 scan metadata from NPZ files into a CSV index."),
        StructuredTool.from_function(func=run_bin_ms1, name="bin_ms1",
            description="Step 5: Bin MS1 scans into sliding RT windows with drift correction."),
        StructuredTool.from_function(func=run_combine_tdportal_reports, name="combine_tdportal_reports",
            description="Step 6: Merge multiple TDPortal CSV reports into one combined CSV."),
        StructuredTool.from_function(func=run_import_tdportal_ids, name="import_tdportal_ids",
            description="Step 7: Match TDPortal protein IDs to databank scans by scan number."),
        StructuredTool.from_function(func=run_remove_missing_pfr, name="remove_missing_pfr",
            description="Step 8: Remove rows with no PFR identification from a databank CSV."),
    ]
