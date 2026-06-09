#!/usr/bin/env python3
"""
bioinformatic_specialist.py
Bioinformatics specialist agent — owns annotations, quantification, identification.
Called by the supervisor in the multi-agent framework.
"""

import json
from langchain_core.tools import StructuredTool
import bioinformatic_backend as bio

SYSTEM_PROMPT = """You are the Bioinformatics Specialist Agent in the ProteoAgent Server multi-agent system.
You are responsible for three sequential bioinformatics steps:

  run_annotations     → detect neutral-mass peaks from deconvolution output and assign
                        charge series from raw MS1 CSVs. Outputs combined assignments CSV.
  run_quantification  → quantify proteoforms by summing cast_* intensity columns from the
                        training dataset for each matched m/z in the assignments CSV.
  run_identification  → fast NumPy databank search to match charge assignments to protein
                        identifications using RT window, m/z tolerance, and mass tolerance.

Rules:
- These steps run sequentially: annotations → quantification → identification.
- run_annotations requires a deconvolution output folder and a raw MS1 folder.
- run_quantification requires the training dataset CSV and the assignments CSV.
- run_identification requires the databank CSV (with rt_aligned, precursor_mz, MASS, Accession, PFR).
- Always report output file paths and row counts after each step.
- Return a clear summary of all output files when done."""


def _logs():
    logs = []; return logs, logs.append


def run_annotations_tool(
    deconv_folder: str,
    raw_ms1_folder: str,
    out_dir: str,
    z_min: int = 5,
    z_max: int = 50,
    ppm_tol: float = 1000.0,
    abs_da_tol: float = 1.0,
    min_matched: int = 4,
) -> str:
    """Detect neutral-mass peaks and assign charge series from raw MS1 CSVs."""
    logs, log = _logs()
    try:
        combined = bio.run_annotations(
            deconv_folder=deconv_folder, raw_ms1_folder=raw_ms1_folder,
            out_dir=out_dir, z_min=z_min, z_max=z_max,
            ppm_tol=ppm_tol, abs_da_tol=abs_da_tol,
            min_matched=min_matched, log_fn=log,
        )
        return json.dumps({"status": "ok", "combined_report": combined, "log": logs})
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e),
                           "traceback": traceback.format_exc(), "log": logs})


def run_quantification_tool(
    dataset_path: str,
    assignments_path: str,
    out_path: str,
) -> str:
    """Quantify proteoforms by summing cast_* intensity columns from the training dataset."""
    logs, log = _logs()
    try:
        bio.run_quantification(
            dataset_path=dataset_path,
            assignments_path=assignments_path,
            out_path=out_path,
            log_fn=log,
        )
        return json.dumps({"status": "ok", "file": out_path, "log": logs})
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e),
                           "traceback": traceback.format_exc(), "log": logs})


def run_identification_tool(
    charge_file: str,
    databank_path: str,
    output: str,
    param_sets: list = None,
    keep_placeholders: bool = True,
    min_mode_pfr_count: int = 1,
) -> str:
    """Fast NumPy databank search to match charge assignments to protein identifications."""
    logs, log = _logs()
    try:
        if param_sets is None:
            param_sets = [[55.0, 2.0, 90.0]]
        param_sets = [tuple(ps) for ps in param_sets]
        bio.run_identification(
            charge_file=charge_file,
            databank_path=databank_path,
            output=output,
            param_sets=param_sets,
            keep_placeholders=keep_placeholders,
            min_mode_pfr_count=min_mode_pfr_count,
            log_fn=log,
        )
        return json.dumps({"status": "ok", "file": output, "log": logs})
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e),
                           "traceback": traceback.format_exc(), "log": logs})


def build_tools():
    return [
        StructuredTool.from_function(
            func=run_annotations_tool,
            name="run_annotations",
            description=(
                "Bioinformatics Step 1: Detect neutral-mass peaks from deconvolution output "
                "and assign charge series from raw MS1 CSVs. Outputs combined assignments report."
            ),
        ),
        StructuredTool.from_function(
            func=run_quantification_tool,
            name="run_quantification",
            description=(
                "Bioinformatics Step 2: Quantify proteoforms by summing cast_* intensity "
                "columns from the training dataset for each matched m/z in assignments CSV."
            ),
        ),
        StructuredTool.from_function(
            func=run_identification_tool,
            name="run_identification",
            description=(
                "Bioinformatics Step 3: Fast NumPy databank search to match charge assignments "
                "to protein identifications using RT window, m/z tolerance, and mass tolerance."
            ),
        ),
    ]
