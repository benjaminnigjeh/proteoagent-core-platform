#!/usr/bin/env python3
"""
proxai_specialist.py
ProXAI specialist agent — owns neural-network gradient feature discovery.
Called by the supervisor in the multi-agent framework.
"""

import json
from langchain_core.tools import StructuredTool
import proxai_backend as proxai

SYSTEM_PROMPT = """You are the ProXAI Specialist Agent in the ProteoAgent Server multi-agent system.
You are responsible for neural-network gradient feature discovery across retention bins.

Your tools:
  proxai_discover_bins  → lists available retention bins in a CSV before running
  run_proxai            → trains dual neural-network ensembles (Run A + Run B) per bin
                          and extracts signed/positive/negative-abs gradients as m/z
                          importance scores. Outputs gradient CSVs, mirror plots, and
                          summary JSONs per bin.

Rules:
- Always run proxai_discover_bins first to show the user available bins.
- Confirm the target column and retention bin column before running.
- Report the output root folder and how many bins were processed when done.
- Warn the user if TensorFlow is not installed — ProXAI requires it.
- Return a clear summary of all output folders when done."""


def _logs():
    logs = []; return logs, logs.append


def proxai_discover_bins_tool(csv_path: str, retention_bin_col: str) -> str:
    """List all available retention bins in a ProXAI CSV before running feature discovery."""
    try:
        bins = proxai.proxai_discover_bins(csv_path, retention_bin_col)
        return json.dumps({"status": "ok", "bins": bins, "count": len(bins)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def run_proxai_tool(
    csv_path: str,
    out_root: str,
    target_col: str,
    retention_bin_col: str,
    cast_col_regex: str = r"^cast_(\d+)$",
    task_type: str = "auto",
    hidden_units: list = None,
    dropout: float = 0.2,
    l1: float = 1e-4,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 32,
    k_splits: int = 5,
    n_repeats: int = 1,
    seed_A: int = 42,
    seed_B: int = 99,
    normalization_method: str = "none",
    mz_start: float = 600.0,
    mz_step: float = 0.1,
    retention_bin_values: list = None,
    max_points: int = None,
    fillna_value: float = 0.0,
    use_batch_embedding: bool = False,
    batch_col: str = "",
    batch_embedding_dim: int = 4,
    save_combined_csv: bool = True,
    save_plot_signed: bool = True,
    save_plot_pos: bool = True,
    save_plot_negabs: bool = True,
    save_summary_json: bool = True,
    gradient_sample_mode: str = "all",
    eps: float = 1e-7,
) -> str:
    """Run ProXAI neural-network gradient feature discovery across retention bins."""
    logs, log = _logs()
    try:
        cfg = {
            "csv_path": csv_path, "out_root": out_root,
            "target_col": target_col, "retention_bin_col": retention_bin_col,
            "cast_col_regex": cast_col_regex, "task_type": task_type,
            "hidden_units": tuple(hidden_units or [128, 64]),
            "dropout": dropout, "l1": l1, "learning_rate": learning_rate,
            "epochs": epochs, "batch_size": batch_size,
            "k_splits": k_splits, "n_repeats": n_repeats,
            "seed_A": seed_A, "seed_B": seed_B,
            "normalization_method": normalization_method,
            "mz_start": mz_start, "mz_step": mz_step,
            "retention_bin_values": retention_bin_values,
            "max_points": max_points, "fillna_value": fillna_value,
            "use_batch_embedding": use_batch_embedding,
            "batch_col": batch_col, "batch_embedding_dim": batch_embedding_dim,
            "save_combined_csv": save_combined_csv,
            "save_pos_columns": True, "save_negabs_columns": True,
            "save_plot_signed": save_plot_signed,
            "save_plot_pos": save_plot_pos,
            "save_plot_negabs": save_plot_negabs,
            "save_summary_json": save_summary_json,
            "split_combined_csv": False,
            "save_batch_embedding_summary": False,
            "save_output_data_map": False, "output_data_map": {},
            "gradient_sample_mode": gradient_sample_mode, "eps": eps,
            "classification_extractions": None, "regression_extractions": None,
        }
        proxai.run_proxai(cfg, log=log)
        return json.dumps({"status": "ok", "out_root": out_root, "log": logs})
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e),
                           "traceback": traceback.format_exc(), "log": logs})


def build_tools():
    return [
        StructuredTool.from_function(
            func=proxai_discover_bins_tool,
            name="proxai_discover_bins",
            description="List available retention bins in a ProXAI CSV before running feature discovery.",
        ),
        StructuredTool.from_function(
            func=run_proxai_tool,
            name="run_proxai",
            description=(
                "Run neural-network gradient feature discovery across retention bins. "
                "Trains dual ensembles (Run A + B) and extracts m/z importance scores "
                "as signed, positive, and negative-abs gradient CSVs with mirror plots."
            ),
        ),
    ]
