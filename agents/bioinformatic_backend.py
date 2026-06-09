#!/usr/bin/env python3
"""
bioinformatic_backend.py
Pure backend for the Bioinformatics Module.
No GUI dependencies. Runs in the claude_databank conda env (Python 3.11).
Extracted from bioinformatic_gui.py.

Modules:
  1. Annotations  — signal detection + charge-series assignment
  2. Quantification
  3. Identification — fast NumPy databank search
"""

from __future__ import annotations

import ast, re, json, os, glob, warnings, collections
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

warnings.filterwarnings("ignore")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PROTON_MASS = 1.007276466812
Z_MIN = 5
Z_MAX = 50
PPM_TOL = 1000.0
ABS_DA_TOL = 1.0
MIN_MATCHED_CHARGE_STATES = 4


@dataclass
class PeakFindingParams:
    min_prominence: Optional[float] = None
    min_height: Optional[float] = None
    min_distance_pts: int = 20
    smooth_window: int = 0
    min_snr: float = 10.0


DECONV_DETECT_PARAMS = PeakFindingParams(min_distance_pts=20, min_snr=10, smooth_window=0)


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mad_sigma(y):
    if not HAS_NP or len(y) == 0: return 1.0
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return 1.4826 * mad

def _smooth(y, window):
    if not HAS_NP or window < 3 or window % 2 == 0: return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")

def _extract_id_list(name, key):
    m = re.search(rf"(?:^|[_-]){key}((?:[_-]\d+)+)(?=[_-]|$)", name, flags=re.I)
    if not m:
        m1 = re.search(rf"(?:^|[_-]){key}[_-]?(\d+)(?=[_-]|$)", name, flags=re.I)
        if m1: return [int(m1.group(1))]
        return None
    parts = re.findall(r"\d+", m.group(1))
    return [int(x) for x in parts] if parts else None

def parse_metadata_from_filename(path):
    p = Path(path)
    name = p.stem
    meta = {"bin": None, "experiments": None, "controls": None,
            "experiments_ids": None, "controls_ids": None,
            "experiments_n": None, "controls_n": None,
            "regulation": None, "replicate": None, "source_file": p.name}
    m = re.search(r"(?:^|[_-])bin[_-]?(\d+)(?=[_-]|$)", name, flags=re.I)
    if m:
        meta["bin"] = int(m.group(1))
    else:
        m2 = re.match(r"^(\d+)(?=[_-])", name)
        if m2: meta["bin"] = int(m2.group(1))
    exp_ids = _extract_id_list(name, "pos")
    ctl_ids = _extract_id_list(name, "neg")
    if exp_ids is not None:
        meta["experiments_ids"] = ",".join(str(x) for x in exp_ids)
        meta["experiments_n"] = len(exp_ids)
        meta["experiments"] = len(exp_ids)
    if ctl_ids is not None:
        meta["controls_ids"] = ",".join(str(x) for x in ctl_ids)
        meta["controls_n"] = len(ctl_ids)
        meta["controls"] = len(ctl_ids)
    reg_tokens = [m.group(1).lower() for m in re.finditer(
        r"(?:^|[_-])(negabs|posabs|neg|pos)(?=[_-]|$)", name, flags=re.I)]
    if reg_tokens:
        reg_map = {"negabs": "downregulated", "neg": "downregulated",
                   "posabs": "upregulated", "pos": "upregulated"}
        meta["regulation"] = reg_map.get(reg_tokens[-1])
    m = re.search(r"(?:^|[_-])run([A-Za-z])(?=[_-]|$)", name)
    if m: meta["replicate"] = m.group(1).upper()
    return meta

def load_deconv_file(path):
    if not HAS_PANDAS: raise RuntimeError("pandas is required")
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=None,
                         names=["mass", "intensity"], comment="#")
    except Exception:
        df = pd.read_csv(path, header=None)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]; df.columns = ["mass", "intensity"]
        else:
            raise ValueError(f"Deconvoluted file must have at least two columns: {path}")
    df["mass"] = pd.to_numeric(df["mass"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mass", "intensity"])
    return df.sort_values("mass").reset_index(drop=True)

def detect_signals(df, params=None):
    if not HAS_NP or not HAS_SCIPY: raise RuntimeError("numpy and scipy are required")
    if params is None: params = DECONV_DETECT_PARAMS
    if not {"mass", "intensity"}.issubset(df.columns):
        if df.shape[1] >= 2:
            df = df.copy(); df.columns = ["mass", "intensity"] + [f"col{i}" for i in range(2, df.shape[1])]
        else:
            raise ValueError("Input DataFrame must have columns ['mass', 'intensity'].")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mass", "intensity"])
    df = df.sort_values("mass").reset_index(drop=True)
    x = df["mass"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    y_proc = _smooth(y, params.smooth_window)
    sigma = _mad_sigma(y_proc)
    ymax = float(np.max(y_proc)) if y_proc.size else 0.0
    min_prom = params.min_prominence or max(6.0 * sigma, 0.001 * ymax)
    min_h = params.min_height or max(4.0 * sigma, 0.0005 * ymax)
    peaks, props = find_peaks(y_proc, prominence=min_prom, height=min_h,
                               distance=max(1, int(params.min_distance_pts)))
    out = pd.DataFrame({
        "mass": x[peaks], "intensity": y[peaks],
        "prominence": props.get("prominences", np.full(peaks.shape, np.nan)),
        "left_base_idx": props.get("left_bases", np.full(peaks.shape, -1)),
        "right_base_idx": props.get("right_bases", np.full(peaks.shape, -1)),
    })
    snr_den = sigma if sigma > 0 else (np.std(y_proc) if y_proc.size else 1.0)
    snr_den = snr_den if snr_den > 0 else 1.0
    out["snr"] = out["intensity"] / snr_den
    if params.min_snr > 0:
        out = out[out["snr"] >= params.min_snr].reset_index(drop=True)
    return out.sort_values("intensity", ascending=False).reset_index(drop=True)

def read_raw_ms1(path):
    if not HAS_PANDAS: raise RuntimeError("pandas is required")
    path = Path(path)
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        df.columns = ["mz", "intensity"]
    else:
        cols_lower = [str(c).lower() for c in df.columns]
        mz_cands = [i for i, c in enumerate(cols_lower)
                    if ("mz" in c or "m/z" in c or "mass/charge" in c or c.strip() == "m z")]
        int_cands = [i for i, c in enumerate(cols_lower)
                     if ("int" in c or "abund" in c or "height" in c or "signal" in c)]
        if not mz_cands: mz_cands = [0]
        if not int_cands: int_cands = [1 if df.shape[1] > 1 else 0]
        df = df.iloc[:, [mz_cands[0], int_cands[0]]].copy()
        df.columns = ["mz", "intensity"]
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mz", "intensity"])
    df = df[df["intensity"] > 0].copy()
    return df.sort_values("mz").reset_index(drop=True)

def _ppm_window(target_mz, ppm, abs_da):
    da = target_mz * ppm * 1e-6
    tol = max(da, abs_da)
    return target_mz - tol, target_mz + tol

def _match_targets(sorted_mz, targets, ppm, abs_da, available_mask):
    results = {}
    for ti, t in enumerate(targets):
        lo, hi = _ppm_window(t, ppm, abs_da)
        j = bisect_left(sorted_mz, t)
        best_idx, best_delta = None, float("inf")
        for k in (j, j-1, j+1, j-2, j+2, j-3, j+3):
            if 0 <= k < len(sorted_mz):
                mz_k = sorted_mz[k]
                if available_mask[k] and lo <= mz_k <= hi:
                    delta = abs(mz_k - t)
                    if delta < best_delta:
                        best_delta = delta; best_idx = k
        results[ti] = best_idx
    return results

def _generate_charge_series(neutral_mass, z_min, z_max):
    z = np.arange(z_min, z_max + 1, dtype=int)
    mz = (neutral_mass + z * PROTON_MASS) / z
    return pd.DataFrame({"z": z, "target_mz": mz})

def assign_ms1_peaks(raw_df, deconv_peaks_df, meta=None,
                     z_min=Z_MIN, z_max=Z_MAX,
                     ppm_tol=PPM_TOL, abs_da_tol=ABS_DA_TOL,
                     min_matched=MIN_MATCHED_CHARGE_STATES):
    if not HAS_NP or not HAS_PANDAS: raise RuntimeError("numpy and pandas are required")
    raw_df = raw_df.sort_values("mz").reset_index(drop=True)
    mz_arr = raw_df["mz"].to_numpy()
    inten_arr = raw_df["intensity"].to_numpy()
    available = np.ones(len(raw_df), dtype=bool)
    summary_rows = []
    for r in deconv_peaks_df.itertuples(index=False):
        mass = float(r.mass)
        series = _generate_charge_series(mass, z_min, z_max)
        matches = _match_targets(mz_arr, series["target_mz"].to_numpy(), ppm_tol, abs_da_tol, available)
        matched_indices, matched_z, matched_mz, matched_inten = [], [], [], []
        for ti, k in matches.items():
            if k is not None:
                matched_indices.append(k)
                matched_z.append(int(series.iloc[ti]["z"]))
                matched_mz.append(float(mz_arr[k]))
                matched_inten.append(float(inten_arr[k]))
        if len(matched_indices) >= min_matched:
            for idx in matched_indices:
                if available[idx]: available[idx] = False
            total_raw_intensity = float(np.sum(inten_arr))
            matched_intensity_sum = float(np.sum(inten_arr[matched_indices]))
            row = {
                "neutral_mass": mass,
                "deconv_intensity": float(r.intensity),
                "snr": float(getattr(r, "snr", np.nan)),
                "n_matches": len(matched_indices),
                "matched_z_list": json.dumps(matched_z),
                "matched_mz_list": json.dumps([round(float(x), 4) for x in matched_mz]),
                "matched_intensity_sum": matched_intensity_sum,
                "ppm_tol": ppm_tol, "abs_da_tol": abs_da_tol,
                "z_min": z_min, "z_max": z_max,
                "min_matched_charge_states": min_matched,
                "fraction_total_intensity_captured": (
                    matched_intensity_sum / total_raw_intensity if total_raw_intensity > 0 else 0.0),
            }
            if meta:
                row.update({k: meta.get(k) for k in
                             ["bin","experiments_ids","controls_ids","experiments_n",
                              "controls_n","regulation","replicate","source_file"]})
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("deconv_intensity", ascending=False).reset_index(drop=True)
    return summary

def process_one_pair(deconv_path, raw_path, out_dir, params=None,
                     z_min=Z_MIN, z_max=Z_MAX, ppm_tol=PPM_TOL,
                     abs_da_tol=ABS_DA_TOL, min_matched=MIN_MATCHED_CHARGE_STATES, log_fn=print):
    if params is None: params = DECONV_DETECT_PARAMS
    deconv_path = Path(deconv_path)
    base = deconv_path.stem
    if base.endswith("_mass"): base = base[:-5]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"\n=== Processing: {base} ===")
    if raw_path is None or not Path(raw_path).exists():
        log_fn("WARNING: No matching raw MS1 CSV found. Skipping."); return None
    meta = parse_metadata_from_filename(deconv_path)
    deconv_raw = load_deconv_file(deconv_path)
    deconv_peaks = detect_signals(deconv_raw, params=params)
    if deconv_peaks.empty:
        log_fn("WARNING: No neutral-mass peaks detected. Skipping."); return None
    raw_df = read_raw_ms1(raw_path)
    summary = assign_ms1_peaks(raw_df, deconv_peaks, meta=meta,
                                z_min=z_min, z_max=z_max, ppm_tol=ppm_tol,
                                abs_da_tol=abs_da_tol, min_matched=min_matched)
    if summary.empty:
        log_fn("WARNING: No accepted assignments found."); return None
    summary.insert(0, "base_name", base)
    summary.insert(1, "raw_file", Path(raw_path).name)
    summary.insert(2, "deconv_file", deconv_path.name)
    out_summary = out_dir / f"{base}_assignments_summary.csv"
    summary.to_csv(out_summary, index=False)
    log_fn(f"Raw MS1 rows: {len(raw_df):,}")
    log_fn(f"Detected neutral-mass peaks: {len(deconv_peaks):,}")
    log_fn(f"Accepted assigned neutral masses: {len(summary):,}")
    log_fn(f"Saved summary: {out_summary}")
    return str(out_summary)

def combine_assignment_summaries(out_dir, final_report_name="combined_assignments_report.csv", log_fn=print):
    out_dir = Path(out_dir)
    summary_files = sorted(out_dir.glob("*_assignments_summary.csv"))
    summary_files = [f for f in summary_files if f.name != final_report_name]
    if not summary_files:
        log_fn("WARNING: No *_assignments_summary.csv files found to combine."); return None
    df_list = [pd.read_csv(f) for f in summary_files]
    for f in summary_files: log_fn(f"Adding: {f.name}")
    combined_df = pd.concat(df_list, ignore_index=True)
    final_report = out_dir / final_report_name
    combined_df.to_csv(final_report, index=False)
    log_fn(f"Final combined report: {final_report}  ({len(combined_df)} rows)")
    return str(final_report)


# ══════════════════════════════════════════════════════════════════════════════
# Module 2: Quantification
# ══════════════════════════════════════════════════════════════════════════════

def to_cast_col(mz):
    col_num = int((float(mz) - 600.0) * 10.0)
    return "cast_" + str(col_num).zfill(5)

def parse_mz_list(val):
    try:
        out = ast.literal_eval(str(val))
        if isinstance(out, (list, tuple)):
            return [float(x) for x in out]
    except Exception:
        pass
    return []

def _linreg(x, y):
    n = len(x)
    if n < 2: return np.nan, np.nan
    xm, ym = np.mean(x), np.mean(y)
    ss_xx = np.sum((x - xm) ** 2)
    if ss_xx == 0: return np.nan, np.nan
    ss_xy = np.sum((x - xm) * (y - ym))
    ss_yy = np.sum((y - ym) ** 2)
    slope = ss_xy / ss_xx
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else np.nan
    return float(slope), float(r2)

def _is_continuous_target(target_series):
    vals = target_series.dropna().unique()
    if len(vals) > 10: return True
    return any(float(v) != float(int(v)) for v in vals)

def run_quantification(dataset_path: str, assignments_path: str, out_path: str, log_fn=print):
    """Quantify proteoforms using cast_* columns from the training dataset."""
    if not HAS_PANDAS or not HAS_NP: raise RuntimeError("numpy and pandas are required")
    log_fn(f"Loading dataset: {dataset_path}")
    df_rt = pd.read_csv(dataset_path)
    log_fn(f"  shape: {df_rt.shape}")
    log_fn(f"Loading assignments: {assignments_path}")
    df_asn = pd.read_csv(assignments_path)
    log_fn(f"  shape: {df_asn.shape}")
    if "bin" not in df_rt.columns: raise KeyError("'bin' column required in dataset CSV")
    if "target" not in df_rt.columns: raise KeyError("'target' column required in dataset CSV")
    if "bin" not in df_asn.columns or "matched_mz_list" not in df_asn.columns:
        raise KeyError("assignments CSV must contain 'bin' and 'matched_mz_list' columns")
    df_rt["bin"] = pd.to_numeric(df_rt["bin"], errors="coerce")
    df_rt["target"] = pd.to_numeric(df_rt["target"], errors="coerce")
    df_asn["bin"] = pd.to_numeric(df_asn["bin"], errors="coerce")
    continuous = _is_continuous_target(df_rt["target"])
    if continuous:
        log_fn("  Target mode: CONTINUOUS — outputs regression_slope, regression_r2")
        result_cols = ["regression_slope", "regression_r2"]
    else:
        targets = sorted(int(t) for t in df_rt["target"].dropna().unique())
        group_cols = [f"group_{t}_sum" for t in targets]
        log_fn(f"  Target mode: DISCRETE — groups: {targets}")
        result_cols = group_cols
    fixed_cols = result_cols + ["n_mz_used", "n_mz_found", "missing_cast_columns"]
    for c in fixed_cols:
        if c in df_asn.columns: df_asn.drop(columns=[c], inplace=True)
    rows_out = []
    n = len(df_asn)
    for i, (_, row) in enumerate(df_asn.iterrows()):
        bin_value = row["bin"]
        mz_list = parse_mz_list(row["matched_mz_list"])
        cast_cols = [to_cast_col(mz) for mz in mz_list]
        df_bin = df_rt[df_rt["bin"] == bin_value]
        res = {col: np.nan for col in result_cols}
        res.update({"n_mz_used": len(cast_cols), "n_mz_found": 0, "missing_cast_columns": ""})
        if df_bin.empty:
            res["missing_cast_columns"] = ", ".join(cast_cols) if cast_cols else ""
            rows_out.append(res); continue
        existing_cast = [c for c in cast_cols if c in df_bin.columns]
        missing_cast = [c for c in cast_cols if c not in df_bin.columns]
        res["n_mz_found"] = len(existing_cast)
        res["missing_cast_columns"] = ", ".join(missing_cast) if missing_cast else ""
        if existing_cast:
            if continuous:
                y = df_bin[existing_cast].sum(axis=1).to_numpy(float)
                x = df_bin["target"].to_numpy(float)
                mask = ~(np.isnan(x) | np.isnan(y))
                slope, r2 = _linreg(x[mask], y[mask])
                res["regression_slope"] = slope; res["regression_r2"] = r2
            else:
                grouped = df_bin.groupby("target")[existing_cast].sum(min_count=1)
                total_per_target = grouped.sum(axis=1)
                for t in targets:
                    res[f"group_{t}_sum"] = float(total_per_target.get(t, np.nan))
        rows_out.append(res)
        if (i + 1) % 500 == 0: log_fn(f"  Processed {i+1}/{n} rows...")
    df_quant = pd.DataFrame(rows_out, index=df_asn.index)
    df_out = pd.concat([df_asn, df_quant], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log_fn(f"Saved quantification: {out_path}  ({len(df_out)} rows)")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Module 3: Identification
# ══════════════════════════════════════════════════════════════════════════════

def _num(s): return pd.to_numeric(s, errors="coerce")

def _to_scalar(x):
    if isinstance(x, np.ndarray) and x.ndim == 0: x = x.item()
    if isinstance(x, str):
        try: return float(x)
        except Exception: return x
    return x

def _is_null_like(v):
    if v is None: return True
    if isinstance(v, float) and np.isnan(v): return True
    return False

def _mode_and_count(items):
    clean = [v for v in items if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean: return None, 0
    val, cnt = collections.Counter(clean).most_common(1)[0]
    return val, int(cnt)

def _mode_or_none(items):
    val, _ = _mode_and_count(items); return val

def _mode_and_count_with_cutoff(pfrs, accs, min_count):
    pfr_val, pfr_cnt = _mode_and_count(pfrs)
    acc_val = _mode_or_none(accs)
    if pfr_cnt < min_count: return None, 0, None
    return pfr_val, pfr_cnt, acc_val

def prepare_databank_arrays(df2):
    return {
        "rt": df2["rt_aligned"].to_numpy(dtype=float),
        "mz": df2["precursor_mz"].to_numpy(dtype=float),
        "mass": df2["MASS"].to_numpy(dtype=float),
        "accession": df2["Accession"].to_numpy(dtype=object),
        "pfr": df2["PFR"].to_numpy(dtype=object),
    }

def search_best_fast(db, rt_query, mz_query, mass_query, rt_window, mz_tol, mass_tol):
    d_rt = np.abs(db["rt"] - rt_query)
    rt_mask = d_rt <= rt_window
    if not np.any(rt_mask): return None
    idx_rt = np.where(rt_mask)[0]
    d_mz = np.abs(db["mz"][idx_rt] - mz_query)
    mz_mask = d_mz <= mz_tol
    if not np.any(mz_mask): return None
    idx_mz = idx_rt[mz_mask]
    d_mass = np.abs(db["mass"][idx_mz] - mass_query)
    mass_mask = d_mass <= mass_tol
    if not np.any(mass_mask): return None
    idx_final = idx_mz[mass_mask]
    score = (np.abs(db["rt"][idx_final] - rt_query) / max(rt_window, 1e-12)
             + np.abs(db["mz"][idx_final] - mz_query) / max(mz_tol, 1e-12)
             + np.abs(db["mass"][idx_final] - mass_query) / max(mass_tol, 1e-12))
    best_idx = idx_final[np.argmin(score)]
    return {"Accession": db["accession"][best_idx],
            "MASS": db["mass"][best_idx],
            "PFR": db["pfr"][best_idx]}

def _collect_matches_for_row_fast(row, db, rt_window, mz_tol, mass_tol):
    neutral_mass = row.get("neutral_mass", np.nan)
    retention_time = row.get("bin", np.nan)
    mz_list = parse_mz_list(row.get("matched_mz_list", []))
    if pd.isna(neutral_mass) or pd.isna(retention_time) or not mz_list:
        return [], [], [], []
    tokens, pfrs, accs, masses = [], [], [], []
    neutral_mass = float(neutral_mass)
    retention_time = float(retention_time)
    for mz_value in mz_list:
        mz_value = float(mz_value)
        res = search_best_fast(db=db, rt_query=retention_time, mz_query=mz_value,
                               mass_query=neutral_mass, rt_window=rt_window,
                               mz_tol=mz_tol, mass_tol=mass_tol)
        if res is not None:
            accession = res.get("Accession", "NA")
            mass_match = _to_scalar(res.get("MASS", neutral_mass))
            pfr_val = _to_scalar(res.get("PFR", None))
            if _is_null_like(pfr_val):
                tokens.append(f"{mz_value}: {accession}, {mass_match}")
                pfrs.append(None)
            else:
                tokens.append(f"{mz_value}: {accession}, {mass_match}, {pfr_val}")
                pfrs.append(pfr_val)
            accs.append(accession); masses.append(mass_match)
        else:
            tokens.append(f"{mz_value}: NA")
            pfrs.append(None); accs.append(None); masses.append(None)
    return tokens, pfrs, accs, masses

def _fmt_suffix(v):
    return f"{int(v)}" if float(v).is_integer() else str(v).replace(".", "p")

def _matched_pfr_from_list(pfrs, keep_placeholders):
    if not pfrs: return None
    if keep_placeholders:
        return "[" + ", ".join("null" if v is None else str(v) for v in pfrs) + "]"
    pruned = [str(v) for v in pfrs if v is not None]
    return "[" + ", ".join(pruned) + "]" if pruned else None

def run_identification(charge_file: str, databank_path: str, output: str,
                       param_sets=None, keep_placeholders: bool = True,
                       min_mode_pfr_count: int = 1, log_fn=print):
    """Fast NumPy databank search to match charge assignments to protein identifications."""
    if not HAS_PANDAS or not HAS_NP: raise RuntimeError("numpy and pandas are required")
    if param_sets is None or not param_sets:
        param_sets = [(55.0, 2.0, 90.0)]
    log_fn(f"Loading charge file: {charge_file}")
    df1 = pd.read_csv(charge_file)
    df1.columns = [c.strip() for c in df1.columns]
    log_fn(f"  shape: {df1.shape}")
    if "bin" not in df1.columns and "bin " in df1.columns:
        df1["bin"] = df1["bin "]
    if "bin" not in df1.columns:
        raise KeyError("charge file must contain a 'bin' column")
    if "neutral_mass" not in df1.columns or "matched_mz_list" not in df1.columns:
        raise KeyError("charge file must contain 'neutral_mass' and 'matched_mz_list' columns")
    log_fn(f"Loading databank: {databank_path}")
    df2 = pd.read_csv(databank_path)
    df2.columns = [c.strip() for c in df2.columns]
    log_fn(f"  shape: {df2.shape}")
    required_db_cols = ["rt_aligned", "precursor_mz", "MASS", "Accession", "PFR"]
    missing = [c for c in required_db_cols if c not in df2.columns]
    if missing: raise KeyError(f"Databank CSV missing required column(s): {missing}")
    df2 = df2.copy()
    df2["rt_aligned"] = _num(df2["rt_aligned"])
    df2["precursor_mz"] = _num(df2["precursor_mz"])
    df2["MASS"] = _num(df2["MASS"])
    before = len(df2)
    df2 = df2.dropna(subset=["rt_aligned", "precursor_mz", "MASS"]).reset_index(drop=True)
    after = len(df2)
    if after < before: log_fn(f"Removed {before - after} databank rows with invalid numeric values.")
    log_fn("Preparing fast NumPy databank arrays...")
    db = prepare_databank_arrays(df2)
    n_rows = len(df1)
    for param_index, (rt_w, mz_t, mass_t) in enumerate(param_sets, start=1):
        suffix = f"rt{_fmt_suffix(rt_w)}_mz{_fmt_suffix(mz_t)}_mass{_fmt_suffix(mass_t)}"
        log_fn(f"\nParameter set {param_index}/{len(param_sets)}: RT={rt_w}, mz={mz_t}, mass={mass_t}")
        match_col   = f"best_match_{suffix}"
        pfr_col     = f"matched_pfr_{suffix}"
        mode_pfr    = f"mode_pfr_{suffix}"
        mode_pfr_n  = f"mode_pfr_count_{suffix}"
        mode_acc    = f"mode_accession_{suffix}"
        best_tokens_series, pfr_list_series, acc_list_series = [], [], []
        for i, (_, row) in enumerate(df1.iterrows()):
            tokens, pfrs, accs, _masses = _collect_matches_for_row_fast(
                row=row, db=db, rt_window=rt_w, mz_tol=mz_t, mass_tol=mass_t)
            best_tokens_series.append(tokens)
            pfr_list_series.append(pfrs)
            acc_list_series.append(accs)
            if (i + 1) % 1000 == 0: log_fn(f"  Searched {i+1}/{n_rows} rows...")
        df1[match_col] = ["[" + ", ".join(toks) + "]" if toks else None for toks in best_tokens_series]
        df1[pfr_col] = [_matched_pfr_from_list(pfrs, keep_placeholders) for pfrs in pfr_list_series]
        mode_results = [_mode_and_count_with_cutoff(pfrs, accs, min_mode_pfr_count)
                        for pfrs, accs in zip(pfr_list_series, acc_list_series)]
        df1[mode_pfr]   = [mr[0] for mr in mode_results]
        df1[mode_pfr_n] = [mr[1] for mr in mode_results]
        df1[mode_acc]   = [mr[2] for mr in mode_results]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df1.to_csv(output, index=False)
    log_fn(f"\nSaved identification results: {output}  ({len(df1)} rows)")
    return df1


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points for the agent
# ══════════════════════════════════════════════════════════════════════════════

def run_annotations(deconv_folder: str, raw_ms1_folder: str, out_dir: str,
                    z_min: int = Z_MIN, z_max: int = Z_MAX,
                    ppm_tol: float = PPM_TOL, abs_da_tol: float = ABS_DA_TOL,
                    min_matched: int = MIN_MATCHED_CHARGE_STATES,
                    log_fn=print) -> str:
    """
    Run signal detection + charge-series assignment for all deconvoluted files
    paired with raw MS1 CSVs.

    deconv_folder : folder with deconvolution output files (*_mass.txt or *.txt)
    raw_ms1_folder: folder with raw MS1 CSV files
    out_dir       : output directory
    Returns path to the combined assignments report CSV.
    """
    deconv_files = sorted(
        glob.glob(os.path.join(deconv_folder, "*_mass.txt")) +
        glob.glob(os.path.join(deconv_folder, "*.txt"))
    )
    deconv_files = list(dict.fromkeys(deconv_files))  # deduplicate

    if not deconv_files:
        log_fn(f"WARNING: No deconvolution files found in {deconv_folder}")
        return ""

    raw_files = sorted(
        glob.glob(os.path.join(raw_ms1_folder, "*.csv")) +
        glob.glob(os.path.join(raw_ms1_folder, "*.txt"))
    )

    def _best_raw_match(deconv_path):
        base = Path(deconv_path).stem
        if base.endswith("_mass"): base = base[:-5]
        for r in raw_files:
            if Path(r).stem == base: return r
        for r in raw_files:
            if base[:20] in Path(r).stem: return r
        return raw_files[0] if raw_files else None

    for deconv_path in deconv_files:
        raw_path = _best_raw_match(deconv_path)
        process_one_pair(deconv_path, raw_path, out_dir,
                         z_min=z_min, z_max=z_max,
                         ppm_tol=ppm_tol, abs_da_tol=abs_da_tol,
                         min_matched=min_matched, log_fn=log_fn)

    return combine_assignment_summaries(out_dir, log_fn=log_fn) or ""
