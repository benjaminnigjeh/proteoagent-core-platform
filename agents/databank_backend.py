#!/usr/bin/env python3
"""
databank_backend.py
Pure backend logic for the Databank Generation Workflow.
No GUI dependencies. Runs in the claude_databank conda env (Python 3.11).
Steps 3-8 live here. Steps 1-2 are in raw_worker.py (casting env).
"""

import os, re, glob, gc
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — Steps 3-4: Combine MS2 → HDF5 / CSV
# ══════════════════════════════════════════════════════════════════════════════

def combine_ms2_npz_to_csv(npz_paths, csv_out_path, log=print, prog1=None, prog2=None):
    """Combine metadata from multiple MS2 NPZ files into a single CSV (scan index only)."""
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    frames = []
    _n = len(npz_paths)
    for _pi, p in enumerate(npz_paths):
        if prog2 is not None:
            prog2(_pi * 100 // max(_n, 1))
        z = np.load(p, allow_pickle=True)
        ms2_file_id = z["ms2_file_id"].astype(int)
        file_names  = z["file_names_lookup"]
        gn          = z["group_name"]
        group_name  = str(gn) if np.asarray(gn).ndim == 0 else str(np.asarray(gn).ravel()[0])
        meta = pd.DataFrame({
            "scan":         z["ms2_scan"],
            "rt_min":       z["ms2_rt"],
            "precursor_mz": z["ms2_precursor_mz"],
            "file_name":    file_names[ms2_file_id],
            "group_name":   group_name,
            "source_npz":   os.path.basename(p),
        })
        frames.append(meta)
    if prog2 is not None:
        prog2(100)
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(csv_out_path) or ".", exist_ok=True)
    combined.to_csv(csv_out_path, index=False)
    log(f"Saved {len(combined):,} MS2 scan records → {csv_out_path}")
    return combined


def combine_ms2_npz_to_hdf5(npz_paths, h5_out_path, log=print, prog1=None, prog2=None):
    """Combine MS2 NPZ files into a single HDF5 spectral library."""
    if not HAS_H5PY:
        raise ImportError("h5py required.")
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    matrices, frames = [], []
    _n = len(npz_paths)
    for _pi, p in enumerate(npz_paths):
        if prog2 is not None:
            prog2(_pi * 100 // max(_n, 1))
        z           = np.load(p, allow_pickle=True)
        ms2_matrix  = z["ms2_matrix"]
        ms2_file_id = z["ms2_file_id"].astype(int)
        file_names  = z["file_names_lookup"]
        gn          = z["group_name"]
        group_name  = str(gn) if np.asarray(gn).ndim == 0 else str(np.asarray(gn).ravel()[0])
        meta = pd.DataFrame({
            "scan":         z["ms2_scan"],
            "rt_min":       z["ms2_rt"],
            "precursor_mz": z["ms2_precursor_mz"],
            "file_name":    file_names[ms2_file_id],
            "group_name":   group_name,
        })
        matrices.append(ms2_matrix)
        frames.append(meta)
    if prog2 is not None:
        prog2(100)
    ms2_lib  = np.vstack(matrices)
    metadata = pd.concat(frames, ignore_index=True)
    with h5py.File(h5_out_path, "w") as f:
        f.create_dataset("ms2_lib", data=ms2_lib, compression="gzip")
        for col in metadata.columns:
            vals = metadata[col].values
            if metadata[col].dtype == object:
                vals = vals.astype("S")
            f.create_dataset(col, data=vals)
    log(f"Saved {ms2_lib.shape[0]:,} MS2 scans → {h5_out_path}")
    return ms2_lib, metadata


def load_ms2_hdf5(h5_path, log=print):
    if not HAS_H5PY or not HAS_PANDAS:
        raise ImportError("h5py and pandas required.")
    with h5py.File(h5_path, "r") as f:
        ms2_lib  = f["ms2_lib"][:]
        metadata = pd.DataFrame({col: f[col][:] for col in f if col != "ms2_lib"})
    for col in metadata.columns:
        if metadata[col].dtype == object or str(metadata[col].dtype).startswith("|S"):
            metadata[col] = metadata[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    log(f"Loaded {ms2_lib.shape[0]:,} MS2 scans from {h5_path}")
    return ms2_lib, metadata


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — Step 4b: RT Alignment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a, b):
    try:
        va = np.asarray(a, dtype=float).ravel()
        vb = np.asarray(b, dtype=float).ravel()
    except Exception:
        return -np.inf
    if va.size == 0 or vb.size == 0:
        return -np.inf
    n    = min(va.size, vb.size)
    va, vb = va[:n], vb[:n]
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else -np.inf


def _decode_df(df):
    for col in df.columns:
        dt = df[col].dtype
        if dt == object or str(dt).startswith("|S"):
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)


def _harmonize(df):
    if "sample_name" not in df.columns:
        for c in ("file_name", "raw_name", "run_name"):
            if c in df.columns:
                df["sample_name"] = df[c].astype(str)
                break
    if "m/z" not in df.columns:
        for c in ("mz", "precursor_mz"):
            if c in df.columns:
                df["m/z"] = df[c].astype(float)
                break
    if "retntion time" not in df.columns:
        for c in ("retention_time", "rt_min", "rt"):
            if c in df.columns:
                df["retntion time"] = df[c].astype(float)
                break
        else:
            if {"rt_min", "rt_max"}.issubset(df.columns):
                df["retntion time"] = (df["rt_min"] + df["rt_max"]) / 2.0


def _load_h5_df(h5_path):
    with h5py.File(h5_path, "r") as f:
        ms2_lib = f["ms2_lib"][:]
        meta    = {k: f[k][:] for k in f if k != "ms2_lib"}
    df = pd.DataFrame(meta)
    _decode_df(df)
    _harmonize(df)
    df["cast spectra"] = pd.Series(list(ms2_lib), index=df.index)
    return df


def _make_interp(xv, yv):
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    if xv.size == 0:
        return lambda rt: np.zeros_like(np.asarray(rt, float))
    if xv.size == 1:
        c = float(yv[0])
        return lambda rt, c=c: np.full_like(np.asarray(rt, float), c)
    def f(rt, x=xv, y=yv):
        return np.interp(np.asarray(rt, float), x, y, left=y[0], right=y[-1])
    return f


def _collect_drifts(bin_df, mz_ref, rt_ref, cast_ref, sim_thr, mz_win, target_n):
    if bin_df.empty:
        return []
    drifts = []
    for _, row in bin_df.sample(frac=1.0, random_state=42).reset_index(drop=True).iterrows():
        if len(drifts) >= target_n:
            break
        mz_i   = float(row["m/z"])
        rt_i   = float(row["retntion time"])
        cast_i = row["cast spectra"]
        idxs   = np.where(np.abs(mz_ref - mz_i) < mz_win)[0]
        if not idxs.size:
            continue
        matches = [rt_ref[j] for j in idxs if _cosine(cast_i, cast_ref[j]) > sim_thr]
        if matches:
            drifts.append(rt_i - float(np.mean(matches)))
    return drifts


def _drift_table(df_t, mz_ref, rt_ref, cast_ref,
                 rt_start=10.0, rt_end=80.0, bin_w=10.0, bin_step=5.0,
                 target_n=20, sim_thr=0.95, mz_win=1.0):
    df_t = df_t[(df_t["retntion time"] >= rt_start) & (df_t["retntion time"] < rt_end)].copy()
    if df_t.empty:
        return pd.DataFrame()
    bins, t = [], rt_start
    while t + bin_w <= rt_end + 1e-9:
        bins.append((t, t + bin_w))
        t += bin_step
    records = []
    for t0, t1 in bins:
        sub    = df_t[(df_t["retntion time"] >= t0) & (df_t["retntion time"] < t1)]
        drifts = _collect_drifts(sub, mz_ref, rt_ref, cast_ref, sim_thr, mz_win, target_n)
        records.append({
            "bin_start_min":  t0,
            "bin_end_min":    t1,
            "bin_center_min": 0.5 * (t0 + t1),
            "n_valid_used":   len(drifts),
            "avg_rt_drift":   float(np.mean(drifts)) if drifts else float("nan"),
        })
    return pd.DataFrame.from_records(records)


def get_sample_names_from_h5(h5_path):
    """Return sorted unique sample names from an HDF5 MS2 dataset."""
    if not HAS_H5PY:
        raise ImportError("h5py required.")
    with h5py.File(h5_path, "r") as f:
        for key in ("file_name", "sample_name", "group_name"):
            if key in f:
                raw   = f[key][:]
                names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                         for x in raw]
                return sorted(set(names))
    return []


def align_runs_from_h5(h5_path, ref_name=None,
                       rt_start=10.0, rt_end=80.0, bin_w=10.0, bin_step=5.0,
                       target_n=20, sim_thr=0.95, mz_win=1.0,
                       log=print, prog1=None, prog2=None):
    if not HAS_H5PY or not HAS_PANDAS:
        raise ImportError("h5py and pandas required.")
    df      = _load_h5_df(h5_path)
    samples = df["sample_name"].dropna().unique().tolist()
    log(f"Loaded {len(samples)} sample(s): {', '.join(samples)}")
    if len(samples) < 2:
        raise ValueError(f"Need ≥2 samples; found {len(samples)}: {samples}")
    if ref_name and ref_name in samples:
        ref = ref_name
    elif ref_name:
        raise ValueError(f"Reference run '{ref_name}' not found.\nAvailable: {samples}")
    else:
        ref = samples[0]
    log(f"Reference run: {ref}")
    df_ref   = df[df["sample_name"] == ref]
    df_ref   = df_ref[(df_ref["retntion time"] >= rt_start) & (df_ref["retntion time"] < rt_end)]
    mz_ref   = df_ref["m/z"].to_numpy()
    rt_ref   = df_ref["retntion time"].to_numpy()
    cast_ref = df_ref["cast spectra"].to_numpy(object)
    df       = df.copy()
    df["rt_correction"] = 0.0
    df["rt_aligned"]    = df["retntion time"].astype(float)
    all_drifts = []
    targets    = [s for s in samples if s != ref]
    for _si, tname in enumerate(targets):
        if prog1 is not None and targets:
            prog1(_si * 100 // len(targets))
        dft = df[df["sample_name"] == tname]
        if dft.empty:
            continue
        res = _drift_table(dft, mz_ref, rt_ref, cast_ref,
                           rt_start=rt_start, rt_end=rt_end,
                           bin_w=bin_w, bin_step=bin_step,
                           target_n=target_n, sim_thr=sim_thr, mz_win=mz_win)
        res = res.copy()
        res["target_name"] = tname
        all_drifts.append(res)
        valid = res.dropna(subset=["avg_rt_drift"])
        if not valid.empty:
            w = valid["n_valid_used"].to_numpy()
            v = valid["avg_rt_drift"].to_numpy()
            log(f"{tname}: weighted avg drift = {np.average(v, weights=w):.3f} min  "
                f"({len(valid)}/{len(res)} bins with matches)")
        else:
            log(f"{tname}: no valid drift matches found.")
        fn = (_make_interp(valid["bin_center_min"].to_numpy(), valid["avg_rt_drift"].to_numpy())
              if not valid.empty else lambda rt: np.zeros_like(np.asarray(rt, float)))
        rt_vals = dft["retntion time"].to_numpy(dtype=float)
        corr    = fn(rt_vals)
        df.loc[dft.index, "rt_correction"] = corr
        df.loc[dft.index, "rt_aligned"]    = rt_vals - corr
    if prog1 is not None:
        prog1(100)
    drift_table = pd.concat(all_drifts, ignore_index=True) if all_drifts else pd.DataFrame()
    return df, drift_table


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — Step 5: MS1 Binning
# ══════════════════════════════════════════════════════════════════════════════

def _decode_bytes_arr(a):
    if isinstance(a, np.ndarray) and (a.dtype.kind in ("S", "O")):
        return np.array(
            [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in a],
            dtype=object)
    return a


def _safe_metadata_npz(z, n_rows):
    cols = {}
    for k in z.files:
        if k == "ms1_matrix":
            continue
        arr = z[k]
        if k in ("file_names_lookup", "file_paths_lookup"):
            cols[k] = arr
            continue
        a = np.asarray(arr)
        if a.ndim == 0:
            cols[k] = np.repeat(a.item(), n_rows)
        elif a.ndim == 1:
            if a.shape[0] == n_rows:
                cols[k] = a
            elif a.shape[0] == 1:
                cols[k] = np.repeat(a[0], n_rows)
    df = pd.DataFrame({k: cols[k] for k in cols
                       if k not in ("file_names_lookup", "file_paths_lookup")})
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("|S"):
            df[c] = pd.Series([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
                                for x in df[c]])
    if "ms1_file_id" in df.columns and "file_names_lookup" in cols:
        fid  = df["ms1_file_id"].astype(int).to_numpy()
        lut  = np.asarray(_decode_bytes_arr(cols["file_names_lookup"]), dtype=object)
        ok   = (fid >= 0) & (fid < lut.shape[0])
        names = np.array([f"fid_{i}" for i in fid], dtype=object)
        names[ok] = lut[fid[ok]]
        df["sample_name"] = names.astype(str)
    elif "file_name" in df.columns:
        df["sample_name"] = df["file_name"].astype(str)
    else:
        df["sample_name"] = "UnknownRun"
    if "group_name" not in df.columns:
        df["group_name"] = "Unknown"
    if "retntion time" not in df.columns:
        for c in ("ms1_rt", "retention_time", "rt"):
            if c in df.columns:
                df["retntion time"] = df[c].astype(float)
                break
        else:
            if {"rt_min", "rt_max"}.issubset(df.columns):
                df["retntion time"] = (df["rt_min"].astype(float) + df["rt_max"].astype(float)) / 2
    return df


def _build_align_fns(drift_path):
    p = drift_path
    if not os.path.exists(p) and os.path.exists(p + ".csv"):
        p = p + ".csv"
    wide = pd.read_csv(p, index_col=0)
    try:
        bin_centers = [float(c) for c in wide.columns]
        ord_        = np.argsort(bin_centers)
        x_all       = np.array([bin_centers[i] for i in ord_], dtype=float)
        wide        = wide.iloc[:, ord_].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        fns = {str(run): _make_interp(x_all, row.to_numpy(dtype=float))
               for run, row in wide.iterrows()}
        return fns, lambda rt: np.zeros_like(np.asarray(rt, float))
    except Exception:
        pass
    long     = pd.read_csv(p)
    all_bins = np.sort(long["bin_center_min"].astype(float).unique())
    fns      = {}
    for run, grp in long.groupby("target_name"):
        y       = np.zeros_like(all_bins, dtype=float)
        idx_map = {bx: i for i, bx in enumerate(all_bins)}
        for xr, yr in zip(grp["bin_center_min"].astype(float), grp["avg_rt_drift"].astype(float)):
            i = idx_map.get(xr)
            if i is not None and np.isfinite(yr):
                y[i] = yr
        fns[str(run)] = _make_interp(all_bins, y)
    return fns, lambda rt: np.zeros_like(np.asarray(rt, float))


def _sum_rows(M, idxs, chunk=1024):
    if idxs.size == 0:
        return np.zeros(M.shape[1], dtype=np.float32)
    acc = np.zeros(M.shape[1], dtype=np.float64)
    for s in range(0, idxs.size, chunk):
        acc += M[idxs[s:s + chunk]].sum(axis=0, dtype=np.float64)
    return acc.astype(np.float32)


def bin_ms1_npz(npz_path, drift_path, out_csv,
                rt_start=10.0, rt_end=80.0, bin_w=10.0, bin_step=5.0,
                log=print, prog1=None, prog2=None):
    """Sum MS1 scans into sliding RT bins with drift correction."""
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    bins = []
    t    = rt_start
    while t + bin_w <= rt_end + 1e-9:
        bins.append((t, t + bin_w))
        t += bin_step
    log(f"RT bins: {len(bins)}  ({rt_start}–{rt_end} min, width={bin_w}, step={bin_step})")
    z      = np.load(npz_path, allow_pickle=True)
    MS1    = z["ms1_matrix"]
    N, L   = MS1.shape
    meta   = _safe_metadata_npz(z, N)
    align_fns, default_fn = _build_align_fns(drift_path)
    rt_raw = meta["retntion time"].to_numpy(dtype=float)
    runs   = meta["sample_name"].astype(str).to_numpy()
    groups = meta["group_name"].astype(str).to_numpy()
    rt_corr = np.zeros_like(rt_raw)
    for run in np.unique(runs):
        fn = align_fns.get(run, default_fn)
        m  = (runs == run)
        rt_corr[m] = fn(rt_raw[m])
    rt_aligned  = rt_raw - rt_corr
    cast_cols   = [f"cast_{i:05d}" for i in range(L)]
    rows        = []
    _unique_runs = np.unique(runs)
    _n_runs      = len(_unique_runs)
    for _ri, run in enumerate(_unique_runs):
        if prog2 is not None and _n_runs > 0:
            prog2(_ri * 100 // _n_runs)
        idx_run   = np.flatnonzero(runs == run)
        grp_vals  = np.unique(groups[idx_run])
        grp_label = grp_vals[0] if grp_vals.size else "Unknown"
        rt_run    = rt_aligned[idx_run]
        for t0, t1 in bins:
            mask    = (rt_run >= t0) & (rt_run < t1)
            idxs    = idx_run[mask]
            n_scans = int(idxs.size)
            vec     = _sum_rows(MS1, idxs) if n_scans else np.zeros(L, dtype=np.float32)
            rows.append([run, grp_label, t0, t1, n_scans] + vec.tolist())
    if prog2 is not None:
        prog2(100)
    out_df = pd.DataFrame(
        rows,
        columns=["sample_name", "group_name", "rt_start", "rt_end", "n_scans"] + cast_cols)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log(f"Saved {len(out_df)} rows → {out_csv}")
    return out_csv


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — Steps 6-8: TDPortal identifications
# ══════════════════════════════════════════════════════════════════════════════

def _parse_scans(value):
    return [int(x) for x in re.findall(r"\d+", str(value))]


def add_tdportal_ids(tdportal_df, databank_df, output_csv, log=print, prog1=None, prog2=None):
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    lookup = {}
    for td_idx, row in tdportal_df.iterrows():
        for scan in _parse_scans(row["Fragment Scans"]):
            lookup[(str(row["File Name"]), int(scan))] = td_idx
    seq, mass, acc, pfr = [], [], [], []
    _n_db = len(databank_df)
    for _di, (_, row) in enumerate(databank_df.iterrows()):
        if prog2 is not None and _n_db > 0:
            prog2(_di * 100 // _n_db)
        i = lookup.get((str(row["sample_name"]), int(row["scan"])))
        if i is None:
            seq.append(None); mass.append(None); acc.append(None); pfr.append(None)
        else:
            seq.append(tdportal_df.at[i, "Sequence"])
            mass.append(tdportal_df.at[i, "Average Mass"])
            acc.append(tdportal_df.at[i, "Accession"])
            pfr.append(tdportal_df.at[i, "PFR"])
    if prog2 is not None:
        prog2(100)
    out = databank_df.copy()
    out["sequence"] = seq
    out["MASS"]     = mass
    out["Accession"] = acc
    out["PFR"]      = pfr
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    log(f"Saved databank with TDPortal IDs → {output_csv}")
    return out


def combine_tdportal_reports(csv_paths, output_path, log=print):
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    csvs = sorted(csv_paths)
    if not csvs:
        raise FileNotFoundError("No CSV files provided.")
    combined = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.to_csv(output_path, index=False)
    log(f"Combined {len(csvs)} report(s) → {output_path}")
    return combined


def remove_missing_pfr(input_csv, output_csv, log=print):
    if not HAS_PANDAS:
        raise ImportError("pandas required.")
    df    = pd.read_csv(input_csv)
    clean = df.dropna(subset=["PFR"])
    clean = clean[clean["PFR"].astype(str).str.strip() != ""]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    clean.to_csv(output_csv, index=False)
    log(f"Removed {len(df) - len(clean)} rows with missing PFR → {output_csv}")
    return clean
