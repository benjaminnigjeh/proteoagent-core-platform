#!/usr/bin/env python3
"""
raw_worker.py
Runs under the 'casting' conda env (Python 3.8 + fisher_py).
Called as a subprocess by databank_tools.py.
Reads a JSON job from stdin, writes a JSON result to stdout.
"""

import sys, json, os, re, glob, gc
import numpy as np

try:
    from fisher_py.data.business import Scan
    from fisher_py import RawFile
    HAS_FISHER = True
except ImportError:
    HAS_FISHER = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_type_label(text):
    m = re.search(r"Full\s+(\w+)", str(text), flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""


def _as_float_array(x):
    if x is None:
        return np.array([], dtype=float)
    a = np.asarray(x)
    return a.astype(float, copy=False) if a.size else np.array([], dtype=float)


def _sanitize_metadata_dict(md):
    safe = {}
    for k, v in md.items():
        if isinstance(v, (int, float, np.number, np.bool_)):
            safe[k] = np.array(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.dtype == object:
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    arr = arr.astype("U")
            if np.issubdtype(arr.dtype, np.character):
                arr = arr.astype("U")
            safe[k] = arr
        elif isinstance(v, str):
            safe[k] = np.array(v, dtype="U")
        else:
            safe[k] = np.array(str(v), dtype="U")
    return safe


def _out_paths(out_dir, group):
    base = os.path.join(os.path.abspath(out_dir), group)
    return f"{base}.ms1.npz", f"{base}.ms2.npz", f"{base}.meta.npz"


def _group_from_name(name, groups):
    for g in groups:
        if g in name:
            return g
    return "Unknown"


# ── Core RAW → NPZ processing ─────────────────────────────────────────────────

def process_group(job):
    """
    Process all .raw files for one sample group and save MS1/MS2/meta NPZ files.
    Mirrors _process_group from the original databank_gui.py exactly.
    """
    group       = job["group"]
    group_files = job["group_files"]
    out_dir     = job["out_dir"]
    ms1_mz_min  = job.get("ms1_mz_min", 600.0)
    ms1_mz_max  = job.get("ms1_mz_max", 1969.0)
    ms1_bin_da  = job.get("ms1_bin_da", 0.1)
    ms2_mz_min  = job.get("ms2_mz_min", 400.0)
    ms2_mz_max  = job.get("ms2_mz_max", 1999.0)
    ms2_bin_da  = job.get("ms2_bin_da", 1.0)
    logs = []
    log  = logs.append

    if not HAS_FISHER:
        return {"status": "error",
                "message": "fisher_py not available in this environment.",
                "log": logs}

    if not group_files:
        log(f"[{group}] No files — skipped.")
        return {"status": "ok", "group": group, "skipped": True, "log": logs}

    # Bin index parameters
    _ms1_pts     = int(round(1.0 / ms1_bin_da))
    _ms1_min_idx = int(round(ms1_mz_min * _ms1_pts))
    _ms1_len     = int(round((ms1_mz_max - ms1_mz_min) * _ms1_pts))
    _ms1_max_exc = _ms1_min_idx + _ms1_len
    _ms2_pts     = int(round(1.0 / ms2_bin_da))
    _ms2_min_idx = int(round(ms2_mz_min * _ms2_pts))
    _ms2_len     = int(round((ms2_mz_max - ms2_mz_min) * _ms2_pts)) + 1
    _ms2_max_exc = _ms2_min_idx + _ms2_len

    os.makedirs(out_dir, exist_ok=True)
    ms1_path, ms2_path, meta_path = _out_paths(out_dir, group)

    file_basenames, file_abspaths, file_to_id   = [], [], {}
    ms1_rows, ms1_scan, ms1_rt, ms1_file_id     = [], [], [], []
    ms2_rows, ms2_scan, ms2_rt, ms2_prec_mz, ms2_file_id = [], [], [], [], []

    for raw_abs in group_files:
        raw_name = os.path.basename(raw_abs)
        if raw_abs not in file_to_id:
            file_to_id[raw_abs] = len(file_basenames)
            file_basenames.append(raw_name)
            file_abspaths.append(raw_abs)
        f_id = file_to_id[raw_abs]

        try:
            raw = RawFile(raw_abs)
        except Exception as e:
            log(f"[skip] Cannot open {raw_name}: {e}")
            continue

        total_scans = int(getattr(raw, "number_of_scans", 0) or 0)
        log(f"[{group}] {raw_name}  ({total_scans:,} scans) ...")

        for i in range(1, total_scans + 1):
            try:
                raw_scan = Scan.from_file(raw._raw_file_access, scan_number=i)
            except Exception:
                continue

            stype  = _scan_type_label(raw_scan.scan_type)
            sc_num = getattr(raw_scan.scan_statistics, "scan_number", i)
            try:
                rt = float(raw.get_retention_time_from_scan_number(sc_num))
            except Exception:
                rt = float("nan")

            masses = _as_float_array(getattr(raw_scan, "preferred_masses", None))
            intens = _as_float_array(getattr(raw_scan, "preferred_intensities", None))
            if masses.size == 0 or intens.size == 0:
                continue

            if stype == "ms":
                idx  = np.rint(masses * _ms1_pts).astype(np.int32)
                mask = (idx >= _ms1_min_idx) & (idx < _ms1_max_exc)
                if not mask.any():
                    continue
                v32 = np.zeros(_ms1_len, dtype=np.float32)
                np.add.at(v32, idx[mask] - _ms1_min_idx,
                          intens[mask].astype(np.float32, copy=False))
                ms1_rows.append(v32)
                ms1_scan.append(sc_num)
                ms1_rt.append(rt)
                ms1_file_id.append(f_id)

            elif stype == "ms2":
                idx  = np.rint(masses * _ms2_pts).astype(np.int32)
                mask = (idx >= _ms2_min_idx) & (idx < _ms2_max_exc)
                if not mask.any():
                    continue
                v32 = np.zeros(_ms2_len, dtype=np.float32)
                np.add.at(v32, idx[mask] - _ms2_min_idx,
                          intens[mask].astype(np.float32, copy=False))
                vmax = float(v32.max())
                if vmax > 0:
                    v32 /= vmax
                vec  = v32.astype(np.float16, copy=False)
                prec = float("nan")
                for attr in ("precursor_mz", "master_precursor_mz", "isolation_mz"):
                    if hasattr(raw_scan, attr):
                        try:
                            prec = float(getattr(raw_scan, attr))
                            break
                        except Exception:
                            pass
                if prec != prec:  # still nan
                    m2   = re.findall(r'\d+\.\d+', str(raw_scan.scan_type))
                    prec = float(m2[1]) if len(m2) > 1 else float("nan")
                ms2_rows.append(vec)
                ms2_scan.append(sc_num)
                ms2_rt.append(rt)
                ms2_prec_mz.append(prec)
                ms2_file_id.append(f_id)

        try:
            raw.dispose()
        except Exception:
            pass

    # Build metadata and save NPZ files
    meta_raw = dict(
        group_name        = np.array(group, dtype="U"),
        ms1_scan          = np.asarray(ms1_scan,     dtype=np.int32),
        ms1_rt            = np.asarray(ms1_rt,       dtype=np.float32),
        ms1_file_id       = np.asarray(ms1_file_id,  dtype=np.int32),
        ms2_scan          = np.asarray(ms2_scan,     dtype=np.int32),
        ms2_rt            = np.asarray(ms2_rt,       dtype=np.float32),
        ms2_precursor_mz  = np.asarray(ms2_prec_mz,  dtype=np.float32),
        ms2_file_id       = np.asarray(ms2_file_id,  dtype=np.int32),
        file_names_lookup = np.asarray(file_basenames, dtype="U"),
        file_paths_lookup = np.asarray(file_abspaths,  dtype="U"),
    )
    metadata = _sanitize_metadata_dict(meta_raw)

    MS1 = (np.vstack(ms1_rows).astype(np.float32, copy=False) if ms1_rows
           else np.zeros((0, _ms1_len), dtype=np.float32))
    np.savez_compressed(ms1_path, ms1_matrix=MS1, **metadata)
    log(f"[{group}] MS1 saved → {ms1_path}  shape={MS1.shape}")
    del MS1, ms1_rows
    gc.collect()

    MS2 = (np.vstack(ms2_rows).astype(np.float16, copy=False) if ms2_rows
           else np.zeros((0, _ms2_len), dtype=np.float16))
    np.savez_compressed(ms2_path, ms2_matrix=MS2, **metadata)
    log(f"[{group}] MS2 saved → {ms2_path}  shape={MS2.shape}")
    del MS2, ms2_rows
    gc.collect()

    np.savez_compressed(meta_path, **metadata)
    log(f"[{group}] META saved → {meta_path}")
    del metadata, meta_raw
    gc.collect()

    return {
        "status": "ok",
        "group":  group,
        "ms1":    ms1_path,
        "ms2":    ms2_path,
        "meta":   meta_path,
        "log":    logs,
    }


# ── Top-level dispatcher ──────────────────────────────────────────────────────

def gather_and_dispatch(job):
    """
    Gather .raw files from folder_paths, assign to groups, process each group.
    Equivalent to wholeCasting_per_group in the original GUI.
    """
    folder_paths = job["folder_paths"]
    groups       = job["groups"]
    ignore_kw    = [k.lower() for k in job.get("ignore_keywords", [])]
    out_dir      = job["out_dir"]

    # Collect all .raw files
    raw_files = []
    for fp in folder_paths:
        fp = os.path.abspath(fp)
        if not os.path.isdir(fp):
            return {"status": "error",
                    "message": f"Folder not found: {fp}",
                    "log": []}
        raw_files.extend(glob.glob(os.path.join(fp, "*.raw")))
        raw_files.extend(glob.glob(os.path.join(fp, "*.RAW")))

    raw_files = sorted(set(os.path.abspath(p) for p in raw_files))

    if ignore_kw:
        raw_files = [p for p in raw_files
                     if not any(k in p.lower() for k in ignore_kw)]

    if not raw_files:
        return {"status": "error",
                "message": "No .raw files found after filtering.",
                "log": []}

    # Assign files to groups
    by_group = {g: [] for g in groups}
    for p in raw_files:
        g = _group_from_name(os.path.basename(p), groups)
        if g in by_group:
            by_group[g].append(p)

    outputs, all_logs = {}, []
    for g in groups:
        sub_job = {**job, "group": g, "group_files": by_group[g]}
        result  = process_group(sub_job)
        all_logs.extend(result.pop("log", []))
        outputs[g] = result

    return {"status": "ok", "outputs": outputs, "log": all_logs}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        job       = json.loads(raw_input)
        result    = gather_and_dispatch(job)
    except Exception as e:
        result = {"status": "error", "message": str(e), "log": []}

    print(json.dumps(result))
    sys.stdout.flush()
