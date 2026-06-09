#!/usr/bin/env python3
"""
deconvolution_backend.py
Pure backend for Proteoform Deconvolution.
No GUI dependencies. Runs in the claude_databank conda env (Python 3.11).
Extracted from deconvolution_gui.py.
"""

import os, sys, shutil, subprocess, glob, warnings
from dataclasses import dataclass, field
from typing import Optional, List

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
    from scipy.optimize import nnls as scipy_nnls
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeconvConfig:
    input_csv: str = ""
    out_dir: Optional[str] = None
    mz_col: str = "m/z"
    intensity_col: Optional[str] = None
    proton_mass: float = 1.007276466812
    min_charge: int = 2
    max_charge: int = 30
    min_mass: float = 5000.0
    max_mass: float = 100000.0
    mass_bin_size: float = 1.0
    peak_height_frac: float = 0.001
    peak_prominence_frac: float = 0.005
    smooth_sigma: float = 1.0
    consensus_ppm: float = 30.0
    unidec_exe: Optional[str] = None
    flashdeconv_exe: Optional[str] = None
    topfd_exe: Optional[str] = None
    make_plots: bool = True
    enabled_algorithms: Optional[List[str]] = None


# ══════════════════════════════════════════════════════════════════════════════
# Deconvolver
# ══════════════════════════════════════════════════════════════════════════════

class ProteoformDeconvolver:
    ALGORITHM_KEYS = [
        "UniDec", "ms_deisotope", "pyOpenMS", "FLASHDeconv",
        "TopFD", "Sparse_NNLS", "Peak_Charge_Collapse",
    ]

    def __init__(self, cfg: DeconvConfig, log_fn=print, stop_event=None, progress_cb=None):
        self.cfg = cfg
        self._log_fn = log_fn
        self._stop = stop_event
        self._prog = progress_cb
        self._enabled = set(cfg.enabled_algorithms) if cfg.enabled_algorithms else set(self.ALGORITHM_KEYS)

        self.input_csv = os.path.abspath(cfg.input_csv)
        self.base = os.path.splitext(os.path.basename(self.input_csv))[0]
        self.out_dir = cfg.out_dir or os.path.join(
            os.path.dirname(self.input_csv), "ALL_PROTEOFORM_DECONVOLUTION"
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.df = None
        self.mz = None
        self.intensity = None
        self.mz_col = None
        self.intensity_col = None
        self.two_col_txt = None
        self.mzml_path = None

    def log(self, *args):
        self._log_fn(" ".join(str(a) for a in args))

    def _check_stop(self):
        if self._stop and self._stop.is_set():
            raise RuntimeError("Aborted by user.")

    def load_input(self):
        cfg = self.cfg
        df = pd.read_csv(self.input_csv)
        mz_col = cfg.mz_col
        if mz_col not in df.columns:
            mz_candidates = [c for c in df.columns if c.lower() in ["mz", "m/z", "mass_to_charge"]]
            if not mz_candidates:
                raise ValueError("Could not find m/z column.")
            mz_col = mz_candidates[0]
        intensity_col = cfg.intensity_col
        if not intensity_col:
            intensity_candidates = [
                c for c in df.columns
                if c != mz_col and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not intensity_candidates:
                raise ValueError("Could not find intensity column.")
            intensity_col = intensity_candidates[0]
        elif intensity_col not in df.columns:
            raise ValueError(f"Intensity column {intensity_col!r} not found in CSV.")
        mz = df[mz_col].astype(float).to_numpy()
        intensity = df[intensity_col].astype(float).to_numpy()
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
        intensity[intensity < 0] = 0
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]
        self.df = df
        self.mz = mz
        self.intensity = intensity
        self.mz_col = mz_col
        self.intensity_col = intensity_col
        self.log("Loaded:", self.input_csv)
        self.log("m/z column:", mz_col, "| Intensity column:", intensity_col)
        self.log("Rows:", len(df), "| m/z range:", float(mz.min()), "to", float(mz.max()))
        self.log("Output:", self.out_dir)

    def export_basic_formats(self):
        self.two_col_txt = os.path.join(self.out_dir, f"{self.base}__two_column_mz_intensity.txt")
        pd.DataFrame({"mz": self.mz, "intensity": self.intensity}).to_csv(
            self.two_col_txt, sep="\t", index=False, header=False
        )
        self.log("Saved two-column input:", self.two_col_txt)
        self.mzml_path = os.path.join(self.out_dir, f"{self.base}__pseudo_input.mzML")
        try:
            import pyopenms as oms
            exp = oms.MSExperiment()
            spec = oms.MSSpectrum()
            spec.setMSLevel(1)
            for m, inten in zip(self.mz, self.intensity):
                p = oms.Peak1D()
                p.setMZ(float(m)); p.setIntensity(float(inten))
                spec.push_back(p)
            spec.sortByPosition()
            exp.addSpectrum(spec)
            oms.MzMLFile().store(self.mzml_path, exp)
            self.log("Saved pseudo mzML:", self.mzml_path)
        except Exception as e:
            self.log("Could not write mzML with pyOpenMS:", e)
            self.mzml_path = None

    def clean_result(self, table, algorithm: str):
        cfg = self.cfg
        if table is None or len(table) == 0:
            return pd.DataFrame(columns=["algorithm", "proteoform_mass", "intensity"])
        out = table.copy()
        if "algorithm" not in out.columns:
            out["algorithm"] = algorithm
        rename_map = {}
        for c in out.columns:
            lc = c.lower()
            if lc in ["mass", "neutral_mass", "monoisotopic_mass", "mono_mass"]:
                rename_map[c] = "proteoform_mass"
            if lc in ["abundance", "height", "area", "score"]:
                rename_map[c] = "intensity"
        out = out.rename(columns=rename_map)
        if "proteoform_mass" not in out.columns or "intensity" not in out.columns:
            return pd.DataFrame(columns=["algorithm", "proteoform_mass", "intensity"])
        out = out[["algorithm", "proteoform_mass", "intensity"]]
        out["proteoform_mass"] = pd.to_numeric(out["proteoform_mass"], errors="coerce")
        out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        out = out[(out["proteoform_mass"] >= cfg.min_mass) & (out["proteoform_mass"] <= cfg.max_mass)]
        out = out[out["intensity"] > 0]
        return out.sort_values("intensity", ascending=False).reset_index(drop=True)

    def save_result(self, table: pd.DataFrame, name: str) -> str:
        path = os.path.join(self.out_dir, f"{self.base}__{name}.csv")
        table.to_csv(path, index=False)
        self.log("Saved:", path)
        return path

    def run_unidec(self) -> pd.DataFrame:
        exe = self.cfg.unidec_exe or shutil.which("unidec") or shutil.which("unidec.exe")
        if exe is None:
            self.log("UniDec skipped: executable not found.")
            return pd.DataFrame()
        self.log("Running UniDec:", exe)
        before = set(glob.glob(os.path.join(self.out_dir, "*")))
        for cmd in [[exe, self.two_col_txt], [exe, "-f", self.two_col_txt]]:
            try:
                subprocess.run(cmd, cwd=self.out_dir, check=False, timeout=240)
                after = set(glob.glob(os.path.join(self.out_dir, "*")))
                mass_files = [
                    f for f in list(after - before) + glob.glob(os.path.join(self.out_dir, "*mass*.txt"))
                    if "mass" in os.path.basename(f).lower() and f.endswith(".txt")
                ]
                if mass_files:
                    mass_file = max(mass_files, key=os.path.getmtime)
                    d = pd.read_csv(mass_file, sep=None, engine="python", header=None)
                    if d.shape[1] >= 2:
                        return self.clean_result(pd.DataFrame({
                            "algorithm": "UniDec",
                            "proteoform_mass": d.iloc[:, 0],
                            "intensity": d.iloc[:, 1],
                        }), "UniDec")
            except Exception as e:
                self.log("UniDec attempt failed:", e)
        return pd.DataFrame()

    def run_ms_deisotope(self) -> pd.DataFrame:
        try:
            from ms_deisotope import deconvolute_peaks
            from ms_deisotope.averagine import peptide
            peaks = sorted(
                [(float(m), float(i)) for m, i in zip(self.mz, self.intensity) if i > 0],
                key=lambda x: -x[1]
            )[:1000]
            if not peaks:
                return pd.DataFrame()
            max_z = min(self.cfg.max_charge, 25)
            self.log(f"  ms_deisotope: {len(peaks)} peaks, charge {self.cfg.min_charge}-{max_z}")
            result = deconvolute_peaks(peaks, averagine=peptide,
                                       charge_range=(self.cfg.min_charge, max_z), truncate_after=0.95)
            rows = [{"algorithm": "ms_deisotope",
                     "proteoform_mass": getattr(p, "neutral_mass", None),
                     "intensity": getattr(p, "intensity", None)}
                    for p in result.peak_set]
            return self.clean_result(pd.DataFrame(rows), "ms_deisotope")
        except Exception as e:
            self.log("ms_deisotope skipped/failed:", e)
            return pd.DataFrame()

    def run_pyopenms_deisotoping(self) -> pd.DataFrame:
        try:
            import pyopenms as oms
            spec = oms.MSSpectrum()
            spec.setMSLevel(1)
            for m, i in zip(self.mz, self.intensity):
                p = oms.Peak1D(); p.setMZ(float(m)); p.setIntensity(float(i))
                spec.push_back(p)
            spec.sortByPosition()
            min_c = max(1, self.cfg.min_charge)
            max_c = min(self.cfg.max_charge, 100)
            deisotoped = False
            for call_args in [
                (spec, 0.01, False, min_c, max_c, True, 2, 10, True, False, False, True, False, True, 2),
                (spec, min_c, max_c, True, 1, False, False, False),
                (spec, min_c, max_c, True, 1),
            ]:
                try:
                    oms.Deisotoper.deisotopeAndSingleCharge(*call_args)
                    deisotoped = True; break
                except TypeError:
                    continue
            if not deisotoped:
                return pd.DataFrame()
            rows = [{"algorithm": "pyOpenMS_deisotoper",
                     "proteoform_mass": float(p.getMZ()),
                     "intensity": float(p.getIntensity())} for p in spec]
            return self.clean_result(pd.DataFrame(rows), "pyOpenMS_deisotoper")
        except Exception as e:
            self.log("pyOpenMS skipped/failed:", e)
            return pd.DataFrame()

    def _parse_generic_mass_files(self, algorithm: str, candidate_files: List[str]) -> pd.DataFrame:
        rows = []
        for f in candidate_files:
            try:
                if os.path.isdir(f): continue
                d = pd.read_csv(f, sep=None, engine="python")
                cols = {str(c).lower(): c for c in d.columns}
                mass_col = next((v for k, v in cols.items() if "mass" in k or "mono" in k), None)
                inten_col = next((v for k, v in cols.items()
                                  if any(x in k for x in ["intensity","abundance","area","height"])), None)
                if mass_col:
                    if inten_col is None:
                        d["intensity_auto"] = 1.0; inten_col = "intensity_auto"
                    rows.append(pd.DataFrame({"algorithm": algorithm,
                                              "proteoform_mass": d[mass_col],
                                              "intensity": d[inten_col]}))
            except Exception:
                pass
        return self.clean_result(pd.concat(rows, ignore_index=True), algorithm) if rows else pd.DataFrame()

    def run_flashdeconv(self) -> pd.DataFrame:
        exe = self.cfg.flashdeconv_exe or shutil.which("FLASHDeconv") or shutil.which("FLASHDeconv.exe")
        if exe is None or self.mzml_path is None or not os.path.exists(self.mzml_path):
            self.log("FLASHDeconv skipped.")
            return pd.DataFrame()
        self.log("Running FLASHDeconv:", exe)
        output_tsv = os.path.join(self.out_dir, f"{self.base}__FLASHDeconv_output.tsv")
        before = set(glob.glob(os.path.join(self.out_dir, "*")))
        for cmd in [
            [exe, "-in", self.mzml_path, "-out", output_tsv],
            [exe, "-in", self.mzml_path, "-out", output_tsv,
             "-min_charge", str(self.cfg.min_charge), "-max_charge", str(self.cfg.max_charge)],
        ]:
            try:
                subprocess.run(cmd, cwd=self.out_dir, check=False, timeout=360)
                after = set(glob.glob(os.path.join(self.out_dir, "*")))
                candidates = list(after - before) + glob.glob(os.path.join(self.out_dir, "*FLASH*"))
                res = self._parse_generic_mass_files("FLASHDeconv", candidates)
                if len(res): return res
            except Exception as e:
                self.log("FLASHDeconv attempt failed:", e)
        return pd.DataFrame()

    def run_topfd(self) -> pd.DataFrame:
        exe = (self.cfg.topfd_exe or shutil.which("topfd") or shutil.which("topfd.exe")
               or shutil.which("TopFD") or shutil.which("TopFD.exe"))
        if exe is None or self.mzml_path is None or not os.path.exists(self.mzml_path):
            self.log("TopFD skipped.")
            return pd.DataFrame()
        self.log("Running TopFD:", exe)
        before = set(glob.glob(os.path.join(self.out_dir, "*")))
        try:
            subprocess.run([exe, self.mzml_path], cwd=self.out_dir, check=False, timeout=360)
            after = set(glob.glob(os.path.join(self.out_dir, "*")))
            candidates = list(after - before) + glob.glob(os.path.join(self.out_dir, "*.tsv"))
            return self._parse_generic_mass_files("TopFD", candidates)
        except Exception as e:
            self.log("TopFD failed:", e)
            return pd.DataFrame()

    def run_sparse_nnls(self) -> pd.DataFrame:
        if not HAS_SCIPY or not HAS_NP:
            self.log("Sparse_NNLS skipped: scipy not installed.")
            return pd.DataFrame()
        try:
            cfg = self.cfg
            mz_min = float(self.mz.min()) if self.mz.size else cfg.min_mass / cfg.max_charge
            mz_max = float(self.mz.max()) if self.mz.size else cfg.max_mass / cfg.min_charge
            n_mass_bins = max(1, int((cfg.max_mass - cfg.min_mass) / cfg.mass_bin_size))
            mass_axis = np.linspace(cfg.min_mass, cfg.max_mass, n_mass_bins)
            mz_axis = self.mz.copy()
            smoothed = gaussian_filter1d(self.intensity.astype(float), sigma=cfg.smooth_sigma)
            A_cols = []
            for M in mass_axis:
                col = np.zeros_like(smoothed)
                for z in range(cfg.min_charge, cfg.max_charge + 1):
                    target_mz = (M + z * cfg.proton_mass) / z
                    if mz_min <= target_mz <= mz_max:
                        idx = int(np.argmin(np.abs(mz_axis - target_mz)))
                        if abs(mz_axis[idx] - target_mz) < cfg.mass_bin_size:
                            col[idx] += 1.0
                if col.max() > 0:
                    col /= col.max()
                A_cols.append(col)
            if not A_cols:
                return pd.DataFrame()
            A = np.column_stack(A_cols)
            coeffs, _ = scipy_nnls(A, smoothed)
            nonzero = coeffs > 0
            rows = [{"algorithm": "Sparse_NNLS",
                     "proteoform_mass": mass_axis[i], "intensity": coeffs[i]}
                    for i in range(len(mass_axis)) if nonzero[i]]
            return self.clean_result(pd.DataFrame(rows), "Sparse_NNLS")
        except Exception as e:
            self.log("Sparse_NNLS failed:", e)
            return pd.DataFrame()

    def run_peak_charge_collapse(self) -> pd.DataFrame:
        if not HAS_SCIPY or not HAS_SKL or not HAS_NP:
            self.log("Peak_Charge_Collapse skipped.")
            return pd.DataFrame()
        try:
            cfg = self.cfg
            smoothed = gaussian_filter1d(self.intensity.astype(float), sigma=cfg.smooth_sigma)
            ymax = smoothed.max() if smoothed.size else 1.0
            min_h    = cfg.peak_height_frac * ymax
            min_prom = cfg.peak_prominence_frac * ymax
            peaks, props = find_peaks(smoothed, height=min_h, prominence=min_prom)
            if not peaks.size:
                return pd.DataFrame()
            peak_df = pd.DataFrame({
                "peak_mz": self.mz[peaks],
                "peak_intensity": self.intensity[peaks],
            })
            rows = []
            for _, r in peak_df.iterrows():
                for z in range(cfg.min_charge, cfg.max_charge + 1):
                    M = z * (float(r["peak_mz"]) - cfg.proton_mass)
                    if cfg.min_mass <= M <= cfg.max_mass:
                        rows.append({"algorithm": "Peak_Charge_Collapse",
                                     "proteoform_mass": M,
                                     "intensity": float(r["peak_intensity"]),
                                     "source_mz": float(r["peak_mz"]), "charge": z})
            raw = pd.DataFrame(rows)
            if raw.empty: return pd.DataFrame()
            raw.to_csv(os.path.join(self.out_dir, f"{self.base}__raw_peak_charge_assignments.csv"), index=False)
            eps_da = np.median(raw["proteoform_mass"]) * cfg.consensus_ppm / 1e6
            raw["cluster"] = DBSCAN(eps=eps_da, min_samples=1).fit_predict(raw[["proteoform_mass"]])
            collapsed = raw.groupby("cluster").agg(
                algorithm=("algorithm", "first"),
                proteoform_mass=("proteoform_mass", "median"),
                intensity=("intensity", "sum"),
                n_assignments=("charge", "count"),
                min_charge=("charge", "min"),
                max_charge=("charge", "max"),
            ).reset_index(drop=True)
            return self.clean_result(collapsed, "Peak_Charge_Collapse")
        except Exception as e:
            self.log("Peak_Charge_Collapse failed:", e)
            return pd.DataFrame()

    def build_consensus(self, all_res: pd.DataFrame) -> pd.DataFrame:
        if not HAS_SKL or all_res is None or all_res.empty:
            return pd.DataFrame()
        all_res = self.clean_result(all_res, "combined")
        if all_res.empty: return pd.DataFrame()
        eps_da = np.median(all_res["proteoform_mass"]) * self.cfg.consensus_ppm / 1e6
        all_res["consensus_cluster"] = DBSCAN(eps=eps_da, min_samples=1).fit_predict(
            all_res[["proteoform_mass"]])
        return all_res.groupby("consensus_cluster").agg(
            proteoform_mass=("proteoform_mass", "median"),
            intensity=("intensity", "sum"),
            n_algorithms=("algorithm", "nunique"),
            algorithms=("algorithm", lambda x: ";".join(sorted(set(x)))),
            n_total_calls=("algorithm", "count"),
        ).reset_index(drop=True).sort_values(["n_algorithms", "intensity"], ascending=False).assign(
            relative_intensity=lambda df: df["intensity"] / df["intensity"].max()
        )

    def make_plots_output(self, results: dict, consensus: pd.DataFrame):
        if not self.cfg.make_plots or not HAS_MPL: return
        try:
            plt.figure(figsize=(14, 5))
            plt.plot(self.mz, self.intensity, linewidth=1)
            plt.xlabel("m/z"); plt.ylabel(self.intensity_col or "Intensity")
            plt.title("Input spectrum"); plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{self.base}__input_spectrum.png"), dpi=300)
            plt.close()
            for name, table in results.items():
                if table is not None and len(table):
                    t = table.sort_values("intensity", ascending=False).head(300)
                    plt.figure(figsize=(14, 5))
                    plt.vlines(t["proteoform_mass"], 0, t["intensity"], linewidth=1)
                    plt.xlabel("Neutral proteoform mass / Da"); plt.ylabel("Intensity")
                    plt.title(name); plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, f"{self.base}__{name}.png"), dpi=300)
                    plt.close()
        except Exception as e:
            self.log("Plotting failed:", e)

    def run_all(self):
        self.load_input()
        self._check_stop()
        self.export_basic_formats()
        self._check_stop()

        algorithm_map = [
            ("UniDec",               "01_UniDec",               self.run_unidec),
            ("ms_deisotope",         "02_ms_deisotope",         self.run_ms_deisotope),
            ("pyOpenMS",             "03_pyOpenMS_deisotoper",  self.run_pyopenms_deisotoping),
            ("FLASHDeconv",          "04_FLASHDeconv",          self.run_flashdeconv),
            ("TopFD",                "05_TopFD",                self.run_topfd),
            ("Sparse_NNLS",          "06_Sparse_NNLS",          self.run_sparse_nnls),
            ("Peak_Charge_Collapse", "07_Peak_Charge_Collapse", self.run_peak_charge_collapse),
        ]

        results = {}
        n = len(algorithm_map)
        for i, (key, tag, fn) in enumerate(algorithm_map):
            self._check_stop()
            if self._prog: self._prog(int(5 + i * 65 / n))
            if key not in self._enabled:
                self.log(f"Skipped (disabled): {key}")
                results[tag] = pd.DataFrame()
                continue
            self.log(f"\n--- Algorithm {i+1}/{n}: {key} ---")
            results[tag] = fn()

        self._check_stop()
        if self._prog: self._prog(75)

        for name, table in results.items():
            self.save_result(table, f"{name}_proteoform_masses")

        valid = [t for t in results.values() if t is not None and len(t)]
        all_res = (pd.concat(valid, ignore_index=True) if valid
                   else pd.DataFrame(columns=["algorithm", "proteoform_mass", "intensity"]))
        all_res = self.clean_result(all_res, "combined")
        all_csv = self.save_result(all_res, "ALL_algorithms_combined_long_table")

        if self._prog: self._prog(85)
        consensus = self.build_consensus(all_res)
        consensus_csv = self.save_result(consensus, "CONSENSUS_proteoform_masses")

        summary = pd.DataFrame({
            "algorithm": list(results.keys()),
            "n_masses": [len(v) for v in results.values()],
        })
        summary_csv = os.path.join(self.out_dir, f"{self.base}__algorithm_summary.csv")
        summary.to_csv(summary_csv, index=False)
        self.log("Saved:", summary_csv)

        if self._prog: self._prog(92)
        self.make_plots_output(results, consensus)
        if self._prog: self._prog(100)

        self.log("\nDONE")
        self.log("Combined CSV:", all_csv)
        self.log("Consensus CSV:", consensus_csv)
        self.log("\nTop consensus proteoforms:")
        self.log(consensus.head(30).to_string(index=False) if len(consensus) else "None found.")

        return {
            "results":   results,
            "combined":  all_res,
            "consensus": consensus,
            "summary":   summary,
            "out_dir":   self.out_dir,
            "all_csv":   all_csv,
            "consensus_csv": consensus_csv,
            "summary_csv":   summary_csv,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point used by the agent
# ══════════════════════════════════════════════════════════════════════════════

def run_deconvolution(
    input_csv: str,
    out_dir: str = None,
    mz_col: str = "m/z",
    intensity_col: str = None,
    min_charge: int = 2,
    max_charge: int = 30,
    min_mass: float = 5000.0,
    max_mass: float = 100000.0,
    consensus_ppm: float = 30.0,
    enabled_algorithms: list = None,
    make_plots: bool = True,
    unidec_exe: str = None,
    flashdeconv_exe: str = None,
    topfd_exe: str = None,
    log=print,
) -> dict:
    """Run proteoform deconvolution on an m/z intensity CSV."""
    cfg = DeconvConfig(
        input_csv=input_csv,
        out_dir=out_dir,
        mz_col=mz_col,
        intensity_col=intensity_col,
        min_charge=min_charge,
        max_charge=max_charge,
        min_mass=min_mass,
        max_mass=max_mass,
        consensus_ppm=consensus_ppm,
        enabled_algorithms=enabled_algorithms,
        make_plots=make_plots,
        unidec_exe=unidec_exe,
        flashdeconv_exe=flashdeconv_exe,
        topfd_exe=topfd_exe,
    )
    deconvolver = ProteoformDeconvolver(cfg, log_fn=log)
    return deconvolver.run_all()
