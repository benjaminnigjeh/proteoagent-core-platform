#!/usr/bin/env python3
"""
proxai_backend.py
Pure backend for ProXAI Generalized Feature Discovery.
No GUI dependencies. Runs in the claude_databank conda env (Python 3.11).
Extracted from proxai_gui.py.
"""

import os, re, gc, json, math
from typing import List, Tuple, Optional, Dict, Any, Iterable

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Flatten, Concatenate
    from tensorflow.keras.models import Model
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
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
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)


def hard_free():
    if HAS_MPL:
        try: plt.close("all")
        except Exception: pass
    if HAS_TF:
        try: tf.keras.backend.clear_session()
        except Exception: pass
    gc.collect(); gc.collect()


def safe_tag(x: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(x).strip()).strip("_")
    return s or "value"


def collect_cast_columns(columns: Iterable[str], regex: str) -> List[Tuple[int, str]]:
    pat = re.compile(regex)
    out = []
    for c in columns:
        m = pat.match(str(c))
        if m:
            out.append((int(m.group(1)), str(c)))
    return sorted(out, key=lambda t: t[0])


def infer_task_type(y: np.ndarray, requested: str) -> str:
    requested = requested.lower().strip()
    if requested in {"binary", "multiclass", "regression"}:
        return requested
    y_nonan = pd.Series(y).dropna().values
    unique = np.unique(y_nonan)
    integer_like = np.allclose(unique, unique.astype(int), atol=1e-8)
    if integer_like and unique.size == 2:
        return "binary"
    if integer_like and 2 < unique.size <= max(20, int(0.2 * len(y_nonan))):
        return "multiclass"
    return "regression"


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def save_json(path, obj):
    def _serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_serial)


def split_csv(csv_path, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    paths = []
    for col in df.columns:
        fname = f"{prefix}_{col}.csv" if prefix else f"{col}.csv"
        out_path = os.path.join(out_dir, fname)
        df[[col]].to_csv(out_path, index=False)
        paths.append(out_path)
    return paths


def mirror_plot(mz, gA, gB, title, out_path):
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(mz, gA, width=(mz[1] - mz[0]) if len(mz) > 1 else 1, label="Run A", alpha=0.7)
    ax.bar(mz, -gB, width=(mz[1] - mz[0]) if len(mz) > 1 else 1, label="Run B (mirrored)", alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("m/z"); ax.set_ylabel("Gradient")
    ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def discover_bins(csv_path: str, retention_bin_col: str) -> list:
    """Return sorted unique values of the retention bin column."""
    df = pd.read_csv(csv_path, usecols=[retention_bin_col])
    vals = list(pd.Series(df[retention_bin_col]).dropna().unique())
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals, key=lambda x: str(x))


def _prepare_batch_codes(sub, good_mask, batch_col, use_batch_embedding):
    if not use_batch_embedding or batch_col not in sub.columns:
        return None, None, None
    batch_values = sub.loc[good_mask, batch_col].astype(str).fillna("missing_batch")
    codes, uniques = pd.factorize(batch_values, sort=True)
    return codes.astype("int32"), [str(x) for x in list(uniques)], int(len(uniques))


def load_bin_data(cfg: dict, bin_value):
    df = pd.read_csv(cfg["csv_path"])
    if cfg["target_col"] not in df.columns:
        raise ValueError(f"target_col={cfg['target_col']!r} not found.")
    if cfg["retention_bin_col"] not in df.columns:
        raise ValueError(f"retention_bin_col={cfg['retention_bin_col']!r} not found.")
    cast_items = collect_cast_columns(df.columns, cfg["cast_col_regex"])
    if not cast_items:
        raise ValueError(f"No cast columns matched regex: {cfg['cast_col_regex']}")
    if cfg.get("max_points") is not None:
        cast_items = cast_items[:cfg["max_points"]]
    sub = df[df[cfg["retention_bin_col"]].astype(str) == str(bin_value)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for retention bin {bin_value!r}")
    cast_indices = np.array([i for i, _ in cast_items], dtype=int)
    cast_cols = [c for _, c in cast_items]
    X_all = sub[cast_cols].apply(pd.to_numeric, errors="coerce").fillna(
        cfg.get("fillna_value", 0.0)).values.astype("float32")
    y_all = pd.to_numeric(sub[cfg["target_col"]], errors="coerce").values
    good = ~pd.isna(y_all)
    X = X_all[good]
    y_raw = y_all[good]
    B, batch_labels, n_batches = _prepare_batch_codes(sub, good, cfg.get("batch_col", ""), cfg.get("use_batch_embedding", False))
    mz = cfg.get("mz_start", 600.0) + cfg.get("mz_step", 0.1) * cast_indices.astype(float)
    return X, y_raw, mz, sub.loc[good].reset_index(drop=True), cast_cols, B, batch_labels, n_batches


# ══════════════════════════════════════════════════════════════════════════════
# Normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_features(X: np.ndarray, method: str, eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = (method or "none").lower().strip()
    X = np.asarray(X, dtype="float32")
    info: Dict[str, Any] = {"method": method}
    if method in {"none", "raw", "false"}:
        return X.astype("float32"), info
    if method in {"standard", "standardscaler", "zscore"}:
        scaler = StandardScaler()
        return scaler.fit_transform(X).astype("float32"), {**info, "type": "per_feature_standard_scaler"}
    if method == "feature_max":
        denom = np.nanmax(np.abs(X), axis=0, keepdims=True)
        denom = np.where(denom > eps, denom, 1.0)
        return (X / denom).astype("float32"), {**info, "type": "per_feature_max_abs"}
    if method == "sample_max":
        denom = np.nanmax(np.abs(X), axis=1, keepdims=True)
        denom = np.where(denom > eps, denom, 1.0)
        return (X / denom).astype("float32"), {**info, "type": "per_sample_max_abs"}
    if method == "log1p":
        X_clip = np.clip(X, a_min=0.0, a_max=None)
        return np.log1p(X_clip).astype("float32"), {**info, "type": "log1p_nonnegative_clip"}
    if method == "log1p_standard":
        X_clip = np.clip(X, a_min=0.0, a_max=None)
        scaler = StandardScaler()
        return scaler.fit_transform(np.log1p(X_clip)).astype("float32"), {**info, "type": "log1p_then_standard"}
    raise ValueError(f"Unknown normalization method: {method!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════════════════

def _inputs_and_feature_tensor(input_dim, n_batches, batch_embedding_dim):
    x_in = Input(shape=(input_dim,), dtype="float32", name="ms1_features")
    if n_batches is not None and n_batches > 0:
        b_in = Input(shape=(1,), dtype="int32", name="batch_id")
        b_emb = Embedding(input_dim=n_batches, output_dim=batch_embedding_dim, name="batch_emb")(b_in)
        b_vec = Flatten(name="batch_emb_flat")(b_emb)
        z = Concatenate(name="concat_ms1_batch")([x_in, b_vec])
        return [x_in, b_in], z
    return x_in, x_in


def build_classifier(input_dim, num_classes, cfg, n_batches=None):
    inputs, x = _inputs_and_feature_tensor(input_dim, n_batches, cfg["batch_embedding_dim"])
    for u in cfg["hidden_units"]:
        x = Dense(u, activation="relu", kernel_regularizer=regularizers.l1(cfg["l1"]))(x)
        if cfg["dropout"] > 0:
            x = Dropout(cfg["dropout"])(x)
    out = tf.keras.layers.Activation("softmax", name="probabilities")(Dense(num_classes, name="logits")(x))
    model = Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model


def build_regressor(input_dim, cfg, n_batches=None):
    inputs, x = _inputs_and_feature_tensor(input_dim, n_batches, cfg["batch_embedding_dim"])
    for u in cfg["hidden_units"]:
        x = Dense(u, activation="relu", kernel_regularizer=regularizers.l1(cfg["l1"]))(x)
        if cfg["dropout"] > 0:
            x = Dropout(cfg["dropout"])(x)
    out = Dense(1, activation="linear", name="regression_output")(x)
    model = Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
                  loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def model_inputs(X_part, B_part):
    if B_part is None: return X_part
    return [X_part, B_part.reshape(-1, 1).astype("int32")]


def encode_class_labels(y_raw):
    classes = np.array(sorted(np.unique(y_raw.astype(int))))
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[int(v)] for v in y_raw], dtype="int32")
    return y, classes, class_to_idx


def train_ensemble(X, y, task_type, cfg, seed_base, num_classes=None,
                   B=None, n_batches=None, log=print, stop_event=None):
    models, metrics = [], []
    k = min(cfg["k_splits"], len(y))
    if k < 2:
        raise ValueError("Need at least 2 samples in a bin.")
    for rep in range(cfg["n_repeats"]):
        if stop_event and stop_event.is_set(): break
        seed = seed_base + rep
        set_seed(seed)
        use_batch = B is not None
        if task_type in {"binary", "multiclass"}:
            counts = pd.Series(y).value_counts()
            splits = (list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))
                      if counts.min() >= k
                      else list(KFold(n_splits=k, shuffle=True, random_state=seed).split(X)))
        else:
            splits = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(X))
        for fold, (tr, va) in enumerate(splits, start=1):
            if stop_event and stop_event.is_set(): break
            B_tr = None if B is None else B[tr]
            B_va = None if B is None else B[va]
            if task_type in {"binary", "multiclass"}:
                model = build_classifier(X.shape[1], int(num_classes), cfg,
                                         n_batches=n_batches if use_batch else None)
                hist = model.fit(model_inputs(X[tr], B_tr), y[tr],
                                 validation_data=(model_inputs(X[va], B_va), y[va]),
                                 epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0)
                pred = np.argmax(model.predict(model_inputs(X[va], B_va), verbose=0), axis=1)
                score = accuracy_score(y[va], pred)
                metrics.append({"repeat": rep+1, "fold": fold, "accuracy": float(score)})
                log(f"  rep {rep+1}, fold {fold}: val_acc={score:.4f}")
            else:
                model = build_regressor(X.shape[1], cfg,
                                        n_batches=n_batches if use_batch else None)
                hist = model.fit(model_inputs(X[tr], B_tr), y[tr],
                                 validation_data=(model_inputs(X[va], B_va), y[va]),
                                 epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0)
                pred = model.predict(model_inputs(X[va], B_va), verbose=0).ravel()
                rmse = math.sqrt(mean_squared_error(y[va], pred))
                r2 = r2_score(y[va], pred) if len(np.unique(y[va])) > 1 else float("nan")
                metrics.append({"repeat": rep+1, "fold": fold, "rmse": float(rmse), "r2": float(r2)})
                log(f"  rep {rep+1}, fold {fold}: val_rmse={rmse:.4f}  val_r2={r2:.4f}")
            models.append(model)
    return models, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Gradient extraction
# ══════════════════════════════════════════════════════════════════════════════

def _call_model(model, x1, b1):
    if b1 is None: return model(x1, training=False)
    return model([x1, b1], training=False)

def _logits_out(model, x1, b1):
    lm = Model(model.input, model.get_layer("logits").output)
    if b1 is None: return lm(x1, training=False)
    return lm([x1, b1], training=False)

def gradient_for_sample_classification(x1, b1, model, extraction, num_classes, eps):
    pos = tf.constant(extraction.get("pos_classes", []), dtype=tf.int32)
    neg_list = extraction.get("neg_classes",
                              [i for i in range(num_classes) if i not in extraction.get("pos_classes", [])])
    neg = tf.constant(neg_list, dtype=tf.int32)
    mode = extraction.get("mode", "log_odds")
    with tf.GradientTape() as tape:
        tape.watch(x1)
        prob = _call_model(model, x1, b1)
        if mode in {"log_odds", "class_vs_rest"}:
            p_pos = tf.reduce_sum(tf.gather(prob, pos, axis=1), axis=1)
            p_neg = tf.reduce_sum(tf.gather(prob, neg, axis=1), axis=1)
            score = tf.math.log(p_pos + eps) - tf.math.log(p_neg + eps)
        elif mode == "probability":
            score = tf.reduce_sum(tf.gather(prob, pos, axis=1), axis=1)
        elif mode == "logit":
            logits = _logits_out(model, x1, b1)
            score = tf.reduce_sum(tf.gather(logits, pos, axis=1), axis=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return tf.squeeze(tape.gradient(score, x1), axis=0)

def gradient_for_sample_regression(x1, b1, model, extraction):
    with tf.GradientTape() as tape:
        tape.watch(x1)
        yhat = tf.squeeze(_call_model(model, x1, b1), axis=-1)
    g = tf.squeeze(tape.gradient(yhat, x1), axis=0)
    mode = extraction.get("mode", "output_gradient")
    if mode == "output_gradient": return g
    if mode == "positive_gradient": return tf.nn.relu(g)
    if mode == "negative_abs_gradient": return tf.nn.relu(-g)
    raise ValueError(f"Unknown regression mode: {mode}")

def average_gradient(X, y, models, extraction, task_type, cfg, num_classes=None, B=None):
    eps = float(cfg.get("eps", 1e-7))
    mode = cfg.get("gradient_sample_mode", "all")
    if mode == "positive" and task_type in {"binary","multiclass"}:
        pos_class = extraction.get("pos_classes", [1])[0] if extraction.get("pos_classes") else 1
        indices = np.where(y == pos_class)[0]
    else:
        indices = np.arange(len(X))
    if len(indices) == 0:
        return np.zeros(X.shape[1], dtype="float32")
    grads = []
    for model in models:
        acc = np.zeros(X.shape[1], dtype="float32")
        for idx in indices:
            x1 = tf.constant(X[idx:idx+1], dtype=tf.float32)
            b1 = None if B is None else tf.constant(B[idx:idx+1].reshape(1,1), dtype=tf.int32)
            if task_type in {"binary","multiclass"}:
                g = gradient_for_sample_classification(x1, b1, model, extraction, num_classes, eps)
            else:
                g = gradient_for_sample_regression(x1, b1, model, extraction)
            acc += g.numpy().ravel()
        grads.append(acc / max(len(indices), 1))
    return np.mean(grads, axis=0) if grads else np.zeros(X.shape[1], dtype="float32")

def summarize_batch_embeddings(models, batch_labels, out_dir, run_tag):
    if not batch_labels: return {}
    summaries = {}
    for i, model in enumerate(models):
        try:
            emb_layer = model.get_layer("batch_emb")
            W = emb_layer.get_weights()[0]
            rows = {batch_labels[j]: W[j].tolist() for j in range(len(batch_labels))}
            summaries[f"fold_{i}"] = rows
        except Exception:
            pass
    if summaries:
        save_json(os.path.join(out_dir, f"batch_embedding_summary_run{run_tag}.json"), summaries)
    return summaries

def class_extraction_defaults(task_type, class_indices):
    if task_type == "binary":
        return [{"name": "pos_1__neg_0_logodds", "mode": "log_odds", "pos_classes": [1], "neg_classes": [0]}]
    return [{"name": f"class_{c}_vs_rest_logodds", "mode": "class_vs_rest", "pos_classes": [c]}
            for c in class_indices]


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points used by the agent
# ══════════════════════════════════════════════════════════════════════════════

def proxai_discover_bins(csv_path: str, retention_bin_col: str) -> list:
    """Discover available retention bins in a ProXAI CSV."""
    return discover_bins(csv_path, retention_bin_col)


def run_proxai(cfg: dict, log=print, prog=None, stop_event=None):
    """
    Run the full ProXAI feature-discovery pipeline.
    cfg is the same dict structure as _parse_config() in proxai_gui.py.
    """
    if not HAS_TF:
        raise RuntimeError("tensorflow is required for ProXAI.")
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for ProXAI.")
    if not HAS_SKL:
        raise RuntimeError("scikit-learn is required for ProXAI.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    out_root = cfg["out_root"]
    os.makedirs(out_root, exist_ok=True)

    all_bins = cfg.get("retention_bin_values") or discover_bins(cfg["csv_path"], cfg["retention_bin_col"])
    log(f"ProXAI: {len(all_bins)} bins to process.")

    for bin_idx, bin_value in enumerate(all_bins):
        if stop_event and stop_event.is_set():
            log("[Aborted]"); return
        if prog:
            prog(int(bin_idx * 100 / max(len(all_bins), 1)))

        bin_tag = safe_tag(bin_value)
        bin_dir  = os.path.join(out_root, f"bin_{bin_tag}")
        csv_dir  = os.path.join(bin_dir, "csv")
        plot_dir = os.path.join(bin_dir, "plots")
        result_dir = os.path.join(csv_dir, "result")
        for d in (csv_dir, plot_dir, result_dir):
            os.makedirs(d, exist_ok=True)

        log(f"\n=== Bin {bin_value} ({bin_idx+1}/{len(all_bins)}) ===")
        try:
            X, y_raw, mz, sub, cast_cols, B, batch_labels, n_batches = load_bin_data(cfg, bin_value)
        except Exception as e:
            log(f"  [skip] {e}"); continue

        task = infer_task_type(y_raw, cfg.get("task_type", "auto"))
        log(f"  task={task}  n_samples={X.shape[0]}  n_features={X.shape[1]}")

        X, normalization_info = normalize_features(X, cfg.get("normalization_method", "none"))

        original_classes = None
        class_to_idx = {}
        num_classes = None
        if task in {"binary", "multiclass"}:
            y, original_classes, class_to_idx = encode_class_labels(y_raw)
            num_classes = len(original_classes)
            extractions = cfg.get("classification_extractions") or class_extraction_defaults(task, list(class_to_idx.values()))
        else:
            y = y_raw
            extractions = cfg.get("regression_extractions") or [
                {"name": "reg_output",  "mode": "output_gradient"},
                {"name": "reg_pos",     "mode": "positive_gradient"},
                {"name": "reg_negabs",  "mode": "negative_abs_gradient"},
            ]

        if cfg.get("save_output_data_map") and cfg.get("output_data_map"):
            save_json(os.path.join(csv_dir, f"bin{bin_tag}_data_map.json"), cfg["output_data_map"])

        if stop_event and stop_event.is_set():
            log("[Aborted]"); return

        log("Training Run A ensemble...")
        models_A, metrics_A = train_ensemble(X, y, task, cfg, cfg["seed_A"], num_classes,
                                              B=B, n_batches=n_batches, log=log, stop_event=stop_event)
        if stop_event and stop_event.is_set():
            log("[Aborted]"); return

        log("Training Run B ensemble...")
        models_B, metrics_B = train_ensemble(X, y, task, cfg, cfg["seed_B"], num_classes,
                                              B=B, n_batches=n_batches, log=log, stop_event=stop_event)

        batch_summary_A = summarize_batch_embeddings(models_A, batch_labels, csv_dir, "A") if cfg.get("save_batch_embedding_summary") else {}
        batch_summary_B = summarize_batch_embeddings(models_B, batch_labels, csv_dir, "B") if cfg.get("save_batch_embedding_summary") else {}

        for extraction in extractions:
            if stop_event and stop_event.is_set():
                log("[Aborted]"); return
            name = safe_tag(extraction.get("name", extraction.get("mode", "extract")))
            log(f"  Extracting: {name}")
            gA = average_gradient(X, y, models_A, extraction, task, cfg, num_classes, B=B)
            gB = average_gradient(X, y, models_B, extraction, task, cfg, num_classes, B=B)
            pos_gA = np.maximum(gA, 0); pos_gB = np.maximum(gB, 0)
            neg_gA = np.maximum(-gA, 0); neg_gB = np.maximum(-gB, 0)
            cos_signed = cosine_sim(gA, gB)
            cos_pos    = cosine_sim(pos_gA, pos_gB)
            cos_neg    = cosine_sim(neg_gA, neg_gB)

            cols: Dict[str, Any] = {"m/z": mz, "grad_runA": gA, "grad_runB": gB}
            if cfg.get("save_pos_columns", True):
                cols["pos_runA"] = pos_gA; cols["pos_runB"] = pos_gB
            if cfg.get("save_negabs_columns", True):
                cols["negabs_runA"] = neg_gA; cols["negabs_runB"] = neg_gB
            out_df = pd.DataFrame(cols)

            csv_out = os.path.join(csv_dir, f"bin{bin_tag}_grads_AB__{name}.csv")
            if cfg.get("save_combined_csv", True):
                out_df.to_csv(csv_out, index=False)
                log(f"    wrote {csv_out}")

            if cfg.get("save_plot_signed", True):
                mirror_plot(mz, gA, gB,
                            f"Bin {bin_value} — {name} signed; cos={cos_signed:.4f}",
                            os.path.join(plot_dir, f"bin{bin_tag}_{name}_signed_mirror.png"))
            if cfg.get("save_plot_pos", True):
                mirror_plot(mz, pos_gA, pos_gB,
                            f"Bin {bin_value} — {name} positive; cos={cos_pos:.4f}",
                            os.path.join(plot_dir, f"bin{bin_tag}_{name}_pos_mirror.png"))
            if cfg.get("save_plot_negabs", True):
                mirror_plot(mz, neg_gA, neg_gB,
                            f"Bin {bin_value} — {name} neg-abs; cos={cos_neg:.4f}",
                            os.path.join(plot_dir, f"bin{bin_tag}_{name}_negabs_mirror.png"))

            split_paths = (split_csv(csv_out, result_dir, prefix=f"bin{bin_tag}")
                           if cfg.get("split_combined_csv") and cfg.get("save_combined_csv", True) else [])

            if cfg.get("save_summary_json", True):
                save_json(os.path.join(csv_dir, f"bin{bin_tag}_summary__{name}.json"), {
                    "bin": bin_value, "task_type": task, "extraction": extraction,
                    "original_classes": None if original_classes is None else original_classes.tolist(),
                    "class_to_internal_index": class_to_idx,
                    "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
                    "normalization": normalization_info,
                    "metrics_runA": metrics_A, "metrics_runB": metrics_B,
                    "cosine_signed": cos_signed, "cosine_pos": cos_pos, "cosine_neg_abs": cos_neg,
                    "combined_csv": csv_out, "split_csvs": split_paths,
                })

        for m in (models_A + models_B):
            del m
        hard_free()

    if prog: prog(100)
    log("\nAll ProXAI bins completed.")
