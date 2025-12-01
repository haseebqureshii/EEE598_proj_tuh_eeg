import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost + SHAP (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RNG_SEED = 42
np.random.seed(RNG_SEED)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()
FE_FILE_NAME = "features_nodelta.npz"

TRAIN_PATH = BASE_OUT_DIR / "train_internal" / FE_FILE_NAME
DEV_PATH   = BASE_OUT_DIR / "dev_internal"   / FE_FILE_NAME
TEST_PATH  = BASE_OUT_DIR / "test_internal"  / FE_FILE_NAME

# Effect-size feature selection
TOP_K_BY_ABS_D = 150     # number of features by |d|
MIN_STD_FOR_D  = 1e-6    # avoid division by tiny std

# SHAP control
DO_SHAP = True           # set False if too slow or missing deps
N_SHAP_WINDOWS = 2000    # subsample windows for SHAP on DEV

# Figure directories
FIG_BASE_DIR  = BASE_OUT_DIR / "figures_comp"
FIG_TRAIN_DIR = FIG_BASE_DIR / "train"
FIG_DEV_DIR   = FIG_BASE_DIR / "dev"
FIG_TEST_DIR  = FIG_BASE_DIR / "test"
FIG_SHAP_DIR  = FIG_BASE_DIR / "shap"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Helpers: loading + subject IDs
# -------------------------------------------------------------------

def load_meta_list(meta_array):
    """Handle meta stored as list-of-dicts or ndarray-of-objects."""
    if isinstance(meta_array, list):
        return meta_array
    out = []
    for m in meta_array:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(m.item())
    return out


def subject_ids_from_meta(meta_list):
    """
    Returns a list of subject IDs per recording index:
      subj_id_per_rec[rec_idx] = subject_id (string)
    """
    ids = []
    for i, m in enumerate(meta_list):
        sid = m.get("subject_id", f"subj_{i}")
        ids.append(str(sid))
    return ids


def subject_ids_per_window(rec_idx, subj_id_per_rec):
    """Map window-level rec_idx to subject IDs."""
    return np.array([subj_id_per_rec[i] for i in rec_idx])


def _infer_feature_names(data, X, path: Path):
    """
    Try hard to find a feature-name array in the NPZ.
    Returns (feat_names, source_key or 'synthetic').
    """
    n_features = X.shape[1]
    candidate_keys = [
        "feat_names",
        "feature_names",
        "feature_list",
        "feature_list_all",
        "feature_list_no_delta",
        "names",
    ]

    for key in candidate_keys:
        if key in data.files:
            arr = data[key]
            if arr.ndim == 1 and len(arr) == n_features:
                return [str(x) for x in arr.tolist()], key

    # Heuristic: any 1-D string/object array of length n_features
    for key in data.files:
        if key in ["X", "y", "meta", "rec_idx", "ch_names"]:
            continue
        arr = data[key]
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim == 1
            and arr.shape[0] == n_features
            and arr.dtype.kind in ("O", "U", "S")
        ):
            return [str(x) for x in arr.tolist()], key

    # Fallback
    warnings.warn(
        f"[WARN] No explicit feature-name array found in {path.name}; "
        f"using synthetic names f0..f{n_features-1}."
    )
    return [f"f{j}" for j in range(n_features)], "synthetic"


def load_features_split(path: Path):
    """
    Load X, y, meta, rec_idx, feature_names from NPZ.
    """
    print(f"[Load] {path}")
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    meta_list = load_meta_list(data["meta"])
    rec_idx = data["rec_idx"]

    feat_names, src_key = _infer_feature_names(data, X, path)
    print(
        f"  X shape: {X.shape}, y shape: {y.shape}, "
        f"n_features={X.shape[1]}, n_records={len(meta_list)}"
    )
    print(f"  Feature-name source: {src_key}")
    return X, y, meta_list, rec_idx, feat_names


# -------------------------------------------------------------------
# Effect size (Cohen's d) + SUNSET features + plots + CSVs
# -------------------------------------------------------------------

def compute_cohens_d_by_feature(X, y):
    """
    Compute Cohen's d for each feature between y==1 (preictal) and y==0 (background).
    Returns array of d with shape (n_features,).
    """
    y = np.asarray(y)
    X = np.asarray(X)
    assert X.ndim == 2

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    X_pos = X[idx_pos]
    X_neg = X[idx_neg]

    mu1 = X_pos.mean(axis=0)
    mu0 = X_neg.mean(axis=0)
    var1 = X_pos.var(axis=0, ddof=1)
    var0 = X_neg.var(axis=0, ddof=1)

    n1 = len(idx_pos)
    n0 = len(idx_neg)
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2 + 1e-9)
    pooled_std = np.sqrt(np.maximum(pooled_var, MIN_STD_FOR_D))

    d = (mu1 - mu0) / pooled_std
    return d


def sunset_feature_mask(feat_names):
    """
    Return a boolean mask indicating SUNSET / concept-driven features.
    (graph/network, connectivity, power-law residuals, payoff, etc.)
    """
    tokens = [
        "plv", "coh", "corr",
        "psd_plaw", "psd_slope", "plaw_resid", "plaw_crit",
        "temp_payoff", "payoff",
        "graph_", "deg_", "clust_",
    ]
    mask = np.zeros(len(feat_names), dtype=bool)
    for i, name in enumerate(feat_names):
        lname = name.lower()
        if any(tok in lname for tok in tokens):
            mask[i] = True
    return mask


def _save_effect_size_csv(
    feat_names,
    d,
    abs_d,
    top_indices,
    sunset_mask,
    selected_indices,
    fig_dir: Path,
):
    """Write CSVs with effect-size information."""
    ensure_dir(fig_dir)

    # all features
    all_path = fig_dir / "effect_size_all_features.csv"
    with all_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["feature_name", "d", "abs_d", "in_top_k_absd",
             "is_sunset_candidate", "is_selected_union"]
        )
        top_set = set(top_indices.tolist())
        sel_set = set(selected_indices.tolist())
        for j, name in enumerate(feat_names):
            writer.writerow([
                name,
                float(d[j]),
                float(abs_d[j]),
                int(j in top_set),
                int(bool(sunset_mask[j])),
                int(j in sel_set),
            ])

    # top-20 by |d|
    top20 = top_indices[:20]
    top20_path = fig_dir / "effect_size_top20_by_absd.csv"
    with top20_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_name", "d", "abs_d"])
        for rank, j in enumerate(top20, start=1):
            writer.writerow([rank, feat_names[j], float(d[j]), float(abs_d[j])])


def plot_effect_size_hist_and_top(abs_d, d, feat_names, sel_idx_top, fig_dir: Path):
    ensure_dir(fig_dir)

    # 1) Histogram of |d|
    max_d = float(np.nanmax(abs_d))
    last_edge = max(0.80001, max_d + 1e-6)
    bins = [0.0, 0.2, 0.5, 0.8, last_edge]
    labels = ["<0.2", "0.2–0.5", "0.5–0.8", ">0.8"]

    plt.figure(figsize=(6, 4))
    plt.hist(abs_d, bins=bins, edgecolor="black")
    plt.xticks(
        [(bins[i] + bins[i + 1]) / 2 for i in range(len(labels))],
        labels
    )
    plt.xlabel("|Cohen's d|")
    plt.ylabel("Feature count")
    plt.title("Histogram of |Cohen's d| across features")
    plt.tight_layout()
    plt.savefig(fig_dir / "effect_size_hist.png", dpi=300)
    plt.close()

    # 2) Top-20 features by |d|
    top20 = sel_idx_top[:20]
    top_names = [feat_names[i] for i in top20]
    top_d = d[top20]

    plt.figure(figsize=(max(8, len(top20) * 0.4), 5))
    colors = ["tab:red" if val > 0 else "tab:blue" for val in top_d]
    positions = np.arange(len(top20))
    plt.bar(positions, top_d, color=colors)
    plt.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    plt.xticks(positions, top_names, rotation=90)
    plt.ylabel("Cohen's d (preictal > background)")
    plt.title("Top 20 features by effect size (TRAIN)")
    plt.tight_layout()
    plt.savefig(fig_dir / "effect_size_top20.png", dpi=300)
    plt.close()


def select_features_by_d_and_sunset(X_train, y_train, feat_names, top_k=TOP_K_BY_ABS_D):
    """
    Compute d on TRAIN, pick top-K by |d|, union with SUNSET features.
    Returns:
      selected_indices, selected_names, d (full vector), abs_d, top_indices
    """
    print("\n[Select] Computing Cohen's d on TRAIN_INTERNAL...")
    d = compute_cohens_d_by_feature(X_train, y_train)
    abs_d = np.abs(d)

    # basic histogram of |d|
    bins = [0.0, 0.2, 0.5, 0.8, np.inf]
    counts = np.zeros(len(bins) - 1, dtype=int)
    for val in abs_d:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i + 1]:
                counts[i] += 1
                break
    print("[Select] |d| histogram (TRAIN):")
    print(f"  |d| < 0.2 (tiny)  : {counts[0]} features")
    print(f"  0.2–0.5 (small)   : {counts[1]} features")
    print(f"  0.5–0.8 (medium)  : {counts[2]} features")
    print(f"  > 0.8 (large)     : {counts[3]} features")

    # top-K by |d|
    order = np.argsort(-abs_d)  # descending
    top_indices = order[:top_k]
    print(f"[Select] Top-{top_k} by |d| include e.g.:")
    for rank in range(min(20, top_k)):
        j = top_indices[rank]
        print(
            f"  #{rank+1:2d}: |d|={abs_d[j]:.3f}  d={d[j]:+.3f}  {feat_names[j]}"
        )

    # SUNSET mask
    sunset_mask = sunset_feature_mask(feat_names)
    n_sunset = sunset_mask.sum()
    print(f"[Select] SUNSET feature candidates (by name) : {n_sunset}")

    sunset_indices = np.where(sunset_mask)[0]

    # union
    selected_set = set(top_indices.tolist()) | set(sunset_indices.tolist())
    selected_indices = np.array(sorted(selected_set), dtype=int)
    selected_names = [feat_names[j] for j in selected_indices]

    print(f"[Select] Final feature set size (top-K ∪ SUNSET): {len(selected_indices)}")

    # plots (TRAIN)
    plot_effect_size_hist_and_top(abs_d, d, feat_names, top_indices, FIG_TRAIN_DIR)
    # CSVs for lookup
    _save_effect_size_csv(
        feat_names, d, abs_d, top_indices,
        sunset_mask, selected_indices, FIG_TRAIN_DIR
    )

    return selected_indices, selected_names, d, abs_d, top_indices


# -------------------------------------------------------------------
# Model tuning + evaluation + plots
# -------------------------------------------------------------------

def tune_model(model, param_dist, X, y, groups, model_name, n_iter=30):
    """
    RandomizedSearchCV with GroupKFold on TRAIN_INTERNAL.
    """
    print(f"\n[ML] Tuning {model_name} with GroupKFold + RandomizedSearchCV...")
    cv = GroupKFold(n_splits=5)

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        verbose=1,
        random_state=RNG_SEED,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(X, y, groups=groups)
    print(f"[ML] {model_name} best AUPRC (CV): {rs.best_score_:.4f}")
    print(f"[ML] {model_name} best params: {rs.best_params_}")
    return rs.best_estimator_, rs.best_score_


def pick_best_threshold(y_true, y_prob):
    """
    Sweep threshold in [0,1] to maximize F1.
    Returns best_thr, best_f1, confusion matrix at that thr.
    """
    thresholds = np.linspace(0.0, 1.0, 501)
    best_f1 = -1.0
    best_thr = 0.5
    best_cm = None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_cm = confusion_matrix(y_true, y_pred)

    return best_thr, best_f1, best_cm


def per_subject_auprc(y_true, y_prob, subj_ids):
    """
    Compute AUPRC per subject; returns dict subject -> AUPRC,
    plus summary stats (mean, std, min, max) over subjects
    with non-constant labels.
    """
    subj_to_indices = defaultdict(list)
    for i, sid in enumerate(subj_ids):
        subj_to_indices[sid].append(i)

    subj_scores = {}
    for sid, idxs in subj_to_indices.items():
        idxs = np.asarray(idxs, dtype=int)
        y_s = y_true[idxs]
        p_s = y_prob[idxs]
        if (y_s == 1).any() and (y_s == 0).any():
            # valid AUPRC
            score = average_precision_score(y_s, p_s)
        else:
            score = np.nan
        subj_scores[sid] = score

    valid_scores = [v for v in subj_scores.values() if np.isfinite(v)]
    if valid_scores:
        arr = np.array(valid_scores)
        stats = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    else:
        stats = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return subj_scores, stats


def plot_pr_roc(y_true, y_prob, split_name, fig_dir: Path):
    ensure_dir(fig_dir)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = (y_true == 1).mean()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{split_name} PR")
    plt.axhline(prevalence, linestyle="--", color="gray", label=f"No-skill ({prevalence:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve ({split_name})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split_name.lower()}_pr_curve.png", dpi=300)
    plt.close()

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{split_name} ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-discrimination")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve ({split_name})")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{split_name.lower()}_roc_curve.png", dpi=300)
        plt.close()
    except ValueError:
        print(f"[Plot] ROC not defined for {split_name} (only one class).")


def plot_multi_pr_roc(y_true, prob_dict, split_name, fig_dir: Path):
    """
    prob_dict: {config_name: y_prob}
    Plots multi-curve PR & ROC on same figure (for DEV/TEST).
    Each legend entry includes AUPRC.
    """
    ensure_dir(fig_dir)
    prevalence = (y_true == 1).mean()

    # ---------- Precision–Recall ----------
    plt.figure(figsize=(7, 6))
    for cfg_name, y_prob in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{cfg_name} (AUPRC={ap:.3f})")
    plt.axhline(prevalence, linestyle="--", color="gray",
                label=f"No-skill ({prevalence:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curves ({split_name})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split_name.lower()}_multi_pr_curve.png", dpi=300)
    plt.close()

    # ---------- ROC ----------
    plt.figure(figsize=(7, 6))
    for cfg_name, y_prob in prob_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Single class → ROC undefined, skip
            continue
        plt.plot(fpr, tpr, label=f"{cfg_name} (AUROC={auc_roc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
             label="No-discrimination")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves ({split_name})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split_name.lower()}_multi_roc_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, split_name, thr, fig_dir: Path):
    ensure_dir(fig_dir)
    TN, FP, FN, TP = cm.ravel()

    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion matrix ({split_name}, thr={thr:.3f})")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    # Annotate
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            plt.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(fig_dir / f"{split_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_per_subject_auprc(subj_scores, split_name, fig_dir: Path, max_subjects_to_plot=40):
    ensure_dir(fig_dir)
    # Only finite scores
    items = [(sid, sc) for sid, sc in subj_scores.items() if np.isfinite(sc)]
    if not items:
        print(f"[Plot] No valid per-subject AUPRC to plot for {split_name}.")
        return

    # Sort by score
    items.sort(key=lambda kv: kv[1])
    if len(items) > max_subjects_to_plot:
        items = items[-max_subjects_to_plot:]  # keep top N

    sids = [sid for sid, _ in items]
    scores = [sc for _, sc in items]

    x = np.arange(len(sids))
    plt.figure(figsize=(max(6, len(sids) * 0.25), 4))
    plt.bar(x, scores)
    plt.xticks(x, sids, rotation=90)
    plt.ylabel("AUPRC")
    plt.ylim(0.0, 1.0)
    plt.title(f"Per-subject AUPRC ({split_name})")
    plt.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split_name.lower()}_per_subject_auprc.png", dpi=300)
    plt.close()


def plot_feature_importance(best_model, feat_names_sel, split_name, fig_dir: Path, top_n=20):
    ensure_dir(fig_dir)
    if not hasattr(best_model, "feature_importances_"):
        print(f"[Plot] Model {type(best_model)} has no feature_importances_; skipping FI plot.")
        return

    importances = np.asarray(best_model.feature_importances_)
    order = np.argsort(-importances)
    top = order[:top_n]
    names = [feat_names_sel[i] for i in top]
    vals = importances[top]

    x = np.arange(len(top))
    plt.figure(figsize=(max(6, len(top) * 0.3), 4))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=90)
    plt.ylabel("Feature importance")
    plt.title(f"Top {top_n} feature importances ({split_name})")
    plt.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split_name.lower()}_feature_importance.png", dpi=300)
    plt.close()

    # CSV for lookup
    csv_path = fig_dir / f"{split_name.lower()}_feature_importance_top{top_n}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_name", "importance"])
        for rank, (name, val) in enumerate(zip(names, vals), start=1):
            writer.writerow([rank, name, float(val)])


def summarize_split(name, y_true, y_prob, subj_ids, thr, fig_dir: Path):
    """
    Print global metrics + confusion matrix + per-subject AUPRC stats,
    generate plots, and append metrics to a CSV file.
    """
    ensure_dir(fig_dir)

    # -------- Global metrics --------
    n_win = len(y_true)
    n_subj = len(set(subj_ids))
    prevalence = float((y_true == 1).mean())

    auprc = average_precision_score(y_true, y_prob)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    print(f"\n========== EVAL on {name} ==========")
    print(f"#windows            : {n_win}")
    print(f"#subjects           : {n_subj}")
    print(f"Prevalence (y=1)    : {prevalence:.4f}")
    print(f"AUPRC (prob)        : {auprc:.4f}")
    print(f"AUROC (prob)        : {auroc:.4f}")

    # -------- PR + ROC plots --------
    plot_pr_roc(y_true, y_prob, name, fig_dir)

    # -------- Confusion matrix at chosen threshold --------
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Derived metrics
    sens = TP / (TP + FN + 1e-9)   # recall / TPR
    spec = TN / (TN + FP + 1e-9)   # TNR
    prec = TP / (TP + FP + 1e-9)   # PPV
    rec  = sens
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"\n[CM] threshold={thr:.3f}")
    print(cm)
    print(f"Sensitivity (TPR): {sens:.3f}")
    print(f"Specificity (TNR): {spec:.3f}")
    print(f"Precision (PPV)  : {prec:.3f}")
    print(f"Recall           : {rec:.3f}")
    print(f"F1-score         : {f1:.3f}")

    plot_confusion_matrix(cm, name, thr, fig_dir)

    # -------- Per-subject AUPRC --------
    subj_scores, stats = per_subject_auprc(y_true, y_prob, subj_ids)
    valid_scores = [v for v in subj_scores.values() if np.isfinite(v)]
    n_valid = len(valid_scores)

    print("\n[Per-subject AUPRC] (subjects with both classes):")
    print(f"  Subjects with valid AUPRC: {n_valid}")
    print(
        f"  Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
        f"Min={stats['min']:.4f}, Max={stats['max']:.4f}"
    )

    sorted_subj = sorted(subj_scores.items(), key=lambda kv: (np.isnan(kv[1]), kv[0]))
    for sid, sc in sorted_subj:
        if np.isnan(sc):
            print(f"    {sid:>10s} :   nan (single-class)")
        else:
            print(f"    {sid:>10s} : {sc:6.3f}")

    plot_per_subject_auprc(subj_scores, name, fig_dir)

    # -------- Append metrics to CSV (one row per split) --------
    metrics_csv = FIG_BASE_DIR / "metrics_summary.csv"
    write_header = not metrics_csv.exists()

    with metrics_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "split_name",
                "n_windows",
                "n_subjects",
                "prevalence",
                "auprc",
                "auroc",
                "threshold",
                "TN", "FP", "FN", "TP",
                "sensitivity",
                "specificity",
                "precision",
                "recall",
                "f1",
                "per_subj_auprc_mean",
                "per_subj_auprc_std",
                "per_subj_auprc_min",
                "per_subj_auprc_max",
                "per_subj_auprc_n_valid",
            ])

        writer.writerow([
            name,
            n_win,
            n_subj,
            prevalence,
            auprc,
            auroc,
            thr,
            int(TN), int(FP), int(FN), int(TP),
            sens,
            spec,
            prec,
            rec,
            f1,
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
            n_valid,
        ])


# -------------------------------------------------------------------
# Optional SHAP on best tree model
# -------------------------------------------------------------------
def run_shap_on_dev(best_model, X_dev, y_dev, feat_names, model_name="best_model"):
    ensure_dir(FIG_SHAP_DIR)

    if not HAS_SHAP:
        print("[SHAP] shap library not available; skipping.")
        return
    if not hasattr(best_model, "predict_proba"):
        print("[SHAP] model has no predict_proba; skipping.")
        return

    # Make sure feat_names is a plain Python list (for clean indexing)
    feat_names = list(feat_names)

    # Try TreeExplainer; if fails, bail
    try:
        explainer = shap.TreeExplainer(best_model)
    except Exception as e:
        print(f"[SHAP] Could not create TreeExplainer: {e}")
        return

    # Subsample DEV for SHAP
    n = len(y_dev)
    idx_all = np.arange(n)
    pos_idx = idx_all[y_dev == 1]
    neg_idx = idx_all[y_dev == 0]
    n_pos = min(len(pos_idx), N_SHAP_WINDOWS // 2)
    n_neg = min(len(neg_idx), N_SHAP_WINDOWS - n_pos)
    if n_pos + n_neg == 0:
        print("[SHAP] No windows to sample; skipping.")
        return

    rng = np.random.RandomState(RNG_SEED)
    take_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    take_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    take = np.concatenate([take_pos, take_neg])
    rng.shuffle(take)

    X_shap = X_dev[take]
    y_shap = y_dev[take]
    print(f"[SHAP] Using {len(X_shap)} DEV windows for SHAP analysis.")

    shap_values = explainer.shap_values(X_shap)

    # ----------------------------------------------------
    # Normalize shap_values to a 2D array [n_samples, n_features]
    # ----------------------------------------------------
    sv_raw = shap_values

    # Case 1: list of arrays (class 0, class 1, ...)
    if isinstance(sv_raw, list):
        if len(sv_raw) >= 2:
            sv_raw = sv_raw[1]    # focus on class 1
        else:
            sv_raw = sv_raw[0]

    sv_raw = np.asarray(sv_raw)
    print(f"[SHAP] raw shap_values shape: {sv_raw.shape}")

    if sv_raw.ndim == 2:
        # (n_samples, n_features) – already what we want
        sv = sv_raw

    elif sv_raw.ndim == 3:
        # Common XGBoost case: (n_samples, n_features, n_outputs/classes)
        # Your log showed (2000, 289, 2) – pick class 1 along last axis.
        if sv_raw.shape[2] == 2 and sv_raw.shape[0] == X_shap.shape[0]:
            sv = sv_raw[:, :, 1]  # (n_samples, n_features)
        # Another possible layout: (n_classes, n_samples, n_features)
        elif sv_raw.shape[0] == 2 and sv_raw.shape[1] == X_shap.shape[0]:
            sv = sv_raw[1, :, :]  # (n_samples, n_features)
        else:
            print(f"[SHAP] Unsupported 3D shap_values shape {sv_raw.shape}; skipping SHAP.")
            return
    else:
        print(f"[SHAP] Unsupported shap_values ndim={sv_raw.ndim}, shape={sv_raw.shape}; skipping SHAP.")
        return

    # Safety check
    if sv.shape[0] != X_shap.shape[0] or sv.shape[1] != len(feat_names):
        print(f"[SHAP] Mismatch between SHAP array {sv.shape} and features {len(feat_names)}; skipping SHAP.")
        return

    print(f"[SHAP] Using SHAP matrix shape: {sv.shape} (n_samples, n_features)")

    # ----------------------------------------------------
    # Global importance = mean |SHAP| per feature
    # ----------------------------------------------------
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-mean_abs)

    # Save CSV lookup table with proper feature names
    shap_csv_path = FIG_SHAP_DIR / f"shap_importance_global.csv"
    with shap_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_name", "mean_abs_shap"])
        for rank, j in enumerate(order, start=1):
            idx = int(j)
            writer.writerow([rank, feat_names[idx], float(mean_abs[idx])])

    # ----------------------------------------------------
    # SHAP summary plot (top 20 features)
    # ----------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The NumPy global RNG was seeded by calling `np.random.seed`",
            category=FutureWarning,
        )
        plt.figure(figsize=(7, 10))
        shap.summary_plot(
            sv,
            X_shap,
            feature_names=feat_names,
            show=False,
            max_display=20,
            plot_type="violin",
        )
        plt.tight_layout()
        plt.savefig(FIG_SHAP_DIR / f"shap_summary_{model_name}.png", dpi=300)
        plt.close()

# -------------------------------------------------------------------
# RUN ONE FEATURE CONFIG
# -------------------------------------------------------------------

def run_config(config_name,
               X_tr_cfg, X_dev_cfg, X_te_cfg,
               y_tr, y_dev, y_te,
               subj_tr_win, subj_dev_win, subj_te_win,
               feat_names_cfg):
    """
    Train/tune RF, SVM, XGB for one feature configuration and evaluate.
    Returns dict with dev/test probabilities, threshold, and model info.
    """
    print("\n" + "=" * 60)
    print(f" RUNNING CONFIG: {config_name}")
    print("=" * 60)
    print(f"[Config {config_name}] TRAIN: {X_tr_cfg.shape}, DEV: {X_dev_cfg.shape}, TEST: {X_te_cfg.shape}")

    groups_tr = subj_tr_win

    best_models = []
    cv_scores = []

    # --- RandomForest
    rf = RandomForestClassifier(random_state=RNG_SEED, n_jobs=-1)
    param_rf = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5, 0.7],
        "class_weight": [None, "balanced",
                         {0: 1.0, 1: 3.0},
                         {0: 1.0, 1: 5.0}],
    }
    best_rf, score_rf = tune_model(
        rf, param_rf, X_tr_cfg, y_tr, groups_tr,
        f"RandomForest ({config_name})", n_iter=25
    )
    best_models.append(("RandomForest", best_rf))
    cv_scores.append(score_rf)

    # --- SVM (RBF)
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RNG_SEED)),
        ]
    )
    param_svm = {
        "clf__C": [0.1, 1.0, 10.0, 30.0],
        "clf__gamma": [0.001, 0.01, 0.1],
        "clf__class_weight": [None, "balanced"],
    }
    best_svm, score_svm = tune_model(
        svm, param_svm, X_tr_cfg, y_tr, groups_tr,
        f"SVM_RBF ({config_name})", n_iter=25
    )
    best_models.append(("SVM_RBF", best_svm))
    cv_scores.append(score_svm)

    # --- XGB (if available)
    if HAS_XGB:
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RNG_SEED,
            n_jobs=-1,
            use_label_encoder=False,
        )
        param_xgb = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [1.0, 2.0, 3.0, 5.0],
        }
        best_xgb, score_xgb = tune_model(
            xgb, param_xgb, X_tr_cfg, y_tr, groups_tr,
            f"XGB ({config_name})", n_iter=35
        )
        best_models.append(("XGB", best_xgb))
        cv_scores.append(score_xgb)
    else:
        print(f"\n[ML] xgboost not installed; skipping XGB for {config_name}.")

    # Pick best
    best_idx = int(np.argmax(cv_scores))
    best_name, best_model = best_models[best_idx]
    print("\n[ML] === MODEL SELECTION (config: {cfg}) ===".format(cfg=config_name))
    for (name, _), s in zip(best_models, cv_scores):
        print(f"  {name:12s}  CV AUPRC={s:.4f}")
    print(f"  -> Selected model: {best_name} for {config_name}")

    # Feature importance
    if best_name in ("RandomForest", "XGB"):
        plot_feature_importance(
            best_model, feat_names_cfg,
            f"TRAIN_INTERNAL_{config_name}",
            FIG_TRAIN_DIR
        )

    # DEV eval, threshold
    print(f"\n[ML] Evaluating {config_name} on DEV_INTERNAL for threshold selection...")
    y_dev_prob = best_model.predict_proba(X_dev_cfg)[:, 1]
    dev_auprc = average_precision_score(y_dev, y_dev_prob)
    dev_auroc = roc_auc_score(y_dev, y_dev_prob)
    print(f"  DEV AUPRC (prob): {dev_auprc:.4f}")
    print(f"  DEV AUROC (prob): {dev_auroc:.4f}")

    best_thr, best_f1, dev_cm_thr = pick_best_threshold(y_dev, y_dev_prob)
    print(f"\n[ML] Best threshold on DEV (max F1): thr={best_thr:.3f}, F1={best_f1:.4f}")
    print("DEV confusion matrix at best threshold:")
    print(dev_cm_thr)

    summarize_split(f"DEV_{config_name}", y_dev, y_dev_prob, subj_dev_win, best_thr, FIG_DEV_DIR)

    # TEST (info only)
    print(f"\n[ML] INFO-ONLY: Evaluating {config_name} on TEST_INTERNAL using DEV-chosen threshold...")
    y_te_prob = best_model.predict_proba(X_te_cfg)[:, 1]
    summarize_split(f"TEST_{config_name}", y_te, y_te_prob, subj_te_win, best_thr, FIG_TEST_DIR)

    return {
        "config_name": config_name,
        "best_model_name": best_name,
        "best_model": best_model,
        "feat_names_cfg": list(feat_names_cfg),
        "y_dev_prob": y_dev_prob,
        "y_te_prob": y_te_prob,
        "dev_thr": best_thr,
    }


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("\n==============================")
    print(" 3×CONFIG FINAL ML + VALIDATION SCRIPT")
    print("==============================\n")

    # Ensure figure dirs exist
    ensure_dir(FIG_BASE_DIR)
    ensure_dir(FIG_TRAIN_DIR)
    ensure_dir(FIG_DEV_DIR)
    ensure_dir(FIG_TEST_DIR)
    ensure_dir(FIG_SHAP_DIR)

    # ------------------------------
    # 1) Load TRAIN / DEV / TEST
    # ------------------------------
    X_tr, y_tr, meta_tr, rec_tr, feat_names = load_features_split(TRAIN_PATH)
    X_dev, y_dev, meta_dev, rec_dev, feat_names_dev = load_features_split(DEV_PATH)
    X_te, y_te, meta_te, rec_te, feat_names_te = load_features_split(TEST_PATH)

    # sanity: feature names consistent (just warning, not fatal)
    if feat_names_dev != feat_names or feat_names_te != feat_names:
        warnings.warn(
            "Feature name lists differ across splits; "
            "assuming same ordering but names differ."
        )

    # subject IDs
    subj_tr_rec = subject_ids_from_meta(meta_tr)
    subj_dev_rec = subject_ids_from_meta(meta_dev)
    subj_te_rec = subject_ids_from_meta(meta_te)

    subj_tr_win = subject_ids_per_window(rec_tr, subj_tr_rec)
    subj_dev_win = subject_ids_per_window(rec_dev, subj_dev_rec)
    subj_te_win = subject_ids_per_window(rec_te, subj_te_rec)

    print("\n[Info] TRAIN vs DEV vs TEST window counts:")
    print(f"  TRAIN: X={X_tr.shape}, y sum (positives)={int(y_tr.sum())}")
    print(f"  DEV  : X={X_dev.shape}, y sum (positives)={int(y_dev.sum())}")
    print(f"  TEST : X={X_te.shape}, y sum (positives)={int(y_te.sum())}")

    # ------------------------------
    # 2) Feature selection (d + SUNSET) + TRAIN plots
    # ------------------------------
    sel_idx, sel_names, d, abs_d, top_indices = select_features_by_d_and_sunset(
        X_tr, y_tr, feat_names, top_k=TOP_K_BY_ABS_D
    )

    X_tr_sel = X_tr[:, sel_idx]
    X_dev_sel = X_dev[:, sel_idx]
    X_te_sel = X_te[:, sel_idx]

    print(
        f"\n[Info] After feature selection: "
        f"TRAIN X={X_tr_sel.shape}, DEV X={X_dev_sel.shape}, TEST X={X_te_sel.shape}"
    )

    # Build GT / non-GT masks on the selected features
    sunset_full = sunset_feature_mask(feat_names)   # full feature space
    sunset_sel = sunset_full[sel_idx]               # restricted to selected

    idx_all_sel = np.arange(len(sel_idx))
    idx_gt      = idx_all_sel[sunset_sel]
    idx_non_gt  = idx_all_sel[~sunset_sel]

    print(f"[ConfigPrep] Selected features total: {len(sel_idx)}")
    print(f"[ConfigPrep]   GT/graph/plaw subset: {len(idx_gt)}")
    print(f"[ConfigPrep]   Non-GT/plaw subset : {len(idx_non_gt)}")

    configs = {}

    # NO_GT_PLAW: remove GT/graph/plaw from selected set
    if len(idx_non_gt) > 0:
        Xtr_A  = X_tr_sel[:, idx_non_gt]
        Xdev_A = X_dev_sel[:, idx_non_gt]
        Xte_A  = X_te_sel[:, idx_non_gt]
        names_A = [sel_names[int(i)] for i in idx_non_gt]
        configs["NO_GT_PLAW"] = (Xtr_A, Xdev_A, Xte_A, names_A)
    else:
        print("[WARN] NO_GT_PLAW config is empty; skipping.")

    # ONLY_GT_PLAW: only GT/graph/plaw
    if len(idx_gt) > 0:
        Xtr_B  = X_tr_sel[:, idx_gt]
        Xdev_B = X_dev_sel[:, idx_gt]
        Xte_B  = X_te_sel[:, idx_gt]
        names_B = [sel_names[int(i)] for i in idx_gt]
        configs["ONLY_GT_PLAW"] = (Xtr_B, Xdev_B, Xte_B, names_B)
    else:
        print("[WARN] ONLY_GT_PLAW config is empty; skipping.")

    # FULL_SUITE: everything selected
    configs["FULL_SUITE"] = (X_tr_sel, X_dev_sel, X_te_sel, sel_names)

    dev_probs_for_multi = {}
    test_probs_for_multi = {}
    config_results = {}

    # ------------------------------
    # 3) Run each feature configuration
    # ------------------------------
    for cfg_name, (Xtr_cfg, Xdev_cfg, Xte_cfg, names_cfg) in configs.items():
        res = run_config(
            cfg_name,
            Xtr_cfg, Xdev_cfg, Xte_cfg,
            y_tr, y_dev, y_te,
            subj_tr_win, subj_dev_win, subj_te_win,
            names_cfg,
        )
        dev_probs_for_multi[cfg_name] = res["y_dev_prob"]
        test_probs_for_multi[cfg_name] = res["y_te_prob"]
        config_results[cfg_name] = res

    # ------------------------------
    # 4) Multi-curve PR & ROC on DEV (compare configs)
    # ------------------------------
    if dev_probs_for_multi:
        plot_multi_pr_roc(
            y_dev,
            dev_probs_for_multi,
            "DEV_MULTI_CONFIG",
            FIG_DEV_DIR
        )
    else:
        print("[Multi] No configs produced dev probabilities; skipping multi PR/ROC.")

    # (Optional) Multi-curve PR & ROC on TEST
    if test_probs_for_multi:
        plot_multi_pr_roc(
            y_te,
            test_probs_for_multi,
            "TEST_MULTI_CONFIG",
            FIG_TEST_DIR
        )


    # ------------------------------
    # 5) Optional SHAP on FULL_SUITE best tree model
    # ------------------------------
    if DO_SHAP and "FULL_SUITE" in config_results:
        full_res = config_results["FULL_SUITE"]
        best_name = full_res["best_model_name"]
        best_model = full_res["best_model"]
        feat_names_full = full_res["feat_names_cfg"]
        X_dev_full = configs["FULL_SUITE"][1]

        if best_name in ("XGB", "RandomForest"):
            print(f"\n[SHAP] Running SHAP analysis on DEV for FULL_SUITE ({best_name})...")
            run_shap_on_dev(best_model, X_dev_full, y_dev, feat_names_full,
                            model_name=f"FULL_SUITE_{best_name}")
        else:
            print("\n[SHAP] FULL_SUITE best model is not tree-based; skipping SHAP.")
    else:
        print("\n[SHAP] Skipping SHAP (disabled or FULL_SUITE not available).")


if __name__ == "__main__":
    main()
