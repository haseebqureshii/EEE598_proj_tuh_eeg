import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()

TRAIN_NPZ = BASE_OUT_DIR / "train" / "split_windows.npz"
EVAL_NPZ  = BASE_OUT_DIR / "eval"  / "split_windows.npz"

# Subject-level filtering thresholds (for TRAIN/DEV and TEST)
MIN_TOTAL_WINDOWS = 8   # subject must have at least this many windows
MIN_POS_WINDOWS   = 2    # subject must have at least this many positive (label=1) windows

# Desired internal split sizes (TRAIN/DEV) and TEST size (subjects)
TRAIN_DEV_RATIO = 0.80   # 80% train, 20% dev (on the seizure-rich subjects)
N_TEST_SUBJ     = 5     # desired number of subjects for TEST_INTERNAL


# ---------------------------------------------------------
# helper for meta_list (handles list/array of dicts)
# ---------------------------------------------------------
def load_meta_list(meta_array):
    """
    Convert the 'meta' array loaded from NPZ into a plain Python list of dicts.
    Handles cases where it's stored as an object array.
    """
    if isinstance(meta_array, list):
        return meta_array
    out = []
    for m in meta_array:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(m.item())
    return out


# ---------------------------------------------------------
# helper: subset arrays/meta/rec_idx by subject set
# ---------------------------------------------------------
def subset_by_subjects(
    X_all,
    y_all,
    meta_list,
    rec_idx_all,
    subj_id_per_rec,
    chosen_subjects,
):
    """
    Keep only windows whose subject_id is in chosen_subjects.
    Returns X_sub, y_sub, meta_sub, rec_idx_sub (re-indexed recordings).
    """
    chosen_subjects = set(chosen_subjects)

    # recording indices for chosen subjects
    chosen_rec_indices = [
        r_idx for r_idx, sid in enumerate(subj_id_per_rec)
        if sid in chosen_subjects
    ]
    chosen_rec_set = set(chosen_rec_indices)

    # mask windows whose rec_idx is in chosen_rec_indices
    win_mask = np.array(
        [r_idx in chosen_rec_set for r_idx in rec_idx_all],
        dtype=bool,
    )

    X_sub = X_all[win_mask]
    y_sub = y_all[win_mask]
    old_rec_idx_sub = rec_idx_all[win_mask]

    # remap rec_idx to [0..len(chosen_rec_indices)-1]
    old_to_new = {old: new for new, old in enumerate(chosen_rec_indices)}
    rec_idx_sub = np.array([old_to_new[ri] for ri in old_rec_idx_sub], dtype=np.int32)

    meta_sub = [meta_list[old] for old in chosen_rec_indices]

    print(f"[Split] subset_by_subjects → {X_sub.shape[0]} windows, {len(meta_sub)} recordings.")
    return X_sub, y_sub, meta_sub, rec_idx_sub


# ---------------------------------------------------------
# helper: summary per split (for sanity + Methods section)
# ---------------------------------------------------------
def print_split_summary(split_name, y, rec_idx, subj_id_per_rec):
    """
    Print:
      - #subjects
      - #windows
      - #positive/negative windows
      - #subjects with ≥1 seizure window vs 0-seizure subjects
    """
    subj_ids = np.array([subj_id_per_rec[i] for i in rec_idx])
    unique_subjects = sorted(set(subj_ids))

    n_subj = len(unique_subjects)
    n_win  = len(y)
    n_pos  = int((y == 1).sum())
    n_neg  = int((y == 0).sum())

    subj_with_seizure = 0
    for sid in unique_subjects:
        mask = (subj_ids == sid)
        if (y[mask] == 1).any():
            subj_with_seizure += 1
    subj_no_seizure = n_subj - subj_with_seizure

    print(f"\n[Summary] {split_name}:")
    print(f"  #subjects                  : {n_subj}")
    print(f"  #windows                   : {n_win}")
    print(f"  #positive (seizure) windows: {n_pos}")
    print(f"  #negative windows          : {n_neg}")
    print(f"  subjects with ≥1 seizure   : {subj_with_seizure}")
    print(f"  subjects with 0 seizures   : {subj_no_seizure}")


# =========================================================
# === PART A: TRAIN → internal TRAIN / DEV (80/20) =========
# =========================================================
print("\n===============================")
print(" Loading TRAIN for 80/20 split ")
print("===============================\n")

data_train = np.load(TRAIN_NPZ, allow_pickle=True)

X_all       = data_train["X"]
y_all       = data_train["y"]
meta_list   = load_meta_list(data_train["meta"])
rec_idx_all = data_train["rec_idx"]
ch_names    = data_train["ch_names"] if "ch_names" in data_train.files else None

# subject IDs per recording (from meta)
subj_id_per_rec = [m.get("subject_id", f"subj_{i}") for i, m in enumerate(meta_list)]
# subject IDs per window
subj_id_per_win = np.array([subj_id_per_rec[i] for i in rec_idx_all])

unique_subjects = sorted(set(subj_id_per_win))
print(f"[Split] Found {len(unique_subjects)} subjects in TRAIN.")


# ---- compute per-subject stats based on labels (not just meta)
subj_stats = {}
for sid in unique_subjects:
    mask = (subj_id_per_win == sid)
    y_s  = y_all[mask]
    n_win_s = int(mask.sum())
    n_pos_s = int((y_s == 1).sum())
    n_neg_s = int((y_s == 0).sum())
    subj_stats[sid] = {
        "n_win": n_win_s,
        "n_pos": n_pos_s,
        "n_neg": n_neg_s,
    }

# ---- SUBJECT FILTERING LOGIC (TRAIN/DEV) ----
# We require each subject to have:
#   - at least MIN_TOTAL_WINDOWS total windows
#   - at least MIN_POS_WINDOWS positive (preictal) windows
good_subjects = [
    sid for sid in unique_subjects
    if (subj_stats[sid]["n_win"] >= MIN_TOTAL_WINDOWS
        and subj_stats[sid]["n_pos"] >= MIN_POS_WINDOWS)
]

print(
    f"\n[Split] Subjects passing filters for TRAIN/DEV "
    f"(>= {MIN_POS_WINDOWS} positive window(s) & >= {MIN_TOTAL_WINDOWS} total windows): "
    f"{len(good_subjects)}"
)

if len(good_subjects) < 2:
    print("[WARN] Very few subjects pass filters for TRAIN/DEV. "
          "You may want to relax MIN_POS_WINDOWS or MIN_TOTAL_WINDOWS.")

# Sort seizure subjects by number of positive windows (descending)
good_sorted = sorted(
    good_subjects,
    key=lambda sid: -subj_stats[sid]["n_pos"]
)

n_total = len(good_sorted)
if n_total >= 2:
    # 80/20 split on these seizure-rich subjects, but ensure at least 1 dev subject
    n_dev_subj = max(1, int(round(n_total * (1.0 - TRAIN_DEV_RATIO))))
    n_dev_subj = min(n_dev_subj, n_total - 1)  # leave at least 1 for train
    n_train_subj = n_total - n_dev_subj
else:
    n_train_subj = n_total
    n_dev_subj   = 0

dev_subjects   = good_sorted[:n_dev_subj]
train_subjects = good_sorted[n_dev_subj:]

print(f"[Split] Internal TRAIN subjects: {len(train_subjects)}")
print(f"[Split] Internal DEV subjects  : {len(dev_subjects)}")
print(f"[Split] TRAIN subject IDs: {train_subjects}")
print(f"[Split] DEV subject  IDs: {dev_subjects}")

# ---- subset TRAIN and DEV
X_train_int, y_train_int, meta_train_int, rec_idx_train_int = subset_by_subjects(
    X_all, y_all, meta_list, rec_idx_all, subj_id_per_rec, train_subjects
)

X_dev_int, y_dev_int, meta_dev_int, rec_idx_dev_int = subset_by_subjects(
    X_all, y_all, meta_list, rec_idx_all, subj_id_per_rec, dev_subjects
)

# ---- split summaries for TRAIN/DEV ----
print_split_summary("TRAIN_INTERNAL", y_train_int, rec_idx_train_int, subj_id_per_rec)
print_split_summary("DEV_INTERNAL",   y_dev_int,   rec_idx_dev_int,   subj_id_per_rec)


# =========================================================
# === PART B: EVAL → internal TEST (10 seizure subjects) ===
# =========================================================
print("\n===============================")
print(" Loading EVAL for TEST split   ")
print("===============================\n")

data_eval = np.load(EVAL_NPZ, allow_pickle=True)

X_eval        = data_eval["X"]
y_eval        = data_eval["y"]
meta_eval     = load_meta_list(data_eval["meta"])
rec_idx_eval  = data_eval["rec_idx"]
ch_names_eval = data_eval["ch_names"] if "ch_names" in data_eval.files else ch_names

subj_id_per_rec_eval = [
    m.get("subject_id", f"evalsubj_{i}") for i, m in enumerate(meta_eval)
]
subj_id_per_win_eval = np.array([subj_id_per_rec_eval[i] for i in rec_idx_eval])

unique_eval_subj = sorted(set(subj_id_per_win_eval))
print(f"[Split] Found {len(unique_eval_subj)} subjects in EVAL.")


# ---- compute per-subject stats for EVAL
eval_stats = {}
for sid in unique_eval_subj:
    mask = (subj_id_per_win_eval == sid)
    y_s   = y_eval[mask]
    n_win = int(mask.sum())
    n_pos = int((y_s == 1).sum())

    eval_stats[sid] = {
        "n_win": n_win,
        "n_pos": n_pos,
    }

# EVAL subjects with at least MIN_POS_WINDOWS positives & MIN_TOTAL_WINDOWS windows
seizure_eval_subjects = [
    sid for sid in unique_eval_subj
    if (eval_stats[sid]["n_win"] >= MIN_TOTAL_WINDOWS
        and eval_stats[sid]["n_pos"] >= MIN_POS_WINDOWS)
]

print(
    f"[Split] EVAL subjects with >= {MIN_POS_WINDOWS} positive window(s) "
    f"and >= {MIN_TOTAL_WINDOWS} windows: {len(seizure_eval_subjects)}"
)

if not seizure_eval_subjects:
    print("[WARN] No EVAL subjects meet seizure/window criteria for TEST. "
          "Falling back to largest-window subjects (may include zero-seizure).")
    # fallback: just sort all by windows
    eval_sorted = sorted(unique_eval_subj,
                         key=lambda s: -eval_stats[s]["n_win"])
    test_subjects = eval_sorted[:N_TEST_SUBJ]
else:
    # sort seizure_eval_subjects by number of positive windows (descending)
    seizure_eval_sorted = sorted(
        seizure_eval_subjects,
        key=lambda s: -eval_stats[s]["n_pos"]
    )
    if len(seizure_eval_sorted) >= N_TEST_SUBJ:
        test_subjects = seizure_eval_sorted[:N_TEST_SUBJ]
    else:
        # take all seizure subjects, then fill remaining with next best by n_win
        test_subjects = seizure_eval_sorted[:]
        remaining = N_TEST_SUBJ - len(test_subjects)

        non_chosen = [s for s in unique_eval_subj if s not in test_subjects]
        non_chosen_sorted = sorted(
            non_chosen,
            key=lambda s: -eval_stats[s]["n_win"]
        )
        test_subjects.extend(non_chosen_sorted[:remaining])

print(f"[Split] TEST subjects chosen (from EVAL): {len(test_subjects)}")
print("[Split] TEST subject IDs:", test_subjects)

# ---- subset EVAL → TEST_INTERNAL
X_test_int, y_test_int, meta_test_int, rec_idx_test_int = subset_by_subjects(
    X_eval, y_eval, meta_eval, rec_idx_eval, subj_id_per_rec_eval, test_subjects
)

# ---- summary for TEST ----
print_split_summary("TEST_INTERNAL", y_test_int, rec_idx_test_int, subj_id_per_rec_eval)


# =========================================================
# === PART C: SAVE ALL RESULTS =============================
# =========================================================
(train_int_dir := BASE_OUT_DIR / "train_internal").mkdir(parents=True, exist_ok=True)
(dev_int_dir   := BASE_OUT_DIR / "dev_internal").mkdir(parents=True, exist_ok=True)
(test_int_dir  := BASE_OUT_DIR / "test_internal").mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    train_int_dir / "split_windows.npz",
    X=X_train_int, y=y_train_int, meta=meta_train_int,
    ch_names=ch_names, rec_idx=rec_idx_train_int
)
np.savez_compressed(
    dev_int_dir / "split_windows.npz",
    X=X_dev_int, y=y_dev_int, meta=meta_dev_int,
    ch_names=ch_names, rec_idx=rec_idx_dev_int
)
np.savez_compressed(
    test_int_dir / "split_windows.npz",
    X=X_test_int, y=y_test_int, meta=meta_test_int,
    ch_names=ch_names_eval, rec_idx=rec_idx_test_int
)

print("\n===============================")
print(" Saved: train_internal / dev_internal / test_internal")
print("===============================\n")
