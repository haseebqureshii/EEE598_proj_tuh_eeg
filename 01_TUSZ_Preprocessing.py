import os
import io
import re
import random
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import firwin, filtfilt, resample_poly

# ------------------ USER CONFIG ------------------

# Root of TUSZ EDF folders that contain train/dev/eval
ROOT_DIR = Path(r"../../../../../../../../../../../../TUH-Seizure-Corpus/edf").expanduser().resolve()

# Output base directory for split_windows.npz + plots
BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map TUH splits -> names your FE/ML scripts expect
SPLIT_SAVE_NAME = {
    "train": "train",
    "dev":   "dev",
    "eval":  "eval",   # eval -> test_internal
}

# Which splits to process
SPLITS_TO_RUN = ["train", "eval"]

# Windowing params
WIN_LEN_S = 30    # window length in seconds
HOP_S     = 15    # hop in seconds (50% overlap)

# --- Prediction labeling policy (all times in SECONDS) ---
SOP_MIN = 15 * 60           # pre-ictal span BEFORE onset (start = onset - SOP_MIN)
SPH_MIN = 5 * 60            # prediction horizon BEFORE onset, ignored (start = onset - SPH_MIN)
EARLY_ICTAL_KEPT_S = 0      # portion of ictal after onset treated as NEG (0)
POSTICTAL_BUFFER_S = 2*60    # post-ictal ignored after offset; 0 disables
KEEP_PREICTAL_EVEN_IF_ARTIFACT = False  # keep/drop pre-ictal windows that overlap artifacts

# DSP params
NUM_TAPS  = 1001            # Kaiser FIR taps for 0.5–40 Hz bandpass
TARGET_FS = 250.0           # downsample to 250 Hz if fs_orig > 250; never upsample

# Runtime limiting params
MAX_SUBJECTS   = 303        # per-split; None = no explicit cap
MAX_RECORDINGS = 20         # max recordings per subject; None = no explicit cap
GROUP_MODE     = "subject"  # "subject" or "recording"

# Seizure subtype labels considered "seizure" (label=1)
SEIZURE_LABELS = {
    "seiz", "fnsz", "gnsz", "spsz", "cpsz", "absz",
    "tnsz", "cnsz", "tcsz", "atsz", "mysz",
}

# Artifact / noise labels in TUSZ annotations
ARTIFACT_LABELS = {
    "artf", "eyem", "chew", "shiv", "musc", "elec",
    "eyem_chew", "eyem_shiv", "eyem_musc", "eyem_elec",
    "chew_shiv", "chew_musc", "chew_elec",
    "shiv_musc", "shiv_elec",
    "musc_elec",
}

print("ROOT_DIR:", ROOT_DIR)
print("BASE_OUT_DIR:", BASE_OUT_DIR)


# =========================================================
# 1) LE → TCP montage helpers
# =========================================================

TCP_PAIRS = [
    # Left hemisphere
    ("FP1", "F7"),
    ("F7",  "T3"),   # a.k.a. T7
    ("T3",  "T5"),   # a.k.a. P7
    ("T5",  "O1"),
    ("FP1", "F3"),
    ("F3",  "C3"),
    ("C3",  "P3"),
    ("P3",  "O1"),
    # Right hemisphere
    ("FP2", "F8"),
    ("F8",  "T4"),   # a.k.a. T8
    ("T4",  "T6"),   # a.k.a. P8
    ("T6",  "O2"),
    ("FP2", "F4"),
    ("F4",  "C4"),
    ("C4",  "P4"),
    ("P4",  "O2"),
    # Midline
    ("FZ",  "CZ"),
    ("CZ",  "PZ"),
    # Additional temporal-parietal pairs (optional, often used in TCP)
    ("T3",  "C3"),
    ("C3",  "CZ"),
    ("CZ",  "C4"),
    ("C4",  "T4"),
]


def _base_electrode_from_le_name(ch_name: str) -> str:
    """
    From an LE channel label like 'EEG FP1-LE' or 'FP1-LE' or 'Fp1-le',
    extract the electrode name: 'FP1'.
    """
    n = ch_name.upper().strip()
    n = re.sub(r"^EEG\s+", "", n)
    # split at '-' and take the first token (e.g., FP1-LE -> FP1)
    n = n.split("-")[0]
    return n


def _is_le_montage(ch_names) -> bool:
    """
    Heuristic: is this a referential montage (LE/REF/A1/A2)?
    Treat both '-LE' and '-REF' (plus A1/A2) as LE-style references.
    """
    upper = [c.upper().strip() for c in ch_names]
    le_like = [
        c for c in upper
        if c.endswith("-LE")
        or c.endswith("-REF")
        or c.endswith("-A1")
        or c.endswith("-A2")
    ]
    # Require at least ~10 referential channels to call it LE/REF montage
    return len(le_like) >= 10


def apply_tcp_montage_if_le(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    If the EDF is in LE montage (channels like FP1-LE, F7-LE, ...),
    convert to a 22-channel TCP bipolar montage using (anode-LE) - (cathode-LE).

    If EDF is already bipolar (e.g., FP1-F7, F7-T3), we leave it unchanged.
    """
    ch_names = raw.ch_names

    # If already looks bipolar (FP1-F7 style) with no '-LE', skip
    if (not _is_le_montage(ch_names)) and any("-" in c and not c.upper().endswith("-LE") for c in ch_names):
        print("       [Montage] Appears already bipolar (TCP-like). Skipping LE->TCP.")
        return raw

    if not _is_le_montage(ch_names):
        print("       [Montage] Not recognized as LE; leaving channels as-is.")
        return raw

    print("       [Montage] Detected LE montage; building TCP 22-ch bipolar montage...")

    # Map electrode name -> channel index of its LE channel
    elec_to_idx = {}
    for idx, ch in enumerate(ch_names):
        base = _base_electrode_from_le_name(ch)
        if base not in elec_to_idx:
            elec_to_idx[base] = idx

    data_le = raw.get_data()  # [C_raw, N]
    sfreq = raw.info["sfreq"]
    n_times = data_le.shape[1]

    tcp_data = []
    tcp_names = []

    for anode, cathode in TCP_PAIRS:
        ch_a_idx = elec_to_idx.get(anode)
        ch_c_idx = elec_to_idx.get(cathode)

        if ch_a_idx is None or ch_c_idx is None:
            # If one of electrodes is missing, use zeros (we prefer shape consistency)
            print(f"       [WARN] Missing LE channels for pair {anode}-{cathode}; filling zeros.")
            tcp_row = np.zeros(n_times, dtype=float)
        else:
            tcp_row = data_le[ch_a_idx, :] - data_le[ch_c_idx, :]
        tcp_data.append(tcp_row)
        tcp_names.append(f"{anode}-{cathode}")

    tcp_data = np.vstack(tcp_data)  # [22, N]

    # Create a new RawArray with TCP montage
    info = mne.create_info(
        ch_names=tcp_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(tcp_names),
    )
    raw_tcp = mne.io.RawArray(tcp_data, info)
    print(f"       [Montage] TCP data shape: {raw_tcp.get_data().shape}, channels={tcp_names}")
    return raw_tcp


# =========================================================
# 2) Filtering / resampling / normalization helpers
# =========================================================

def _kaiser_beta_from_ripple(ripple_db: float) -> float:
    A = ripple_db
    if A > 50:
        return 0.1102 * (A - 8.7)
    elif A >= 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        return 0.0


def apply_kaiser_bandpass(data, fs_orig, f_lo=0.5, f_hi=40.0, num_taps=101, ripple_db=60.0):
    n_ch, n_samp = data.shape
    padlen_needed = 3 * (num_taps - 1)
    if n_samp <= padlen_needed:
        raise ValueError(f"too short for filtfilt len={n_samp} need>{padlen_needed}")

    beta = _kaiser_beta_from_ripple(ripple_db)
    taps = firwin(
        numtaps=num_taps,
        cutoff=[f_lo, f_hi],
        pass_zero=False,
        window=("kaiser", beta),
        fs=fs_orig,
        scale=True,
    )
    filtered = np.stack([filtfilt(taps, [1.0], ch, axis=0) for ch in data], axis=0)
    return filtered


def maybe_resample_to_250(data, fs_orig, target_fs=TARGET_FS):
    if abs(fs_orig - target_fs) < 1e-6:
        return data, fs_orig
    if fs_orig < target_fs:
        print(f"[WARN] fs {fs_orig} Hz < {target_fs} Hz, not upsampling.")
        return data, fs_orig

    up = int(target_fs * 1000)
    down = int(fs_orig * 1000)
    g = gcd(up, down)
    up //= g
    down //= g

    resampled = [resample_poly(ch, up, down) for ch in data]
    resampled = np.stack(resampled, axis=0)
    return resampled, target_fs


def robust_zscore(x, axis=1, eps_floor=1e-6, std_frac=0.1):
    med = np.median(x, axis=axis, keepdims=True)
    q1 = np.percentile(x, 25, axis=axis, keepdims=True)
    q3 = np.percentile(x, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    std = np.std(x, axis=axis, keepdims=True)
    denom = np.maximum(iqr, std_frac * std)
    denom = np.maximum(denom, eps_floor)
    return (x - med) / denom


# =========================================================
# 3) Annotation loader (CSV) – seizures & artifacts
# =========================================================

def _parse_annotation_csv(csv_path: Path):
    csv_path = Path(csv_path)
    try:
        raw_text = csv_path.read_text(errors="replace")
    except Exception as e:
        print(f"[WARN] Could not open {csv_path}: {e}")
        return None

    lines = raw_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ("start" in low) and ("stop" in low):
            header_idx = i
            break

    if header_idx is None:
        try:
            ann = pd.read_csv(csv_path, sep=",", engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"WARNING: Could not parse {csv_path}: {e}")
            return None
    else:
        main_body = "\n".join(lines[header_idx:])
        try:
            ann = pd.read_csv(io.StringIO(main_body), sep=",", engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"WARNING: Could not smart-parse {csv_path}: {e}")
            return None

    if ann is None or ann.empty:
        return None

    ann.columns = [c.strip().lower() for c in ann.columns]
    return ann


def _extract_intervals_from_ann_df(ann_df: pd.DataFrame, target_label_set: set[str]):
    if ann_df is None or ann_df.empty:
        return []

    if "start_time" in ann_df.columns:
        start_col = "start_time"
    elif "start" in ann_df.columns:
        start_col = "start"
    else:
        return []

    if "stop_time" in ann_df.columns:
        stop_col = "stop_time"
    elif "stop" in ann_df.columns:
        stop_col = "stop"
    else:
        return []

    if "label" in ann_df.columns:
        label_col = "label"
    elif "type" in ann_df.columns:
        label_col = "type"
    else:
        return []

    tmp = ann_df.copy()
    tmp[label_col] = tmp[label_col].astype(str).str.lower().str.strip()
    use_rows = tmp[tmp[label_col].isin(target_label_set)]
    if use_rows.empty:
        return []

    raw_intervals = []
    for _, r in use_rows.iterrows():
        try:
            s = float(r[start_col])
            e = float(r[stop_col])
        except Exception:
            continue
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        if e <= s:
            continue
        raw_intervals.append([s, e])

    if not raw_intervals:
        return []

    raw_intervals.sort(key=lambda x: x[0])
    merged = [raw_intervals[0]]
    for s, e in raw_intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1][1] = max(le, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def load_seizure_intervals(csv_path: Path):
    ann_df = _parse_annotation_csv(csv_path)
    return _extract_intervals_from_ann_df(ann_df, SEIZURE_LABELS)


def load_artifact_intervals(csv_path: Path):
    ann_df = _parse_annotation_csv(csv_path)
    return _extract_intervals_from_ann_df(ann_df, ARTIFACT_LABELS)


# =========================================================
# 4) Sliding-window extraction + prediction labels
# =========================================================

def window_data_with_artifact(
    data,
    fs,
    seizure_intervals,
    artifact_intervals,
    win_len_s,
    hop_s,
):
    def _merge(iv):
        if not iv:
            return []
        iv = sorted(iv, key=lambda x: x[0])
        merged = [[iv[0][0], iv[0][1]]]
        for s, e in iv[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1][1] = max(le, e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def _regions_prediction(seiz_iv, sop_s, sph_s, early_s, post_s):
        pre, igp, e0, igi, igpo = [], [], [], [], []
        for on, off in seiz_iv:
            if off <= on:
                continue
            pre.append((max(0.0, on - sop_s), max(0.0, on - sph_s)))
            igp.append((max(0.0, on - sph_s), on))
            ei = min(on + early_s, off)
            if ei > on:
                e0.append((on, ei))
            if off > ei:
                igi.append((ei, off))
            if post_s > 0:
                igpo.append((off, off + post_s))
        return {
            "pre":  _merge([x for x in pre  if x[1] > x[0]]),
            "igp":  _merge([x for x in igp  if x[1] > x[0]]),
            "e0":   _merge([x for x in e0   if x[1] > x[0]]),
            "igi":  _merge([x for x in igi  if x[1] > x[0]]),
            "igpo": _merge([x for x in igpo if x[1] > x[0]]),
        }

    def _overlaps(w, ivs):
        ws, we = w
        for s, e in ivs:
            if ws < e and we > s:
                return True
        return False

    def _window_prediction_with_artifacts(
        data, fs, seizure_iv, artifact_iv,
        win_s, hop_s,
        sop_s, sph_s, early_s, post_s,
        keep_pre_artf=True,
    ):
        regs = _regions_prediction(seizure_iv, sop_s, sph_s, early_s, post_s)
        pre, igp, e0, igi, igpo = regs["pre"], regs["igp"], regs["e0"], regs["igi"], regs["igpo"]

        C, N = data.shape
        W = int(round(win_s * fs))
        H = int(round(hop_s * fs))

        Xs, Ys, T0 = [], [], []
        drop_a = drop_i = 0

        i = 0
        while i + W <= N:
            seg = data[:, i : i + W]
            ws, we = i / fs, (i + W) / fs
            w = (ws, we)

            # ignore zones
            if _overlaps(w, igp) or _overlaps(w, igi) or _overlaps(w, igpo):
                drop_i += 1
                i += H
                continue

            # pre-ictal positive
            if _overlaps(w, pre):
                if (not keep_pre_artf) and _overlaps(w, artifact_iv):
                    drop_a += 1
                    i += H
                    continue
                Xs.append(seg); Ys.append(1); T0.append(ws)
                i += H
                continue

            # early ictal negative
            if _overlaps(w, e0):
                if _overlaps(w, artifact_iv):
                    drop_a += 1
                    i += H
                    continue
                Xs.append(seg); Ys.append(0); T0.append(ws)
                i += H
                continue

            # interictal background (unless artifact)
            if _overlaps(w, artifact_iv):
                drop_a += 1
                i += H
                continue

            Xs.append(seg); Ys.append(0); T0.append(ws)
            i += H

        if not Xs:
            return (
                np.empty((0, C, max(1, W)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                drop_a,
                drop_i,
            )

        X = np.stack(Xs, axis=0).astype(np.float32)
        y = np.array(Ys, dtype=np.int64)
        t0 = np.array(T0, dtype=np.float32)
        return X, y, t0, drop_a, drop_i

    X, y, t0s, dropped_art, _ = _window_prediction_with_artifacts(
        data,
        fs,
        seizure_intervals,
        artifact_intervals,
        win_s=WIN_LEN_S,
        hop_s=HOP_S,
        sop_s=SOP_MIN,
        sph_s=SPH_MIN,
        early_s=EARLY_ICTAL_KEPT_S,
        post_s=POSTICTAL_BUFFER_S,
        keep_pre_artf=KEEP_PREICTAL_EVEN_IF_ARTIFACT,
    )
    return X, y, t0s, int(dropped_art)


# =========================================================
# 5) EDF ↔ CSV pairing
# =========================================================

def find_best_annotation_for_edf(edf_path: Path):
    edf_path = Path(edf_path)
    d = edf_path.parent
    stem = edf_path.stem

    exact = d / f"{stem}.csv"
    if exact.exists():
        return exact

    csvs = list(d.glob("*.csv"))
    pref = [c for c in csvs if c.stem.startswith(stem)]
    if pref:
        return pref[0]

    if len(csvs) == 1:
        return csvs[0]

    return None


# =========================================================
# 6) Per-recording preprocessing (includes LE->TCP)
# =========================================================

def preprocess_single_recording_raw(
    edf_path: Path,
    ann_csv_path: Path,
    win_len_s=WIN_LEN_S,
    hop_s=HOP_S,
    num_taps=NUM_TAPS,
):
    print(f"INFO ONLY: EDF: {edf_path.as_posix()}")
    print(f"       CSV: {ann_csv_path.as_posix()}")

    # Load raw EDF
    raw = mne.io.read_raw_edf(edf_path.as_posix(), preload=True, verbose="ERROR")

    # LE -> TCP montage if needed
    raw = apply_tcp_montage_if_le(raw)

    data = raw.get_data()   # [C, N]
    ch_names = raw.ch_names
    fs_orig = float(raw.info["sfreq"])
    n_samp = data.shape[1]
    print(f"       fs_orig={fs_orig} Hz  ch={data.shape[0]}  n_samples={n_samp}")

    padlen_needed = 3 * (num_taps - 1)
    if n_samp <= padlen_needed:
        print(
            f"       [SKIP SHORT] len={n_samp/fs_orig:.1f}s "
            f"(need >{padlen_needed/fs_orig:.1f}s for filtfilt with {num_taps} taps)"
        )
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                "recording_id": edf_path.stem,
                "fs": fs_orig,
                "skipped_short": True,
                "num_windows": 0,
                "num_Pre_Ictal_windows": 0,
                "num_Negative_windows": 0,
                "num_artifact_windows_dropped": 0,
                "total_seizure_seconds_in_file": 0.0,
                "total_artifact_seconds_in_file": 0.0,
                "first_seizure_onset_sec": None,
                "last_seizure_offset_sec": None,
                "raw_ch_names": ch_names,
                "num_channels_raw": len(ch_names),
            },
            None,
        )

    # Bandpass
    data_bp = apply_kaiser_bandpass(
        data,
        fs_orig,
        f_lo=0.5,
        f_hi=40.0,
        num_taps=num_taps,
        ripple_db=60.0,
    )
    print(f"       after bandpass: {data_bp.shape}")

    # Resample if needed
    data_ds, fs_proc = maybe_resample_to_250(data_bp, fs_orig, target_fs=TARGET_FS)
    print(f"       after resample check: {data_ds.shape}, fs_proc={fs_proc}")

    # Robust z-score
    data_norm = robust_zscore(data_ds)
    print(f"       after robust z-score: {data_norm.shape}")

    # Load seizure + artifact intervals
    seizure_intervals = load_seizure_intervals(ann_csv_path)
    artifact_intervals = load_artifact_intervals(ann_csv_path)

    total_sz_sec = sum(e - s for (s, e) in seizure_intervals) if seizure_intervals else 0.0
    total_artf_sec = sum(e - s for (s, e) in artifact_intervals) if artifact_intervals else 0.0

    if seizure_intervals:
        first_seizure_onset_sec = min(s for (s, e) in seizure_intervals)
        last_seizure_offset_sec = max(e for (s, e) in seizure_intervals)
    else:
        first_seizure_onset_sec = None
        last_seizure_offset_sec = None

    print(f"       seizure_intervals={seizure_intervals}")
    print(f"       total seizure sec in file: {total_sz_sec:.2f}")
    print(f"       artifact_intervals={artifact_intervals}")
    print(f"       total artifact sec in file: {total_artf_sec:.2f}")

    # Sliding windows + labels
    X, y, t0s, dropped_art = window_data_with_artifact(
        data_norm,
        fs_proc,
        seizure_intervals,
        artifact_intervals,
        win_len_s=win_len_s,
        hop_s=hop_s,
    )

    print(
        f"       => {X.shape[0]} KEPT windows | "
        f"PI={int(np.sum(y==1))} | "
        f"Ng={int(np.sum(y==0))} | "
        f"dropped_artifact={dropped_art}"
    )

    debug_preview = None
    if X.shape[0] > 0:
        debug_preview = {
            "recording_id": edf_path.stem,
            "all_windows": X,
            "all_labels": y,
            "fs": fs_proc,
            "channel_names": ch_names,
        }

    meta = {
        "recording_id": edf_path.stem,
        "fs": fs_proc,
        "num_windows": int(len(y)),
        "num_Pre_Ictal_windows": int(np.sum(y == 1)),
        "num_Negative_windows": int(np.sum(y == 0)),
        "num_artifact_windows_dropped": int(dropped_art),
        "total_seizure_seconds_in_file": total_sz_sec,
        "total_artifact_seconds_in_file": total_artf_sec,
        "first_seizure_onset_sec": first_seizure_onset_sec,
        "last_seizure_offset_sec": last_seizure_offset_sec,
        "skipped_short": False,
        "raw_ch_names": ch_names,
        "num_channels_raw": data_norm.shape[0],
    }

    return X, y, meta, debug_preview


# =========================================================
# 7) Build splits (subject-wise) with fixed channel count
# =========================================================

def _extract_subject_id(split_root: Path, edf_path: Path) -> str:
    rel = edf_path.relative_to(split_root)
    parts = rel.parts
    if len(parts) < 2:
        return "unknown"
    return parts[0]


def build_split(
    split_root: Path,
    group_mode=GROUP_MODE,
    max_subjects=MAX_SUBJECTS,
    max_recordings=MAX_RECORDINGS,
    win_len_s=WIN_LEN_S,
    hop_s=HOP_S,
    num_taps=NUM_TAPS,
):
    split_root = Path(split_root)
    print(f"\n INFO ONLY ==== Processing split: {split_root.as_posix()} ====")

    edf_files = list(split_root.rglob("*.edf"))
    print(f"INFO ONLY: Found {len(edf_files)} EDF files under {split_root.as_posix()}")

    X_all_list, y_all_list, metas, debug_candidates, rec_idx_all_list = [], [], [], [], []
    C_ref = None  # reference channel count for this split

    def handle_edf_list(edf_iter, subj_id=None):
        nonlocal C_ref
        count_used = 0
        for edf_path in edf_iter:
            if max_recordings is not None and count_used >= max_recordings:
                break

            ann_path = find_best_annotation_for_edf(edf_path)
            if ann_path is None:
                print(
                    "[SKIP NO CSV]",
                    edf_path.as_posix(),
                    "CSV candidates:",
                    [c.name for c in edf_path.parent.glob("*.csv")],
                )
                continue

            X_rec, y_rec, meta_rec, dbg = preprocess_single_recording_raw(
                edf_path,
                ann_path,
                win_len_s=win_len_s,
                hop_s=hop_s,
                num_taps=num_taps,
            )

            if X_rec.shape[0] == 0:
                continue

            # --- enforce consistent channel count across recordings ---
            C_curr = X_rec.shape[1]
            if C_ref is None:
                C_ref = C_curr
                print(f"       [Split] Reference channel count set to {C_ref} from {edf_path.stem}")
            elif C_curr != C_ref:
                print(
                    f"       [SKIP] Channel mismatch for {edf_path.stem}: "
                    f"{C_curr} vs ref {C_ref}"
                )
                continue

            # All recordings are already in TCP / consistent channel order
            X_all_list.append(X_rec)
            y_all_list.append(y_rec)

            meta_with_id = dict(meta_rec)
            if subj_id is not None:
                meta_with_id["subject_id"] = subj_id
            metas.append(meta_with_id)

            meta_idx = len(metas) - 1
            rec_idx_rec = np.full(len(y_rec), meta_idx, dtype=np.int32)
            rec_idx_all_list.append(rec_idx_rec)

            count_used += 1

            if dbg is not None:
                debug_candidates.append(dbg)

    if group_mode == "recording":
        handle_edf_list(edf_files)
    else:
        subj_to_edfs = {}
        for edf_path in edf_files:
            subj_id = _extract_subject_id(split_root, edf_path)
            subj_to_edfs.setdefault(subj_id, []).append(edf_path)

        subjects_sorted = sorted(subj_to_edfs.keys())
        if max_subjects is not None:
            subjects_sorted = subjects_sorted[:max_subjects]
        print(f"[INFO] Subjects used (cap={max_subjects}): {subjects_sorted}")

        for subj_id in subjects_sorted:
            handle_edf_list(subj_to_edfs[subj_id], subj_id=subj_id)

    if len(X_all_list) == 0:
        X_all = np.empty((0, 0, 0), dtype=np.float32)
        y_all = np.empty((0,), dtype=np.int64)
        rec_idx_all = np.empty((0,), dtype=np.int32)
    else:
        X_all = np.concatenate(X_all_list, axis=0)
        y_all = np.concatenate(y_all_list, axis=0)
        rec_idx_all = np.concatenate(rec_idx_all_list, axis=0)

    # pick best debug candidate (max positives, then max windows)
    best_dbg, best_sz, best_total = None, -1, -1
    for dbg in debug_candidates:
        labels = dbg.get("all_labels")
        wins = dbg.get("all_windows")
        if labels is None or wins is None:
            continue
        sz_count = int(np.sum(labels == 1))
        total_w = int(len(labels))
        if sz_count > best_sz or (sz_count == best_sz and total_w > best_total):
            best_dbg, best_sz, best_total = dbg, sz_count, total_w

    return X_all, y_all, metas, best_dbg, rec_idx_all


# =========================================================
# 8) QC window picker
# =========================================================

def pick_informative_window(dbg_dict, motion_thresh=1e-6, iqr_eps=1e-6):
    if dbg_dict is None:
        return None, None
    X_all = dbg_dict.get("all_windows")
    y_all = dbg_dict.get("all_labels")
    if X_all is None or y_all is None or len(y_all) == 0:
        return None, None

    idx_pos = np.where(y_all == 1)[0]
    idx_neg = np.where(y_all == 0)[0]

    for w_i in list(idx_pos) + list(idx_neg):
        W = X_all[w_i]  # [C,T]
        ch_std = np.std(W, axis=1)
        q25 = np.percentile(W, 25, axis=1)
        q75 = np.percentile(W, 75, axis=1)
        ch_iqr = q75 - q25
        ok = np.where((ch_std > motion_thresh) & (ch_iqr > iqr_eps))[0]
        if ok.size:
            ch_i = int(ok[np.argmax(ch_std[ok])])
            return w_i, ch_i
    return None, None


# =========================================================
# 9) Run all splits and save split_windows.npz + plots
# =========================================================

for split_name in SPLITS_TO_RUN:
    split_dir = ROOT_DIR / split_name
    if not split_dir.exists():
        print(f"\n[WARN] Split folder {split_dir.as_posix()} not found. Skipping.")
        continue

    save_split_name = SPLIT_SAVE_NAME.get(split_name, split_name)
    out_dir = (BASE_OUT_DIR / save_split_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(f" Building split: {split_name} -> {save_split_name}")
    print("==============================")

    # Per-split subject caps if you want them different
    if split_name == "train":
        max_subj = MAX_SUBJECTS
    else:
        max_subj = 30  # or smaller, e.g. 50

    X_split, y_split, meta_split, dbg_split, rec_idx_split = build_split(
        split_dir,
        group_mode=GROUP_MODE,
        max_subjects=max_subj,
        max_recordings=MAX_RECORDINGS,
        win_len_s=WIN_LEN_S,
        hop_s=HOP_S,
        num_taps=NUM_TAPS,
    )

    print("\n=== SPLIT SUMMARY ===")
    print(f"Split name   : {save_split_name}")
    print("X_split shape:", getattr(X_split, "shape", None))
    print("y_split shape:", getattr(y_split, "shape", None))

    total_windows = len(y_split)
    num_bg = int(np.sum(y_split == 0)) if total_windows > 0 else 0
    num_pi = int(np.sum(y_split == 1)) if total_windows > 0 else 0
    pct_bg = (100.0 * num_bg / total_windows) if total_windows > 0 else 0.0
    pct_pi = (100.0 * num_pi / total_windows) if total_windows > 0 else 0.0

    print(f"Total windows: {total_windows}")
    print(f"Negative windows (0): {num_bg} ({pct_bg:.2f}%)")
    print(f"Pre-Ictal windows (1): {num_pi} ({pct_pi:.2f}%)")
    print(f"Class ratio (bg : pi) ~ {num_bg}:{num_pi}")

    # ---------- QC plot ----------
    if dbg_split is not None:
        w_i, ch_i = pick_informative_window(dbg_split, motion_thresh=1e-6, iqr_eps=1e-6)
        if w_i is not None:
            X_all_dbg = dbg_split["all_windows"]
            y_all_dbg = dbg_split["all_labels"]
            fs_dbg = float(dbg_split["fs"])
            sig = X_all_dbg[w_i, ch_i, :].astype(float)
            label_dbg = int(y_all_dbg[w_i])
            t_axis = np.arange(sig.shape[0]) / fs_dbg

            med = np.median(sig)
            iqr = np.percentile(sig, 75) - np.percentile(sig, 25)
            std = np.std(sig)
            denom = max(iqr, 0.1 * std, 1e-6)
            sig_plot = np.clip((sig - med) / denom, -8.0, 8.0)

            print(f"[QC] plotting window {w_i} label={label_dbg} channel {ch_i}")
            print("     std(raw):", float(std), "IQR(raw):", float(iqr), "denom_used:", float(denom))

            plt.figure(figsize=(10, 4))
            plt.plot(t_axis, sig_plot)
            plt.xlabel("Time (s)")
            plt.ylabel("Robust z-scored amplitude")
            plt.title(f"{save_split_name} QC window w={w_i} ch={ch_i} label={label_dbg}")
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            qc_path = (out_dir / "qc_window.png").as_posix()
            plt.savefig(qc_path, dpi=300)
            plt.close()
            print("[INFO] Saved QC plot to", qc_path)
        else:
            print("INFO ONLY: No non-flat window/channel found to plot.")
    else:
        print("INFO ONLY: Skipping QC plot; no debug recording found.")

    # ---------- SAVE NPZ + class balance ----------
    if X_split is not None and y_split is not None and len(y_split) > 0:
        out_npz_path = out_dir / "split_windows.npz"

        if dbg_split is not None and "channel_names" in dbg_split:
            ch_names_ref = np.array(dbg_split["channel_names"], dtype=object)
        else:
            ch_names_ref = None

        np.savez_compressed(
            out_npz_path.as_posix(),
            X=X_split,
            y=y_split,
            meta=meta_split,
            ch_names=ch_names_ref,
            rec_idx=rec_idx_split,
        )
        print("INFO ONLY: Saved NPZ to", out_npz_path.as_posix())

        # Class balance bar chart
        num_bg = int(np.sum(y_split == 0))
        num_pi = int(np.sum(y_split == 1))

        plt.figure(figsize=(4, 4))
        plt.bar(["Pre-ictal (1)", "Negative (0)"], [num_pi, num_bg])
        plt.ylabel("Window count")
        plt.title(f"Class distribution in {save_split_name}")
        plt.grid(True, axis="y")
        plt.tight_layout()

        dist_path = (out_dir / "class_balance.png").as_posix()
        plt.savefig(dist_path, dpi=300)
        plt.close()
        print("INFO ONLY: Saved class balance bar chart to", dist_path)
    else:
        print(f"WARNING: Nothing to save / plot class balance for split {save_split_name} (no data).")
