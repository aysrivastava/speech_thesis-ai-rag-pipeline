"""
08_diagnose_retrieval.py
---------------------------
Answers "is my retrieval actually working, and why not?" using ONLY the
feature files from step 01 (no MediaPipe, no DB; takes a minute).

For every clip it asks: if this clip were a brand-new query, and every OTHER
clip were the database, which label would win? (Leave-one-clip-out.)

Columns:
  top-1 acc      all clips
  acc (good)     only clips where MediaPipe saw a hand on >=70% of frames
                 (query AND database restricted to those). If this is far
                 higher than 'top-1 acc', hand-detection quality is the
                 bottleneck, not the embedding.
  sim(diff/same) mean cosine between clips of different / same label.
                 You want same >> diff.

Rows: several embeddings x with/without mean-centering, plus a DTW 1-NN
row. DTW (dynamic time warping) aligns two signs frame-by-frame in time and
is the classic upper bound for "keep the order of the sign". It is NOT a
vector-DB method - it is here to tell you whether a vector embedding that
respects time order CAN work on your data.

Run:   python3 08_diagnose_retrieval.py
"""

import os
import pickle
from collections import Counter

import numpy as np

import config
from embedding_utils import (compute_embedding_v1, compute_embedding_v2, compute_embedding_v3,
                             frame_features_v2, _active_segment)

GOOD_HAND_RATE = 0.7


def load_features():
    d = config.PATHS["features_extracted"]
    items = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".pkl"):
            with open(os.path.join(d, f), "rb") as fh:
                r = pickle.load(fh)
            items.append((r["label"], r.get("source_video", f), r["features"], r.get("hand_detect_rate", 1.0)))
    return items


def _cos_matrix(E, center):
    E = np.asarray(E, dtype=np.float64)
    if center:
        E = E - E.mean(axis=0)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(n, 1e-12)
    return E @ E.T


def evaluate_similarity(labels, S):
    """Leave-one-clip-out top-1 accuracy from a similarity matrix S (higher = more similar)."""
    labels = np.array(labels)
    S = S.copy().astype(np.float64)
    np.fill_diagonal(S, -np.inf)
    n = len(labels)
    correct, evaluated, wrong_winners = 0, 0, Counter()
    for i in range(n):
        if (labels == labels[i]).sum() < 2:
            continue
        evaluated += 1
        j = int(np.argmax(S[i]))
        if labels[j] == labels[i]:
            correct += 1
        else:
            wrong_winners[labels[j]] += 1
    Sf = np.where(np.isfinite(S), S, np.nan)
    diff = labels[:, None] != labels[None, :]
    same = (labels[:, None] == labels[None, :]) & ~np.eye(n, dtype=bool)
    return {
        "acc": correct / evaluated if evaluated else float("nan"),
        "evaluated": evaluated,
        "mean_diff": float(np.nanmean(np.where(diff, Sf, np.nan))),
        "mean_same": float(np.nanmean(np.where(same, Sf, np.nan))) if same.any() else float("nan"),
        "hubs": wrong_winners.most_common(3),
    }


def evaluate(labels, E, center, good_mask=None):
    r = evaluate_similarity(labels, _cos_matrix(E, center))
    r["acc_good"] = float("nan")
    if good_mask is not None and good_mask.sum() >= 20:
        idx = np.where(good_mask)[0]
        Eg = np.asarray(E)[idx]
        r["acc_good"] = evaluate_similarity([labels[i] for i in idx], _cos_matrix(Eg, center))["acc"]
    return r


# ---------------------------------------------------------------- DTW baseline
def _prep_for_dtw(features, max_len=30):
    ff = frame_features_v2(features, hand_invariant=True)
    a, b = _active_segment(ff)
    ff = ff[a:b]
    idx = np.linspace(0, ff.shape[0] - 1, min(max_len, ff.shape[0])).round().astype(int)
    return ff[idx]


def _dtw(a, b):
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return D[n, m] / (n + m)


def dtw_similarity_matrix(items):
    seqs = [_prep_for_dtw(f) for _, _, f, _ in items]
    allf = np.concatenate(seqs)
    sd = allf.std(axis=0)
    sd[sd < 1e-6] = 1.0
    seqs = [s / sd for s in seqs]
    n = len(seqs)
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Dm[i, j] = Dm[j, i] = _dtw(seqs[i], seqs[j])
    return -Dm


def build_variants(items):
    feats = [f for _, _, f, _ in items]
    V = {
        "v1": [compute_embedding_v1(f, verbose=False) for f in feats],
        "v2": [compute_embedding_v2(f, verbose=False) for f in feats],
        "v2 hand-inv": [compute_embedding_v2(f, verbose=False, hand_invariant=True) for f in feats],
        "v3 trajectory K=12": [compute_embedding_v3(f, verbose=False, use_stats=False, K=12) for f in feats],
        "v3 trajectory K=24": [compute_embedding_v3(f, verbose=False, use_stats=False, K=24) for f in feats],
        "v3 trajectory K=6": [compute_embedding_v3(f, verbose=False, use_stats=False, K=6) for f in feats],
        "v3 traj pose-only": [compute_embedding_v3(f, verbose=False, use_stats=False, K=12, parts="pose") for f in feats],
        "v3 traj hands-only": [compute_embedding_v3(f, verbose=False, use_stats=False, K=12, parts="hands") for f in feats],
        "v3 stats+traj": [compute_embedding_v3(f, verbose=False, K=12) for f in feats],
    }
    return V


def run(items, with_dtw=False):
    items = [it if len(it) == 4 else (*it, 1.0) for it in items]
    labels = [it[0] for it in items]
    good = np.array([it[3] >= GOOD_HAND_RATE for it in items])
    rows = []
    for name, E in build_variants(items).items():
        for center in (False, True):
            rows.append((f"{name} {'+ctr' if center else ''}".strip(), evaluate(labels, E, center, good)))
    if with_dtw:
        S = dtw_similarity_matrix(items)
        r = evaluate_similarity(labels, S)
        r["acc_good"] = float("nan")
        if good.sum() >= 20:
            idx = np.where(good)[0]
            r["acc_good"] = evaluate_similarity([labels[i] for i in idx], S[np.ix_(idx, idx)])["acc"]
        r["mean_diff"] = r["mean_same"] = float("nan")
        rows.append(("DTW 1-NN (not vecDB)", r))
    return rows


def main():
    items = load_features()
    if len(items) < 3:
        print("Need the feature files from step 01 (features_extracted/). Run step 01 first.")
        return
    labels = [it[0] for it in items]
    good = sum(1 for it in items if it[3] >= GOOD_HAND_RATE)
    n_labels = len(set(labels))
    print("=" * 92)
    print(f"RETRIEVAL DIAGNOSTIC: {len(items)} clips, {n_labels} labels, {Counter(labels).most_common(1)[0][1]} max clips/label; "
          f"{good} clips have hands on >= {int(GOOD_HAND_RATE*100)}% of frames")
    print("=" * 92)
    print("(DTW comparison takes about a minute...)")
    rows = run(items, with_dtw=True)
    print(f"\n{'setting':<26}{'top-1 acc':>10}{'acc (good)':>12}{'sim diff':>10}{'sim same':>10}   hubs")
    for name, r in rows:
        hubs = ", ".join(f"{h}x{c}" for h, c in r["hubs"]) or "-"
        ag = f"{r['acc_good']*100:>10.1f}%" if r["acc_good"] == r["acc_good"] else f"{'n/a':>11}"
        sd = f"{r['mean_diff']:>10.3f}{r['mean_same']:>10.3f}" if r["mean_diff"] == r["mean_diff"] else f"{'':>20}"
        print(f"{name:<26}{r['acc']*100:>9.1f}%{ag}{sd}   {hubs}")
    chance = 1.0 / n_labels
    print(f"\nChance level with {n_labels} labels = {chance*100:.1f}%.")
    vec_rows = [x for x in rows if not x[0].startswith("DTW")]
    best = max(vec_rows, key=lambda x: np.nan_to_num(x[1]["acc"], nan=-1))
    print(f"Best vector setting: {best[0]}  ({best[1]['acc']*100:.1f}%)")
    print("\nHow to read it:")
    print(" - 'acc (good)' >> 'top-1 acc'  => hand-detection quality is the bottleneck (fix extraction / re-record).")
    print(" - DTW >> best vector row       => time order matters and a better vector embedding is possible/needed.")
    print(" - DTW also low                 => the landmarks themselves do not separate these signs across clips;")
    print("                                   look at the clips (signers, framing, mislabeled folders) before modelling.")
    print(f"\nCurrently configured: embedding v{getattr(config, 'EMBEDDING_VERSION', '?')}, "
          f"hand_invariant={getattr(config, 'HAND_INVARIANT', '?')}, "
          f"centering={'on' if getattr(config, 'RETRIEVAL_CENTERING', False) else 'off'}")


if __name__ == "__main__":
    main()
