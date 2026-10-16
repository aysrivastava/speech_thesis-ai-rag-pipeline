"""
STEP 12 (optional): pick the N signs from a bigger pool that are easiest to tell apart.

Use it when you have more candidate signs than you need (e.g. 97 labels) and want the N that a
demo is most likely to get right: each sign must be retrievable on its own AND must not look
like the other signs you pick.

Layout of --videos (same as data_raw_videos): one FOLDER per sign, folder name = exact label.
   <videos>/Who are you/clip1.mp4, clip2.mp4 ...
(flat files like Thank_You.mp4 also work; label comes from the file name.)

What it does
  1. Extracts MediaPipe landmarks for every clip into selection_cache/ (incremental; your
     features_extracted/ and vector DB are NOT touched).
  2. Cross-validation over the WHOLE pool: trains the encoder on some clips, retrieves the
     held-out ones. Per sign it records: held-out top-1/top-3, and margin = (similarity to its own
     sign) - (similarity to the best wrong sign).
  3. Pairwise overlap: how often sign A's held-out clips come close to / land on sign B.
  4. Greedy pick: --wh question signs (what/where/who/how...) first, then the rest, maximising
     retrievability while penalising overlap with signs already picked.
  5. Re-runs a fresh cross-validation on only the picked signs and prints it.
  6. Optionally copies the picked signs (all their clips) into a folder with --copy_to.

Honest limits (say this in the thesis if you use it):
  * Signs with only 1 clip cannot be held out, so they cannot be scored; they are listed and skipped.
  * The picked set is chosen using the same clips it is then scored on, so its accuracy is
    optimistic. It is a "demo vocabulary", not a result about ISL. Report the full-pool
    numbers (step 3 below) as the scientific result and the picked set as the demo.

Run:
   python3 12_select_demo_signs.py --videos "<PATH>"                       # score + pick
   python3 12_select_demo_signs.py --videos "<PATH>" --copy_to data_raw_videos_demo
   Useful flags: --n 20 --wh 4 --epochs 100 --max_mb 3 --cache_only
"""
import argparse
import importlib
import json
import os
import pickle
import re
import shutil
import time
from collections import Counter, defaultdict

import numpy as np

import config
import encoder_utils as eu

ex = importlib.import_module("01_extracting_features_from_video_using_mediapipe")
tr = importlib.import_module("11_train_encoder")

WH_RE = re.compile(r"\b(what|where|who|whom|whose|why|when|which|how)\b", re.I)
CACHE = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "selection_cache")


def stem_of(rel):
    return os.path.splitext(rel)[0].replace(os.sep, "__").replace(" ", "_")


def load_pool(video_dir, max_mb, cache_only):
    os.makedirs(CACHE, exist_ok=True)
    clips = []                                   # (label, rel_path, features, hand_rate)
    if cache_only:
        for f in sorted(os.listdir(CACHE)):
            if f.endswith(".pkl"):
                r = pickle.load(open(os.path.join(CACHE, f), "rb"))
                clips.append((r["label"], r["source_video"], r["features"], r.get("hand_detect_rate", 1.0)))
        return clips
    from landmark_extraction_utils import extract_landmark_sequence, check_model_files_exist
    if not check_model_files_exist():
        raise SystemExit("MediaPipe model files missing (see README).")
    found = ex.find_labelled_videos(video_dir)
    if not found:
        raise SystemExit(f"No videos found in {video_dir}")
    small = 0
    for i, (label, rel) in enumerate(found, 1):
        path = os.path.join(video_dir, rel)
        if max_mb and os.path.getsize(path) < max_mb * 1024 * 1024:
            small += 1
            continue
        out = os.path.join(CACHE, stem_of(rel) + ".pkl")
        if os.path.isfile(out):
            r = pickle.load(open(out, "rb"))
        else:
            print(f"[{i}/{len(found)}] extracting '{label}' ({rel})")
            res = extract_landmark_sequence(path, verbose=False)
            r = {"label": label, "source_video": rel, "features": res["features"],
                 "hand_detect_rate": res["hand_detect_rate"]}
            pickle.dump(r, open(out, "wb"))
        clips.append((label, rel, r["features"], r["hand_detect_rate"]))
    if small:
        print(f"Skipped {small} clip(s) smaller than {max_mb} MB (--max_mb).")
    return clips


def heldout_scores(feats, labels, epochs, k=5, seed0=0):
    """Returns list of (true_label, {label: best_sim}) for every held-out clip."""
    cnt = Counter(labels)
    evaluable = [i for i, l in enumerate(labels) if cnt[l] >= 2]
    folds = tr.clip_folds([labels[i] for i in evaluable], k)
    folds = [[evaluable[j] for j in f] for f in folds]
    out = []
    for fi, test_idx in enumerate(folds, 1):
        ts = set(test_idx)
        train_idx = [i for i in range(len(feats)) if i not in ts]
        t0 = time.time()
        bundle = eu.train_encoder([feats[i] for i in train_idx], [labels[i] for i in train_idx],
                                  epochs=epochs, seed=seed0 + fi)
        E = {i: eu.embed_with(bundle, feats[i]) for i in train_idx + test_idx}
        train_labels = set(labels[i] for i in train_idx)
        for t in test_idx:
            if labels[t] not in train_labels:
                continue
            best = {}
            for i in train_idx:
                s = float(E[t] @ E[i])
                if s > best.get(labels[i], -9):
                    best[labels[i]] = s
            out.append((labels[t], best))
        print(f"   fold {fi}/{len(folds)} done ({time.time()-t0:.0f}s)")
    return out


def summarise(res):
    per = defaultdict(lambda: {"n": 0, "t1": 0, "t3": 0, "margins": [], "over": defaultdict(list)})
    for true, best in res:
        d = per[true]
        ranked = sorted(best, key=best.get, reverse=True)
        d["n"] += 1
        d["t1"] += ranked[0] == true
        d["t3"] += true in ranked[:3]
        wrong = [best[l] for l in best if l != true]
        d["margins"].append(best[true] - max(wrong) if wrong else 0.0)
        for l, s in best.items():
            if l != true:
                d["over"][l].append(s - best[true])
    stats = {}
    for l, d in per.items():
        stats[l] = {"n": d["n"], "top1": d["t1"] / d["n"], "top3": d["t3"] / d["n"],
                    "margin": float(np.mean(d["margins"])),
                    "over": {o: float(np.mean(v)) for o, v in d["over"].items()}}
        stats[l]["score"] = stats[l]["top1"] + stats[l]["margin"]
    return stats


def sym_overlap(stats, a, b):
    return max(stats[a]["over"].get(b, 0.0), stats[b]["over"].get(a, 0.0))


def greedy(pool, k, chosen, stats, mu):
    chosen = list(chosen)
    for _ in range(k):
        best_c, best_v = None, -1e9
        for c in pool:
            if c in chosen:
                continue
            pen = max([sym_overlap(stats, c, s) for s in chosen], default=0.0)
            v = stats[c]["score"] - mu * pen
            if v > best_v:
                best_c, best_v = c, v
        if best_c is None:
            break
        chosen.append(best_c)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="", help="candidate pool folder (one folder per sign)")
    ap.add_argument("--n", type=int, default=20, help="how many signs to pick")
    ap.add_argument("--wh", type=int, default=4, help="of those, how many what/where/who/how questions")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--mu", type=float, default=2.0, help="overlap penalty weight")
    ap.add_argument("--max_mb", type=float, default=0.0,
                    help="skip clips smaller than this (uncontrolled-signer clips were < ~1.3 MB; use 3)")
    ap.add_argument("--copy_only", action="store_true",
                    help="skip scoring; just copy the signs already listed in selected_signs.json to --copy_to")
    ap.add_argument("--cache_only", action="store_true", help="use selection_cache/ as is, no extraction")
    ap.add_argument("--copy_to", default="", help="copy the picked signs' clips into this folder")
    args = ap.parse_args()
    if not args.videos and not args.cache_only:
        raise SystemExit('Pass --videos "<folder with one sub-folder per sign>"')

    if args.copy_only:
        chosen_set = set(json.load(open("selected_signs.json"))["picked"])
        n = 0
        for label, rel in ex.find_labelled_videos(args.videos):
            if label in chosen_set and (not args.max_mb or os.path.getsize(os.path.join(args.videos, rel)) >= args.max_mb * 1048576):
                os.makedirs(os.path.join(args.copy_to, label), exist_ok=True)
                shutil.copy2(os.path.join(args.videos, rel), os.path.join(args.copy_to, label, os.path.basename(rel)))
                n += 1
        raise SystemExit(f"Copied {n} clips for {len(chosen_set)} signs into {args.copy_to}/")

    print("=" * 74)
    print("STEP 12: SELECT THE MOST RETRIEVABLE, LEAST-OVERLAPPING SIGNS")
    print("=" * 74)
    clips = load_pool(args.videos, args.max_mb, args.cache_only)
    labels = [c[0] for c in clips]
    feats = [c[2] for c in clips]
    cnt = Counter(labels)
    single = sorted(l for l, n in cnt.items() if n < 2)
    print(f"\nPool: {len(clips)} clips, {len(cnt)} signs, clips per sign {min(cnt.values())}-{max(cnt.values())}")
    if single:
        print(f"{len(single)} sign(s) have only 1 clip and cannot be scored (skipped): {single[:10]}{' ...' if len(single) > 10 else ''}")

    print("\n[1/3] Cross-validating the whole pool...")
    res = heldout_scores(feats, labels, args.epochs)
    stats = summarise(res)
    n_ev = sum(s["n"] for s in stats.values())
    print(f"\nFULL-POOL held-out: top-1 {100*sum(s['top1']*s['n'] for s in stats.values())/n_ev:.1f}%  "
          f"top-3 {100*sum(s['top3']*s['n'] for s in stats.values())/n_ev:.1f}%  "
          f"({len(stats)} signs, chance {100/len(stats):.1f}%)  <- report THIS as the scientific number")

    ranked = sorted(stats, key=lambda l: stats[l]["score"], reverse=True)
    print("\nTop 15 individually most retrievable signs:")
    for l in ranked[:15]:
        s = stats[l]
        print(f"   {l:<34} top1 {s['top1']*100:4.0f}%  top3 {s['top3']*100:4.0f}%  margin {s['margin']:+.3f}  (n={s['n']})")

    print("\n[2/3] Picking signs...")
    wh_pool = [l for l in stats if WH_RE.search(l)]
    other_pool = [l for l in stats if not WH_RE.search(l)]
    wh_n = min(args.wh, len(wh_pool), args.n)
    chosen = greedy(wh_pool, wh_n, [], stats, args.mu)
    chosen = greedy(other_pool, args.n - len(chosen), chosen, stats, args.mu)
    if len(chosen) < args.n:
        chosen = greedy([l for l in stats if l not in chosen], args.n - len(chosen), chosen, stats, args.mu)
    print(f"\nPICKED {len(chosen)} signs ({wh_n} question signs first):")
    for l in chosen:
        s = stats[l]
        nearest = max((o for o in chosen if o != l), key=lambda o: sym_overlap(stats, l, o), default="-")
        tag = "Q" if WH_RE.search(l) else " "
        print(f"   [{tag}] {l:<32} top1 {s['top1']*100:4.0f}%  margin {s['margin']:+.3f}   closest picked rival: {nearest}")

    print("\n[3/3] Fresh cross-validation on ONLY the picked signs...")
    idx = [i for i, l in enumerate(labels) if l in set(chosen)]
    sub = heldout_scores([feats[i] for i in idx], [labels[i] for i in idx], args.epochs, seed0=100)
    ss = summarise(sub)
    n2 = sum(s["n"] for s in ss.values())
    t1 = sum(s["top1"] * s["n"] for s in ss.values()) / n2
    t3 = sum(s["top3"] * s["n"] for s in ss.values()) / n2
    print(f"\nPICKED-SET held-out: top-1 {100*t1:.1f}%  top-3 {100*t3:.1f}%  "
          f"(chance {100/len(ss):.1f}%)  <- OPTIMISTIC: signs were chosen on these same clips")

    out = {"picked": chosen, "full_pool": {l: {k: v for k, v in s.items() if k != "over"} for l, s in stats.items()}}
    json.dump(out, open("selected_signs.json", "w"), indent=2)
    print("\nWrote selected_signs.json")

    if args.copy_to:
        chosen_set = set(chosen)
        n = 0
        for label, rel, _, _ in clips:
            if label in chosen_set:
                dst = os.path.join(args.copy_to, label)
                os.makedirs(dst, exist_ok=True)
                shutil.copy2(os.path.join(args.videos, rel), os.path.join(dst, os.path.basename(rel)))
                n += 1
        print(f"Copied {n} clips for {len(chosen)} signs into {args.copy_to}/ (originals untouched).")
    else:
        print("\nTo copy them:  python3 12_select_demo_signs.py --videos <PATH> --copy_only --copy_to data_raw_videos_demo")


if __name__ == "__main__":
    main()
