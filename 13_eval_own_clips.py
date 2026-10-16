"""
STEP 13: test YOUR OWN recordings (data_raw_videos2/) against the vector DB built from data_raw_videos/.

  data_raw_videos/    original clips (untouched) -> 01 / 03 / 04 -> vector DB
  data_raw_videos2/   your own takes, SAME sign labels, one folder per sign -> tested here, never added to the DB

For every clip it reports whether the right sign comes out top-1 / top-3, in two modes:
  auto : the clip is embedded once (automatic dominant-hand choice)
  both : embedded as signed AND mirrored (left<->right hand), better match kept   <- what the demo uses
If 'both' beats 'auto', the left-hand/right-hand difference was part of the problem. If neither does well,
the cause is something else (framing, distance, lighting, signer speed) - see the per-sign hand-detection rate.

Run:
   python3 13_eval_own_clips.py
   python3 13_eval_own_clips.py --cache_only      # reuse features_extracted2/ without re-running MediaPipe
   python3 13_eval_own_clips.py --force           # re-extract every clip
"""
import argparse
import importlib
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

import config
from embedding_utils import compute_motion_pooled_embedding, compute_query_embeddings
import retrieval_utils as ru

ex = importlib.import_module("01_extracting_features_from_video_using_mediapipe")


def stem_of(rel):
    return os.path.splitext(rel)[0].replace(os.sep, "__").replace(" ", "_")


def load_clips(video_dir, cache_dir, force, cache_only):
    os.makedirs(cache_dir, exist_ok=True)
    clips = []
    if cache_only:
        for f in sorted(os.listdir(cache_dir)):
            if f.endswith(".pkl"):
                r = pickle.load(open(os.path.join(cache_dir, f), "rb"))
                clips.append(r)
        return clips
    from landmark_extraction_utils import extract_landmark_sequence, check_model_files_exist
    if not check_model_files_exist():
        raise SystemExit("MediaPipe model files missing (see README).")
    found = ex.find_labelled_videos(video_dir)
    if not found:
        raise SystemExit(f"No clips in {video_dir}. Put them in {video_dir}/<exact sign label>/take1.mp4 ...")
    for i, (label, rel) in enumerate(found, 1):
        out = os.path.join(cache_dir, stem_of(rel) + ".pkl")
        if os.path.isfile(out) and not force:
            clips.append(pickle.load(open(out, "rb")))
            continue
        print(f"[{i}/{len(found)}] extracting '{label}' ({rel})")
        res = extract_landmark_sequence(os.path.join(video_dir, rel), verbose=False)
        r = {"label": label, "source_video": rel, "features": res["features"],
             "hand_detect_rate": res["hand_detect_rate"]}
        pickle.dump(r, open(out, "wb"))
        clips.append(r)
    return clips


def rank_labels(query_vectors, stored):
    sims = np.max([ru.centered_similarities(qv, stored) for qv in query_vectors], axis=0)
    best = {}
    for item, s in zip(stored, sims):
        if s > best.get(item["label"], -9):
            best[item["label"]] = float(s)
    return sorted(best, key=best.get, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache_only", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("STEP 13: TEST YOUR OWN CLIPS (data_raw_videos2) AGAINST THE ORIGINAL DB")
    print("=" * 74)
    config.ensure_dirs()
    stored = ru.load_stored_vectors()
    if not stored:
        raise SystemExit("Vector DB is empty. Run 01, 11 --final, 03 --force, 04 --rebuild on data_raw_videos first.")
    db_labels = sorted(set(s["label"] for s in stored))
    clips = load_clips(config.PATHS["data_raw_videos2"], config.PATHS["features_extracted2"], args.force, args.cache_only)

    missing = sorted(set(c["label"] for c in clips) - set(db_labels))
    if missing:
        print(f"\n[WARN] {len(missing)} folder name(s) in data_raw_videos2 match NO sign in the DB (typo?) and are skipped:")
        for m in missing:
            print(f"   - '{m}'")
    clips = [c for c in clips if c["label"] in db_labels]
    if not clips:
        raise SystemExit("No usable clips.")

    dim = stored[0]["embedding"].shape[0]
    res = {"auto": [], "both": []}
    per = defaultdict(lambda: {"n": 0, "auto1": 0, "both1": 0, "both3": 0, "hand": []})
    conf = Counter()
    for c in clips:
        lab = c["label"]
        v_auto = compute_motion_pooled_embedding(c["features"], verbose=False)
        if v_auto.shape[0] != dim:
            raise SystemExit(f"Embedding size {v_auto.shape[0]} != DB size {dim}. Rebuild: 03 --force, 04 --rebuild.")
        r_auto = rank_labels([v_auto], stored)
        r_both = rank_labels(compute_query_embeddings(c["features"], verbose=False), stored)
        p = per[lab]
        p["n"] += 1
        p["auto1"] += r_auto[0] == lab
        p["both1"] += r_both[0] == lab
        p["both3"] += lab in r_both[:3]
        p["hand"].append(c.get("hand_detect_rate", float("nan")))
        if r_both[0] != lab:
            conf[(lab, r_both[0])] += 1

    n = sum(p["n"] for p in per.values())
    a1 = sum(p["auto1"] for p in per.values())
    b1 = sum(p["both1"] for p in per.values())
    b3 = sum(p["both3"] for p in per.values())
    print(f"\nYour clips: {n} clips, {len(per)} signs; DB has {len(db_labels)} signs (chance {100/len(db_labels):.1f}%)")
    print(f"   auto  (one orientation)   : top-1 {a1}/{n} = {100*a1/n:.1f}%")
    print(f"   both  (as signed+mirrored): top-1 {b1}/{n} = {100*b1/n:.1f}%   top-3 {100*b3/n:.1f}%   <- demo setting")
    print(f"\n{'sign':<36}{'auto':>6}{'both':>6}{'top3':>6}{'hands':>8}")
    for lab, p in sorted(per.items(), key=lambda kv: kv[1]["both1"] / kv[1]["n"]):
        hr = np.nanmean(p["hand"]) if p["hand"] else float("nan")
        print(f"{lab:<36}{p['auto1']}/{p['n']:<4}{p['both1']}/{p['n']:<4}{p['both3']}/{p['n']:<4}{100*hr:7.0f}%")
    if conf:
        print("\nMost common mistakes (your sign -> what the system returned):")
        for (t, pr), k in conf.most_common(8):
            print(f"   {t}  ->  {pr}   x{k}")
    low = [l for l, p in per.items() if np.nanmean(p['hand']) < 0.4]
    if low:
        print(f"\nNOTE: hands were detected in <40% of frames for {len(low)} sign(s) ({', '.join(low[:5])}...). "
              "That is a framing/lighting/distance problem, not a left/right problem - re-record those.")


if __name__ == "__main__":
    main()
