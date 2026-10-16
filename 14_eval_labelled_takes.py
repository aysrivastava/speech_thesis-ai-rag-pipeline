"""
STEP 14: score your own labelled webcam takes and test whether adding your own takes to the DB helps.

Takes are the sign-named files made by record_webcam_and_test.py:  test_query_videos/<sign_slug>__<timestamp>.mp4
(record a few per sign with menu [1]; the sign number you pick is only used for the filename).

Two numbers are printed:
  A. DB only            : each take is searched against the original data_raw_videos vectors.
  B. DB + your OTHER takes: each take is searched against the DB plus every other take of the same sign set
                           (leave-one-take-out, so a take never matches itself). Needs >= 2 takes of a sign to help.
If B >> A, the problem is webcam domain shift and enrolling your own takes is the fix (as the demo should do).

Run:  python3 14_eval_labelled_takes.py            (add --force to re-extract)
"""
import argparse, importlib, os, pickle
from collections import Counter, defaultdict
import numpy as np
import config
from embedding_utils import compute_query_embeddings, compute_motion_pooled_embedding
import retrieval_utils as ru


def slugify_label(label):
    import re
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

ex = importlib.import_module("01_extracting_features_from_video_using_mediapipe")


def rank(qvs, stored):
    sims = np.max([ru.centered_similarities(qv, stored) for qv in qvs], axis=0)
    best = {}
    for it, s in zip(stored, sims):
        best[it["label"]] = max(best.get(it["label"], -9), float(s))
    return sorted(best, key=best.get, reverse=True), best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--after", default=None, help="only takes recorded at/after this stamp, e.g. 20261011_191200")
    args = ap.parse_args()
    config.ensure_dirs()
    from landmark_extraction_utils import extract_landmark_sequence, check_model_files_exist
    stored = ru.load_stored_vectors()
    if not stored:
        raise SystemExit("Vector DB empty - build it first.")
    slug2label = {slugify_label(l): l for l in set(s["label"] for s in stored)}
    qdir = config.PATHS["test_query_videos"]
    cache = os.path.join(qdir, "_features"); os.makedirs(cache, exist_ok=True)
    takes = []
    for f in sorted(os.listdir(qdir)):
        if not f.lower().endswith((".mp4", ".mov")) or "__" not in f:
            continue
        slug = f.split("__")[0]
        if slug not in slug2label:
            continue
        if args.after and os.path.splitext(f.split("__", 1)[1])[0] < args.after:
            continue
        pk = os.path.join(cache, os.path.splitext(f)[0] + ".pkl")
        if os.path.isfile(pk) and not args.force:
            feats = pickle.load(open(pk, "rb"))
        else:
            if not check_model_files_exist():
                raise SystemExit("MediaPipe model files missing.")
            print("extracting", f)
            feats = extract_landmark_sequence(os.path.join(qdir, f), verbose=False)["features"]
            pickle.dump(feats, open(pk, "wb"))
        takes.append({"label": slug2label[slug], "name": f, "feats": feats})
    if not takes:
        raise SystemExit(f"No sign-named takes in {qdir}. Record with record_webcam_and_test.py menu [1] and pick the sign number.")
    for t in takes:
        t["qvs"] = compute_query_embeddings(t["feats"], verbose=False)
        t["enrol"] = {"label": t["label"], "source_video": t["name"],
                      "embedding": compute_motion_pooled_embedding(t["feats"], verbose=False)}
    A = B = A3 = B3 = 0; conf = Counter(); per = defaultdict(lambda: [0, 0, 0])
    for i, t in enumerate(takes):
        ra, _ = rank(t["qvs"], stored)
        rb, _ = rank(t["qvs"], stored + [o["enrol"] for j, o in enumerate(takes) if j != i])
        A += ra[0] == t["label"]; B += rb[0] == t["label"]
        A3 += t["label"] in ra[:3]; B3 += t["label"] in rb[:3]
        p = per[t["label"]]; p[0] += 1; p[1] += ra[0] == t["label"]; p[2] += rb[0] == t["label"]
        if ra[0] != t["label"]:
            conf[(t["label"], ra[0])] += 1
    n = len(takes)
    print(f"\n{n} labelled takes, {len(per)} signs (DB has {len(slug2label)} signs)")
    print(f"A. DB only             : top-1 {A}/{n} = {100*A/n:.0f}%   top-3 {100*A3/n:.0f}%")
    print(f"B. DB + your other takes: top-1 {B}/{n} = {100*B/n:.0f}%   top-3 {100*B3/n:.0f}%")
    print(f"\n{'sign':<36}{'takes':>6}{'A':>5}{'B':>5}")
    for l, (k, a, b) in sorted(per.items()):
        print(f"{l:<36}{k:>6}{a:>5}{b:>5}")
    if conf:
        print("\nMost common mistakes with DB only (signed -> returned):")
        for (a, b), k in conf.most_common(8):
            print(f"   {a} -> {b}  x{k}")


if __name__ == "__main__":
    main()
