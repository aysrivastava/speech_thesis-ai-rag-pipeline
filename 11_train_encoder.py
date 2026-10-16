"""
11_train_encoder.py
----------------------
Train the small sequence encoder (see encoder_utils.py) and measure how well
it works on clips it has NOT seen, using the same retrieval protocol as the
real system: held-out clip -> embedding -> nearest stored clip -> its label.

Two cross-validation modes (default: both)
  clip    5 folds; each fold holds out ~1 clip of every sign.
  signer  holds out all clips of one 'signer id' at a time. The id is guessed
          from the filename: "name (3).MP4" -> 3, "name 1.MP4" -> 1, otherwise 1.
          THIS IS AN ASSUMPTION. If your numbering isn't per-signer, the
          signer-CV number is only an approximation - say so in the thesis.

Then:
  python3 11_train_encoder.py --final      trains on ALL clips and saves encoder/encoder.pt
After --final, switch the system to the trained embedding:
  1. config.py:  EMBEDDING_VERSION = 4
  2. python3 03_converting_extractions_to_vector_embeddings.py --force
  3. python3 04_hosting_vector_db_and_storing.py --rebuild

Examples
  python3 11_train_encoder.py                  # CV only (about a few minutes)
  python3 11_train_encoder.py --epochs 80      # quicker
  python3 11_train_encoder.py --final --skip_cv    # just train on everything and save (about 2 minutes)
"""

import argparse
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np

import config
import encoder_utils as eu


def load_clips():
    d = config.PATHS["features_extracted"]
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".pkl"):
            with open(os.path.join(d, f), "rb") as fh:
                r = pickle.load(fh)
            out.append((r["label"], r.get("source_video", f), r["features"]))
    return out


def parse_signer(source_video):
    stem = os.path.splitext(os.path.basename(source_video))[0]
    m = re.search(r"\((\d+)\)\s*$", stem) or re.search(r"\s(\d+)\s*$", stem)
    return int(m.group(1)) if m else 1


def clip_folds(labels, k=5):
    per = defaultdict(list)
    for i, l in enumerate(labels):
        per[l].append(i)
    folds = [[] for _ in range(k)]
    for l, idx in per.items():
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [f for f in folds if f]


def signer_folds(sources):
    ids = [parse_signer(s) for s in sources]
    groups = defaultdict(list)
    for i, s in enumerate(ids):
        groups[s].append(i)
    return [v for k, v in sorted(groups.items()) if len(v) >= 5], dict(Counter(ids))


def retrieval_accuracy(bundle, feats, labels, test_idx, train_idx, log=None):
    """Same protocol as the real system: score each stored LABEL by its best-matching clip, rank labels.
    Returns (top1_correct, top3_correct, evaluated); appends (true, predicted_top1) to `log`."""
    E = {i: eu.embed_with(bundle, feats[i]) for i in train_idx}
    both = getattr(config, "QUERY_BOTH_ORIENTATIONS", True)
    Q = {t: (eu.embed_both(bundle, feats[t]) if both else eu.embed_with(bundle, feats[t])[None]) for t in test_idx}
    c1 = c3 = evaluated = 0
    train_labels = set(labels[i] for i in train_idx)
    for t in test_idx:
        if labels[t] not in train_labels:
            continue
        evaluated += 1
        best = {}
        for i in train_idx:
            sc = float((Q[t] @ E[i]).max())
            if sc > best.get(labels[i], -9):
                best[labels[i]] = sc
        ranked = sorted(best, key=best.get, reverse=True)
        c1 += ranked[0] == labels[t]
        c3 += labels[t] in ranked[:3]
        if log is not None:
            log.append((labels[t], ranked[0]))
    return c1, c3, evaluated


def run_cv(name, folds, feats, labels, epochs):
    tot_c = tot_c3 = tot_n = 0
    log = []
    for k, test_idx in enumerate(folds, 1):
        test_set = set(test_idx)
        train_idx = [i for i in range(len(feats)) if i not in test_set]
        t0 = time.time()
        bundle = eu.train_encoder([feats[i] for i in train_idx], [labels[i] for i in train_idx], epochs=epochs, seed=k)
        c, c3, n = retrieval_accuracy(bundle, feats, labels, test_idx, train_idx, log)
        tot_c += c; tot_c3 += c3; tot_n += n
        print(f"   [{name}] fold {k}/{len(folds)}: top-1 {c}/{n} = {100*c/max(n,1):.1f}%   top-3 {100*c3/max(n,1):.1f}%   ({time.time()-t0:.0f}s)")
    acc = tot_c / max(tot_n, 1)
    print(f"   [{name}] OVERALL: top-1 {tot_c}/{tot_n} = {100*acc:.1f}%   top-3 = {100*tot_c3/max(tot_n,1):.1f}%")
    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for true, pred in log:
        per[true][1] += 1
        per[true][0] += true == pred
        if true != pred:
            conf[(true, pred)] += 1
    print(f"\n   [{name}] per-sign top-1 (worst first):")
    for lab, (c, n_) in sorted(per.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        print(f"      {lab:<28}{c}/{n_}")
    print(f"   [{name}] most common confusions (true -> predicted):")
    for (t, p_), n_ in conf.most_common(6):
        print(f"      {t}  ->  {p_}   x{n_}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--cv", choices=["clip", "signer", "both", "none"], default="both")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip_cv", action="store_true", help="With --final: skip the cross-validation and just train and save.")
    args = ap.parse_args()
    if args.skip_cv:
        args.cv = "none"

    config.ensure_dirs()
    clips = load_clips()
    if len(clips) < 10:
        print("Not enough feature files. Run step 01 first.")
        return
    labels = [c[0] for c in clips]
    sources = [c[1] for c in clips]
    feats = [c[2] for c in clips]
    n_classes = len(set(labels))
    print("=" * 78)
    print(f"ENCODER TRAINING: {len(clips)} clips, {n_classes} signs (chance = {100/n_classes:.1f}%), epochs={args.epochs}")
    print("=" * 78)

    if args.cv in ("clip", "both"):
        print("\nCross-validation A: hold out ~1 clip of every sign per fold")
        run_cv("clip", clip_folds(labels), feats, labels, args.epochs)
    if args.cv in ("signer", "both"):
        folds, sizes = signer_folds(sources)
        print(f"\nCross-validation B: hold out one 'signer id' at a time (ASSUMED from filenames; ids seen: {sizes})")
        if len(folds) >= 2:
            run_cv("signer", folds, feats, labels, args.epochs)
        else:
            print("   not enough distinct ids - skipped")

    if args.final:
        print("\nTraining final encoder on ALL clips...")
        bundle = eu.train_encoder(feats, labels, epochs=args.epochs, seed=0, verbose=True)
        path = eu.save_bundle(bundle)
        print(f"Saved: {path}")
        print("\nNext:\n  1) set EMBEDDING_VERSION = 4 in config.py"
              "\n  2) python3 03_converting_extractions_to_vector_embeddings.py --force"
              "\n  3) python3 04_hosting_vector_db_and_storing.py --rebuild")


if __name__ == "__main__":
    main()
