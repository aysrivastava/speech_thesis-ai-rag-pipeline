"""
03_converting_extractions_to_vector_embeddings.py
----------------------------------------------------
STEP 3 of the RAG pipeline: turning each variable-length landmark
sequence (T x 225) into ONE fixed-size vector per clip - the "embedding".

====================================================================
WHICH EMBEDDING MODEL ARE WE USING, AND WHY (read this part)
====================================================================
You asked "choosing a video embedding model to convert the embeddings to
vectors". There's a real decision here, so here's the reasoning, written
down so it can go straight into your thesis methodology section:

Option A - reuse your trained BiLSTM classifier's hidden state as the
embedding. Tempting because it's "free" (already trained), but your old
checkpoint has the HIDDEN=128-vs-256 config mismatch, and more importantly
it would mean this very first RAG sanity test depends on a second,
separately fragile piece of code we haven't touched yet. Bad idea for a
Day 1 "does the plumbing even work" test.

Option B - train a new contrastive/self-supervised embedding model
(SignBERT-style or a contrastive landmark encoder). This is genuinely the
better embedding long-term (it's on your roadmap as Model 1/Model 2), but
it needs a training run, a GPU-friendly batch pipeline, and tuning - too
much to stand between you and just testing "does semantic search work at
all", which is explicitly what you asked to check FIRST (your step 5).

Option C (what this script actually does) - a training-free STATISTICAL
MOTION-POOLING embedding:
    For every one of the 225 landmark dimensions, compute across all
    frames in the clip:
        1. mean position         (where does this point sit, on average)
        2. std of position       (how much does it wander around)
        3. mean frame-to-frame velocity   (how fast does it typically move)
        4. std of velocity               (how erratic/jerky is the motion)
    Concatenating all four statistics for all 225 dims gives a
    225 x 4 = 900-dimensional vector. L2-normalize it, and cosine
    similarity between two such vectors becomes a simple dot product.

This is a real, established technique (pooled/statistical motion features
are a standard simple baseline in action recognition - you're not doing
anything hacky), it needs ZERO training, it runs in milliseconds, and -
crucially - it lets you test the ENTIRE RAG plumbing (steps 3, 4, 5) today,
with a result you can actually reason about by hand. Every one of the 900
numbers has a plain-English meaning, which is exactly what you want while
you're still learning how embeddings behave.

Once you've confirmed with step 5 that retrieval works end-to-end, swapping
this for a trained embedding (Option A or B) later is a LOCALIZED change -
you'd only touch embedding_utils.py. Steps 1, 2, 4 and 5 don't need to
change at all. That's the benefit of keeping RAG stages as separate files.

(The actual embedding function lives in embedding_utils.py, shared with
05_semantic_search_test.py - see that file for why.)

Run:
  python3 "03_converting_extractions_to_vector_embeddings.py"
"""

import argparse
import os
import pickle
import json

import numpy as np

import config
from embedding_utils import compute_motion_pooled_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every clip, ignoring previously saved embeddings.")
    args = parser.parse_args()
    print("=" * 70)
    print("STEP 3: CONVERTING LANDMARK SEQUENCES TO VECTOR EMBEDDINGS")
    print("=" * 70)
    print(f"\nEmbedding method : statistical motion-pooling (training-free)")
    print(f"Embedding size   : {config.EMBEDDING_DIM} dimensions")
    print(f"(see the big comment at the top of this file for why)")

    config.ensure_dirs()

    feat_dir = config.PATHS["features_extracted"]
    feature_files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".pkl"))

    if not feature_files:
        print(f"\n[ERROR] No feature files found in {feat_dir}")
        print("Run 01_extracting_features_from_video_using_mediapipe.py first.")
        return

    print(f"\nFound {len(feature_files)} feature file(s) to embed.")

    vector_dict = {}          # human-readable: label -> LIST of vectors, one per clip/signer
    db_rows = []              # richer records for step 04 (DuckDB)

    # Incremental: keep rows from earlier runs, embed only feature files not seen before.
    rows_path_prev = os.path.join(config.PATHS["embeddings"], "embedding_rows_for_db.pkl")
    done_sources = set()
    if os.path.isfile(rows_path_prev) and not args.force:
        with open(rows_path_prev, "rb") as f:
            db_rows = pickle.load(f)
        done_sources = {r["source_video"] for r in db_rows}
        for r in db_rows:
            vector_dict.setdefault(r["label"], []).append(r["embedding"])
        print(f"[incremental] {len(db_rows)} embedding(s) already saved; only new clips will be embedded.")
    new_count = 0

    for i, fname in enumerate(feature_files, start=1):
        path = os.path.join(feat_dir, fname)
        with open(path, "rb") as f:
            record = pickle.load(f)

        label = record["label"]
        features = record["features"]
        if record.get("source_video", "") in done_sources:
            continue
        new_count += 1

        print(f"\n--- [{i}/{len(feature_files)}] Embedding '{label}' ---")
        embedding = compute_motion_pooled_embedding(features, verbose=True, label=label)

        vector_dict.setdefault(label, []).append(embedding.tolist())
        db_rows.append(
            {
                "label": label,
                "source_video": record.get("source_video", ""),
                "embedding": embedding.tolist(),
                "num_frames": record["num_frames"],
                "duration_sec": record["duration_sec"],
            }
        )

    # ------------------------------------------------------------------
    # 1) The human-readable dictionary export you explicitly asked for:
    #    {"How Are You": [12.34, -3.21, ...], "Thank You": [...]}
    #    Saved both as a .pkl (exact floats, easy to reload in Python)
    #    and a .json with rounded floats (easy to just eyeball/open).
    # ------------------------------------------------------------------
    emb_dir = config.PATHS["embeddings"]

    pkl_path = os.path.join(emb_dir, "label_to_vector.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(vector_dict, f)

    json_preview = {
        label: {"num_examples": len(vecs),
                "first_example_first_10_dims": [round(v, 4) for v in vecs[0][:10]]}
        for label, vecs in vector_dict.items()
    }
    json_preview_path = os.path.join(emb_dir, "label_to_vector_PREVIEW.json")
    with open(json_preview_path, "w") as f:
        json.dump(json_preview, f, indent=2)

    full_json_path = os.path.join(emb_dir, "label_to_vector_full.json")
    with open(full_json_path, "w") as f:
        json.dump(vector_dict, f)

    # 2) The richer record list that step 04 will load into DuckDB.
    rows_path = os.path.join(emb_dir, "embedding_rows_for_db.pkl")
    with open(rows_path, "wb") as f:
        pickle.dump(db_rows, f)

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)
    print(f"New this run: {new_count}. Total embedded {len(db_rows)} clip(s) across {len(vector_dict)} label(s), each as a {config.EMBEDDING_DIM}-dim vector.")
    print(f"\n[save] Full dict (label -> vector)      : {pkl_path}")
    print(f"[save] Quick-look preview (first 10 dims): {json_preview_path}")
    print(f"[save] Full dict as plain JSON            : {full_json_path}")
    print(f"[save] DB-ready rows                      : {rows_path}")

    print("\nQuick look at what got stored (first 6 dims of each vector):")
    for label, vecs in vector_dict.items():
        preview = [round(v, 3) for v in vecs[0][:6]]
        print(f"   '{label}' ({len(vecs)} example(s)): {preview} ...")

    print("\nNext step: run 04_hosting_vector_db_and_storing.py")


if __name__ == "__main__":
    main()
