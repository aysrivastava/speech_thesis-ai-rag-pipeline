"""
04_hosting_vector_db_and_storing.py
--------------------------------------
STEP 4 of the RAG pipeline: hosting a vector database LOCALLY and storing
the embeddings from step 03 in it.

Why DuckDB (and not Pinecone):
  Pinecone is a cloud service - you'd need an API key and your vectors
  would leave your machine, which is overkill for a few dozen/hundred
  sign vectors sitting on your own M2 Mac. DuckDB is a local, file-based
  database (think "SQLite, but with analytics superpowers") - the whole
  database is just one file on disk (vector_db/sign_vectors.duckdb), no
  server, no account, no internet needed. That fits "host a vector
  database locally" exactly.

How similarity search is implemented here:
  DuckDB has an optional extension called "vss" that can do fast
  approximate nearest-neighbor search with an HNSW index - genuinely
  useful once you have millions of vectors. At the scale of a thesis
  dataset (tens to a few hundred signs), that's unnecessary machinery,
  and the extension has to be downloaded the first time it's used, which
  may fail on a locked-down network. So this script tries to enable it
  (nice to have, and worth knowing it exists for your write-up), but the
  GUARANTEED path - what 05_semantic_search_test.py actually relies on -
  is plain brute-force cosine similarity computed explicitly in numpy
  after pulling the (small) set of vectors out of DuckDB. At this scale
  that's instant, and it's fully transparent: you can see every number.

This script ALSO writes the human-readable dictionary file again here
(same format as step 03) so that the vector_db/ folder is self-contained -
everything about "what's in the database" is visible without needing to
open DuckDB at all.

Run:
  python3 "04_hosting_vector_db_and_storing.py"
"""

import argparse
import os
import pickle
import json
from datetime import datetime

import duckdb
import numpy as np

import config
import db_utils


def try_load_vss_extension(con):
    """
    Best-effort only. Returns True if the vss extension loaded successfully,
    False otherwise (this is NOT an error - we fall back to numpy either way).
    """
    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        print("[db] Optional 'vss' extension loaded (HNSW ANN index available).")
        return True
    except Exception as e:
        print(f"[db] Optional 'vss' extension not available ({type(e).__name__}: {e}).")
        print("[db] That's fine - this script does not depend on it. Falling back to")
        print("[db] plain numpy cosine similarity, which is what 05_semantic_search_test.py uses anyway.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete all 'curated' rows and re-insert everything (default is incremental).")
    args = parser.parse_args()
    print("=" * 70)
    print("STEP 4: HOSTING A LOCAL VECTOR DATABASE (DuckDB) AND STORING EMBEDDINGS")
    print("=" * 70)

    config.ensure_dirs()

    rows_path = os.path.join(config.PATHS["embeddings"], "embedding_rows_for_db.pkl")
    if not os.path.isfile(rows_path):
        print(f"\n[ERROR] Could not find {rows_path}")
        print("Run 03_converting_extractions_to_vector_embeddings.py first.")
        return

    with open(rows_path, "rb") as f:
        db_rows = pickle.load(f)

    print(f"\nLoaded {len(db_rows)} embedding record(s) to store.")

    db_path = config.PATHS["duckdb_file"]
    print(f"\n[db] Opening/creating local DuckDB file at: {db_path}")
    con = duckdb.connect(db_path)

    try_load_vss_extension(con)

    print("\n[db] Creating table 'sign_embeddings' (or migrating an older one in place)...")
    db_utils.ensure_schema(con, verbose=True)

    # Only the rows that came from data_raw_videos ('curated') are replaced.
    # Signs you enrolled later with 07 / the out-of-vocabulary fallback are
    # left alone, so re-running this step never wipes them.
    kept = con.execute("SELECT COUNT(*) FROM sign_embeddings WHERE provenance <> ?",
                       [db_utils.PROVENANCE_CURATED]).fetchone()[0]
    if args.rebuild:
        print(f"[db] --rebuild: clearing previous 'curated' rows (keeping {kept} enrolled row(s))...")
        db_utils.delete_curated_rows(con)
    existing = {r[0] for r in con.execute(
        "SELECT source_video FROM sign_embeddings WHERE provenance = ?", [db_utils.PROVENANCE_CURATED]).fetchall()}
    to_insert = [r for r in db_rows if r["source_video"] not in existing]
    print(f"\n[db] {len(existing)} curated row(s) already stored; inserting {len(to_insert)} new row(s) "
          f"(skipping {len(db_rows) - len(to_insert)}):")
    now = datetime.now()
    for i, row in enumerate(to_insert, start=1):
        embedding = row["embedding"]
        print(f"[db]   ({i}/{len(to_insert)}) label='{row['label']}' | source={row['source_video']}")
        db_utils.insert_row(
            con, row["label"], row["source_video"], embedding,
            row["num_frames"], row["duration_sec"],
            provenance=db_utils.PROVENANCE_CURATED, stored_at=now,
        )

    con.commit()

    # Sanity read-back - prove to ourselves the data actually landed correctly.
    print("\n[db] Reading back what's actually stored, to verify:")
    result = con.execute(
        "SELECT label, source_video, len(embedding) AS vec_len, num_frames, duration_sec, provenance "
        "FROM sign_embeddings ORDER BY label, provenance"
    ).fetchall()
    print(f"\n{'Label':<25}{'Source video':<25}{'Vec len':>8}{'Frames':>8}{'Duration(s)':>13}  {'Provenance'}")
    for label, src, vec_len, num_frames, duration, prov in result:
        print(f"{label:<25}{src:<25}{vec_len:>8}{num_frames:>8}{duration:>13.2f}  {prov}")

    total_rows = con.execute("SELECT COUNT(*) FROM sign_embeddings").fetchone()[0]
    print(f"\n[db] Total rows now in sign_embeddings: {total_rows}")

    con.close()
    print(f"\n[db] Connection closed. Database file saved at: {db_path}")

    # Human-readable dictionary export, kept alongside the DB for easy eyeballing.
    label_to_vector = {}
    for row in db_rows:
        label_to_vector.setdefault(row["label"], []).append(row["embedding"])  # label -> list of vectors
    dict_path = os.path.join(config.PATHS["vector_db"], "stored_vectors_dict.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump(label_to_vector, f)

    preview = {
        label: {"num_examples": len(vecs), "first_example_first_10_dims": [round(v, 4) for v in vecs[0][:10]]}
        for label, vecs in label_to_vector.items()
    }
    preview_path = os.path.join(config.PATHS["vector_db"], "stored_vectors_PREVIEW.json")
    with open(preview_path, "w") as f:
        json.dump(preview, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)
    print(f"Vector DB file          : {db_path}")
    print(f"Dict export (.pkl)       : {dict_path}")
    print(f"Quick-look preview (json): {preview_path}")
    print(f"\nTry opening the DB yourself any time with:")
    print(f"   python3 -c \"import duckdb; print(duckdb.connect('{db_path}').execute('SELECT label, num_frames, duration_sec FROM sign_embeddings').df())\"")
    print("\nNext step: run 05_semantic_search_test.py to test retrieval.")


if __name__ == "__main__":
    main()
