"""
enrollment_utils.py
----------------------
Adds a NEW sign to the vector DB after the fact, with no retraining.

This is the architectural point of using retrieval instead of a classifier:
the vocabulary is the contents of DuckDB, not weights. The old BiLSTM was
closed-set (97 outputs, full stop). Here, "the closed set" is just
"whatever is currently in sign_embeddings", and growing it is one INSERT.

Two entry points:
  enroll_from_result(result, label, provenance)
      Reuses the embedding that retrieval already computed for a query clip
      (result["query_vector"]), so enrolling the clip you just tested does
      NOT run MediaPipe a second time. Used by the out-of-vocabulary fallback.
  enroll_clip(video_path, label, provenance)
      Starts from a video file: extract -> embed -> store. Used by
      07_enroll_new_sign.py.

Every enrollment copies the clip into enrolled_videos/ so the DB row's
source_video points at a stable file, and records `provenance`:
  'manual'         you chose the label yourself
  'vlm_confirmed'  a vision model proposed it and you confirmed
  ('curated' is reserved for rows from steps 01-04.)

Honest caveat: one enrolled example is a thin basis (the statistical
embedding is sensitive to framing, as your first real test showed). Enroll
2-3 examples of a sign if you can - several rows may share a label; retrieval
scores each LABEL by its best-matching example.
"""

import os
import shutil
from datetime import datetime

import config
import db_utils


def _normalize_label(label):
    return " ".join(str(label).strip().split())


def _copy_into_enrolled_folder(video_path, label):
    """Returns the stored path (basename) - falls back to the original name
    if the copy fails; enrollment itself must not fail over a copy."""
    try:
        os.makedirs(config.PATHS["enrolled_videos"], exist_ok=True)
        ext = os.path.splitext(video_path)[1] or ".mp4"
        name = f"{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        shutil.copy2(video_path, os.path.join(config.PATHS["enrolled_videos"], name))
        return name
    except Exception as e:
        print(f"[enroll] (could not copy the clip into enrolled_videos/: {e})")
        return os.path.basename(video_path)


def _store(label, source_name, embedding, num_frames, duration_sec, provenance):
    con = db_utils.connect()
    try:
        db_utils.ensure_schema(con, verbose=True)
        db_utils.insert_row(con, label, source_name, embedding, num_frames, duration_sec, provenance)
        con.commit()
        n_examples = con.execute("SELECT COUNT(*) FROM sign_embeddings WHERE label = ?", [label]).fetchone()[0]
        total = con.execute("SELECT COUNT(DISTINCT label) FROM sign_embeddings").fetchone()[0]
    finally:
        con.close()
    return n_examples, total


def enroll_from_result(result, label, provenance=db_utils.PROVENANCE_MANUAL):
    label = _normalize_label(label)
    if not label:
        raise ValueError("label must not be empty")
    src = _copy_into_enrolled_folder(result["query_video_path"], label)
    n_examples, total = _store(
        label, src, result["query_vector"], result["num_frames"], result["duration_sec"], provenance
    )
    print(f"[enroll] Stored '{label}' (provenance={provenance}). It now has {n_examples} example(s); "
          f"the DB holds {total} distinct sign(s).")
    return {"label": label, "examples": n_examples, "total_labels": total}


def enroll_clip(video_path, label, provenance=db_utils.PROVENANCE_MANUAL, verbose=True):
    # Imported here (not at module top) so enroll_from_result stays usable
    # without importing mediapipe at all.
    from landmark_extraction_utils import extract_landmark_sequence
    from embedding_utils import compute_motion_pooled_embedding

    label = _normalize_label(label)
    if not label:
        raise ValueError("label must not be empty")

    extraction = extract_landmark_sequence(video_path, verbose=verbose)
    if extraction["num_frames"] < 2:
        raise RuntimeError("Fewer than 2 frames extracted - cannot embed this clip.")
    if extraction["hand_detect_rate"] < 0.5:
        print(f"[enroll] WARNING: hands detected on only {extraction['hand_detect_rate']*100:.0f}% of "
              f"frames - this example may embed poorly.")

    embedding = compute_motion_pooled_embedding(extraction["features"], verbose=verbose, label=f"ENROLL:{label}")
    src = _copy_into_enrolled_folder(video_path, label)
    n_examples, total = _store(
        label, src, embedding, extraction["num_frames"], extraction["duration_sec"], provenance
    )
    print(f"[enroll] Stored '{label}' (provenance={provenance}). It now has {n_examples} example(s); "
          f"the DB holds {total} distinct sign(s).")
    return {"label": label, "examples": n_examples, "total_labels": total}


def list_rows():
    """[(label, provenance, source_video, duration_sec, stored_at), ...]"""
    if not os.path.isfile(config.PATHS["duckdb_file"]):
        return []
    con = db_utils.connect()
    try:
        db_utils.ensure_schema(con)
        return con.execute(
            "SELECT label, provenance, source_video, duration_sec, stored_at "
            "FROM sign_embeddings ORDER BY label, stored_at"
        ).fetchall()
    finally:
        con.close()


def delete_enrolled_label(label):
    """Removes ENROLLED rows (manual / vlm_confirmed) for a label - the undo
    for a bad enrollment. Never touches 'curated' rows. Returns rows deleted."""
    label = _normalize_label(label)
    con = db_utils.connect()
    try:
        db_utils.ensure_schema(con)
        before = con.execute(
            "SELECT COUNT(*) FROM sign_embeddings WHERE label = ? AND provenance <> ?",
            [label, db_utils.PROVENANCE_CURATED],
        ).fetchone()[0]
        con.execute(
            "DELETE FROM sign_embeddings WHERE label = ? AND provenance <> ?",
            [label, db_utils.PROVENANCE_CURATED],
        )
        con.commit()
    finally:
        con.close()
    return before
