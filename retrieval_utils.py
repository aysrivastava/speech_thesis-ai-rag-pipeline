"""
retrieval_utils.py
---------------------
The actual "query the vector DB and rank matches" logic, pulled out of
05_semantic_search_test.py into its own importable module - same reason
embedding_utils.py and landmark_extraction_utils.py exist: so that BOTH
  - 05_semantic_search_test.py (test against a video file already sitting
    in test_query_videos/)
  - record_webcam_and_test.py (test against a clip you just recorded live)
run the exact same extraction -> embedding -> cosine-similarity -> ranking
code path, with zero risk of the two drifting apart over time.
"""

import os

import duckdb
import numpy as np

import config
from landmark_extraction_utils import extract_landmark_sequence
from embedding_utils import compute_motion_pooled_embedding, compute_query_embeddings


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def centered_similarities(query_vec, stored):
    """Cosine similarity after subtracting the mean stored vector from the
    query and every stored vector (hubness fix: removes the component all
    clips share, so what is left is what makes them DIFFERENT). Falls back
    to plain cosine if config.RETRIEVAL_CENTERING is off or <3 vectors."""
    q = np.asarray(query_vec, dtype=np.float64)
    M = np.stack([item["embedding"] for item in stored])
    if getattr(config, "RETRIEVAL_CENTERING", False) and len(stored) >= 3:
        mu = M.mean(axis=0)
        q, M = q - mu, M - mu
    return [cosine_similarity(q, row) for row in M]


def load_stored_vectors():
    db_path = config.PATHS["duckdb_file"]
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No DuckDB file found at {db_path}. Run 04_hosting_vector_db_and_storing.py first."
        )
    con = duckdb.connect(db_path)
    rows = con.execute("SELECT label, source_video, embedding FROM sign_embeddings ORDER BY label").fetchall()
    con.close()
    return [{"label": r[0], "source_video": r[1], "embedding": np.array(r[2], dtype=np.float64)} for r in rows]


def list_stored_labels_with_duration():
    """
    Returns [(label, duration_sec), ...] for everything in the vector DB,
    or [] if the DB doesn't exist yet / is empty. Never raises - callers
    use this to show the user what they can test against, and to know how
    long the original recording of each sign actually took.
    """
    db_path = config.PATHS["duckdb_file"]
    if not os.path.isfile(db_path):
        return []
    try:
        con = duckdb.connect(db_path)
        # GROUP BY: a label can have several examples once signs get enrolled.
        rows = con.execute(
            "SELECT label, AVG(duration_sec) FROM sign_embeddings GROUP BY label ORDER BY label"
        ).fetchall()
        con.close()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        return []


def list_stored_labels():
    """Just the labels, for callers that don't need durations."""
    return [label for label, _ in list_stored_labels_with_duration()]


def run_semantic_search_test(query_video_path, top_k=5, query_label_hint=None):
    """
    Runs the full step-5 experiment against one query video file:
    extract -> embed -> load DB -> cosine similarity -> ranked print.

    Returns a dict with the ranked results and the winner, so a caller
    (like record_webcam_and_test.py) can use the outcome programmatically
    instead of just reading the printout. Returns None if the test could
    not run (empty DB, unreadable video, etc) - the reason is printed.
    """
    print(f"\nQuery video: {query_video_path}")

    print("\n--- Extracting landmarks from the query video (same pipeline as step 1) ---")
    extraction = extract_landmark_sequence(query_video_path, verbose=True)

    if extraction["num_frames"] < 2:
        print("\n[ERROR] Fewer than 2 frames were extracted - can't compute velocity/embedding.")
        print("Check the video file actually has playable frames.")
        return None

    print("\n--- Embedding the query (same pipeline as step 3) ---")
    label_hint = query_label_hint or os.path.splitext(os.path.basename(query_video_path))[0]
    query_vectors = compute_query_embeddings(
        extraction["features"], verbose=True, label=f"QUERY:{label_hint}"
    )
    query_vector = query_vectors[0]
    if len(query_vectors) > 1:
        print("(query embedded in both hand orientations; the better match per stored sign is kept)")

    print("\n--- Loading stored vectors from the local vector DB (step 4) ---")
    stored = load_stored_vectors()
    print(f"Loaded {len(stored)} stored vector(s) from {config.PATHS['duckdb_file']}")
    if not stored:
        print("[ERROR] The vector DB is empty. Run 04_hosting_vector_db_and_storing.py first.")
        return None

    bad = [it for it in stored if it["embedding"].shape[0] != query_vector.shape[0]]
    if bad:
        print(f"\n[ERROR] {len(bad)} stored vector(s) have a different size ({bad[0]['embedding'].shape[0]}) "
              f"than the current embedding ({query_vector.shape[0]}). The DB was built with another "
              f"embedding version. Rebuild it:\n   python3 03_converting_extractions_to_vector_embeddings.py --force"
              f"\n   python3 04_hosting_vector_db_and_storing.py --rebuild\n(Enrolled signs from step 7 must be re-enrolled.)")
        return None

    print("\n--- Computing cosine similarity: query vector vs. every stored vector ---")
    scored = []
    sims = np.max([centered_similarities(qv, stored) for qv in query_vectors], axis=0)
    print(f"(embedding v{config.EMBEDDING_VERSION}, mean-centering={'on' if config.RETRIEVAL_CENTERING else 'off'})")
    for item, sim in zip(stored, sims):
        scored.append((item["label"], item["source_video"], sim))
        print(f"[compare] query vs '{item['label']:<20}' -> cosine similarity = {sim:+.4f}")

    # A label can have several stored examples (enrolled signs). Rank LABELS,
    # scoring each by its best-matching example - otherwise the same sign
    # would appear twice and the "gap to 2nd place" would compare a sign
    # with itself.
    best_per_label = {}
    for label, src, sim in scored:
        if label not in best_per_label or sim > best_per_label[label][2]:
            best_per_label[label] = (label, src, sim)
    scored = list(best_per_label.values())

    scored.sort(key=lambda x: x[2], reverse=True)

    top_k = min(top_k, len(scored))
    print("\n" + "=" * 70)
    print(f"RANKED RESULTS (top {top_k})")
    print("=" * 70)
    print(f"{'Rank':<6}{'Label':<25}{'Cosine similarity':>18}")
    for rank, (label, src, sim) in enumerate(scored[:top_k], start=1):
        marker = "  <-- BEST MATCH" if rank == 1 else ""
        print(f"{rank:<6}{label:<25}{sim:>18.4f}{marker}")

    best_label, best_src, best_sim = scored[0]
    print(f"\nBest match: '{best_label}' (source video: {best_src}), similarity = {best_sim:.4f}")

    gap = None
    if len(scored) > 1:
        gap = scored[0][2] - scored[1][2]
        print(f"Gap to 2nd place ('{scored[1][0]}'): {gap:.4f}")
        if gap < 0.02:
            print("NOTE: top-2 scores are very close - this match is not very confident.")

    hand_rate = extraction["hand_detect_rate"]
    if hand_rate < 0.7:
        print(f"\nNOTE: hand detection rate was only {hand_rate*100:.1f}% on this clip (ideally >70%).")
        print("A low rate here usually means framing/distance issues, not a matching-algorithm problem -")
        print("check that your signing hand(s) stayed inside the camera frame for the whole sign.")

    return {
        "scored": scored,
        "best_label": best_label,
        "best_source_video": best_src,
        "best_similarity": best_sim,
        "gap_to_second": gap,
        "hand_detect_rate": hand_rate,
        "pose_detect_rate": extraction["pose_detect_rate"],
        # Kept so a clip that turns out to be out-of-vocabulary can be
        # enrolled WITHOUT running MediaPipe a second time (see
        # enrollment_utils.enroll_from_result).
        "query_vector": query_vector,
        "num_frames": extraction["num_frames"],
        "duration_sec": extraction["duration_sec"],
        "query_video_path": query_video_path,
    }
