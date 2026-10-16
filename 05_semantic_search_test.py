"""
05_semantic_search_test.py
-----------------------------
STEP 5 of the RAG pipeline - THE EXPERIMENT YOU ACTUALLY WANT TO RUN:

  "Store a video to the vector DB, then capture the SAME sign again,
   and test if semantic search really works."

What this script does:
  1. Takes a new video from test_query_videos/ (you record yourself
     signing something that's ALSO in data_raw_videos/, e.g. sign
     "How Are You" again, save it as test_query_videos/How_Are_You_test.mp4).
  2. Runs it through the EXACT SAME extraction (step 1's logic) and
     embedding (step 3's logic) pipeline, via the shared utility modules -
     so there is no chance of the query being processed differently from
     how the DB was built.
  3. Loads every stored vector back out of the local DuckDB (step 4),
     and computes cosine similarity between your new query vector and
     every stored vector.
  4. Prints a ranked table so you can see with your own eyes whether the
     top match is the sign you actually just performed.

No LLM is involved anywhere in this script - this is pure retrieval,
exactly as you asked for in step 5 of your plan. Step 6 (later) is where
an LLM would take the retrieved label(s) and generate a natural sentence.

Prefer recording your test clip straight from your webcam instead of your
phone? See record_webcam_and_test.py - it records (landscape, matching the
training clips' framing) AND runs this exact same test automatically right
after you stop recording. This script is still here for testing a clip you
already have as a file, from any source.

Run:
  python3 05_semantic_search_test.py
  python3 05_semantic_search_test.py --video test_query_videos/How_Are_You_test.mp4
"""

import argparse
import os

import config
from landmark_extraction_utils import check_model_files_exist
from retrieval_utils import run_semantic_search_test


def pick_query_video(explicit_path=None):
    if explicit_path:
        return explicit_path

    query_dir = config.PATHS["test_query_videos"]
    candidates = sorted(
        f for f in os.listdir(query_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    )
    if not candidates:
        return None
    return os.path.join(query_dir, candidates[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                         help="Path to the query video. If omitted, uses the first video found in test_query_videos/.")
    parser.add_argument("--top_k", type=int, default=5, help="How many ranked matches to show.")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 5: SEMANTIC SEARCH TEST (store once, capture again, retrieve)")
    print("=" * 70)

    config.ensure_dirs()

    if not check_model_files_exist():
        print("\nStopping here until the model files are in place.")
        return

    query_video = pick_query_video(args.video)
    if query_video is None or not os.path.isfile(query_video):
        print(f"\n[ERROR] No query video found.")
        print(f"Record yourself performing one of your stored signs again and save it into:")
        print(f"   {config.PATHS['test_query_videos']}")
        print("Then re-run this script (or pass --video path/to/clip.mp4).")
        print("(Or use record_webcam_and_test.py to record and test in one step.)")
        return

    print("(this should be a FRESH recording of a sign that's already stored in the DB)")

    result = run_semantic_search_test(query_video, top_k=args.top_k)
    if result is None:
        return

    print("\nDoes the best match correspond to the sign you actually performed in")
    print(f"'{query_video}'? If yes, semantic search is working on this pipeline.")
    print("If the wrong sign keeps winning, likely causes to check (in order of likelihood):")
    print("  1. Low hand/pose detection rate during extraction (scroll up - was it under ~70%?)")
    print("  2. The query recording's framing/distance from camera is very different from the")
    print("     stored clip (the embedding is not yet scale/position invariant beyond what")
    print("     MediaPipe's own normalized coordinates already provide).")
    print("  3. Too few stored examples per sign (one recording per sign is a thin basis for")
    print("     matching - natural variation between two performances of the same sign may")
    print("     be larger than expected).")
    print("\nOnce this is reliably working, step 6 is integrating an LLM on top of the")
    print("retrieved label(s) to produce a natural-language sentence - making this officially RAG.")


if __name__ == "__main__":
    main()
