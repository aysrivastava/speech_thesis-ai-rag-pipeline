"""
06_llm_context_rerank.py
---------------------------
STEP 6: the context-aware layer on top of retrieval. This is where the
pipeline becomes RAG proper: the retrieved candidates are handed to an LLM
together with the QUESTION the signer was answering, and the final answer
is decided from both the video evidence and the context.

What it does for one clip:
  1. Runs the exact same retrieval as step 5 (extract -> embed -> cosine
     similarity vs every stored sign).
  2. If you gave a question: asks an LLM how plausible each of the top-k
     candidates is as an answer to it (the LLM never sees the video or the
     similarity scores), and combines that with retrieval as a product of
     two independent pieces of evidence. If "best match #2 makes more
     sense", it says so loudly and shows why. Context can only overturn
     retrieval when the similarity gap is small - see context_utils.py.
  3. If nothing retrieved is a plausible answer (or the best similarity is
     below a calibrated floor), the answer is probably NOT in the
     vocabulary. A vision model then describes the clip and proposes a
     label; if you confirm, the clip is enrolled into the DB (provenance
     'vlm_confirmed') and plain retrieval recognises that sign from then on.

Needs an API key for steps 2-3:   export ANTHROPIC_API_KEY='...'
Without a key it still runs - it just reports retrieval's top-1 and says
the context step was skipped.

Run:
  python3 06_llm_context_rerank.py
  python3 06_llm_context_rerank.py --video test_query_videos/clip.mp4 --question "What is your favourite colour?"
  python3 06_llm_context_rerank.py --video clip.mp4 --no_fallback     # never call the vision model
  python3 06_llm_context_rerank.py --video clip.mp4 --sim_floor 0.80  # try an out-of-vocabulary floor
"""

import argparse
import os

import config
import context_utils
from landmark_extraction_utils import check_model_files_exist
from retrieval_utils import run_semantic_search_test


def _list_videos():
    d = config.PATHS["test_query_videos"]
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))


def _pick_video_interactively():
    files = _list_videos()
    if not files:
        print(f"\n[ERROR] No videos in {config.PATHS['test_query_videos']} - drop a clip there or pass --video.")
        return None
    print("\nFiles in test_query_videos/:")
    for i, f in enumerate(files, start=1):
        print(f"   [{i}] {f}")
    raw = input("Type a number: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(files):
        return os.path.join(config.PATHS["test_query_videos"], files[int(raw) - 1])
    print("Not a valid choice.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--question", type=str, default=None,
                        help="The question the signer was answering. Omit for retrieval only.")
    parser.add_argument("--top_k", type=int, default=config.RERANK_TOP_K)
    parser.add_argument("--no_fallback", action="store_true",
                        help="Never call the vision model / offer enrollment.")
    parser.add_argument("--sim_floor", type=float, default=None,
                        help="Out-of-vocabulary similarity floor (overrides config.OOV_SIM_FLOOR).")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 6: CONTEXT-AWARE RE-RANKING + OUT-OF-VOCABULARY FALLBACK")
    print("=" * 70)

    config.ensure_dirs()
    if not check_model_files_exist():
        print("\nStopping here until the model files are in place.")
        return

    if args.sim_floor is not None:
        config.OOV_SIM_FLOOR = args.sim_floor

    video = args.video or _pick_video_interactively()
    if not video or not os.path.isfile(video):
        print(f"\n[ERROR] Video not found: {video}")
        return

    question = args.question
    if question is None:
        question = input("\nQuestion the signer was answering (Enter = none, retrieval only): ").strip() or None

    result = run_semantic_search_test(video, top_k=args.top_k)
    if result is None:
        return

    context_utils.process_result(result, question, allow_fallback=not args.no_fallback)


if __name__ == "__main__":
    main()
