"""
02_chunking_strategy_analysis.py
----------------------------------
OPTIONAL step - answering your question "Is chunking really needed??"
with actual numbers from your own data, instead of a guess.

Not required for the pipeline to run: 03_converting_extractions_to_vector_
embeddings.py does NOT read this script's output (chunking_decision.json)
- it embeds whatever is in features_extracted/ regardless. The required
run order is 01 -> 03 -> 04. Given your clips are consistently ~5-7s
(single sign/sentence per file), the answer is already known without
running this: chunking is NOT needed now. This script is kept only as a
documented, data-driven "future capability" - run it once if/when you add
longer, continuous, multi-sentence video later and want real evidence for
a segmentation strategy, or if you just want the numbers for your thesis.

What "chunking" means in a text-RAG system:
  A long document gets split into smaller pieces ("chunks") before
  embedding, because (a) embedding models have a max input size, and
  (b) a whole document embedded as one vector is too "blurry" to match a
  specific question - you want the chunk that actually contains the
  answer to come back, not the whole book.

Why that concern may or may not apply to your ISL clips:
  Each file in data_raw_videos/ is already ONE short, single-sentence/
  gloss sign clip (a few seconds long) - not a long continuous recording
  with multiple unrelated sentences in it. If that's true, each clip is
  already the right retrieval unit, and "chunking" would mean artificially
  slicing a 3-second clip into smaller pieces that each contain only HALF
  a sign - which would make retrieval worse, not better.

This script doesn't just assert that - it measures your actual extracted
clips (from step 01's output) and prints the evidence, then writes a
decision file you can literally cite in your thesis methodology section.

Run:
  python3 02_chunking_strategy_analysis.py
"""

import json
import os
import pickle

import numpy as np

import config


def load_all_feature_files():
    feat_dir = config.PATHS["features_extracted"]
    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".pkl"))
    records = []
    for f in files:
        with open(os.path.join(feat_dir, f), "rb") as fh:
            records.append(pickle.load(fh))
    return records


def main():
    print("=" * 70)
    print("STEP 2: CHUNKING STRATEGY ANALYSIS")
    print("=" * 70)

    config.ensure_dirs()

    records = load_all_feature_files()
    if not records:
        print(f"\n[ERROR] No feature files found in {config.PATHS['features_extracted']}")
        print("Run 01_extracting_features_from_video_using_mediapipe.py first.")
        return

    durations = np.array([r["duration_sec"] for r in records])
    num_frames = np.array([r["num_frames"] for r in records])

    print(f"\nLoaded {len(records)} extracted clip(s).\n")
    print(f"{'Label':<25}{'Frames':>8}{'Duration(s)':>13}")
    for r in records:
        print(f"{r['label']:<25}{r['num_frames']:>8}{r['duration_sec']:>13.2f}")

    print("\n--- Clip-length statistics (this is the evidence) ---")
    print(f"   min duration   : {durations.min():.2f} s")
    print(f"   max duration   : {durations.max():.2f} s")
    print(f"   mean duration  : {durations.mean():.2f} s")
    print(f"   median duration: {np.median(durations):.2f} s")
    print(f"   min frames     : {int(num_frames.min())}")
    print(f"   max frames     : {int(num_frames.max())}")

    # Decision rule, made explicit and data-driven rather than hand-waved:
    #   - If every clip is reasonably short (we use 15s as a generous upper
    #     bound for a single ISL sentence) AND the dataset is "one file =
    #     one sign/sentence" by construction (true here - that's how you
    #     built data_raw_videos/), then each clip is already a single
    #     retrieval unit and chunking would only hurt.
    #   - If some clip is unusually long (e.g. you dropped in a multi-
    #     sentence continuous recording), that ONE clip is flagged as
    #     needing segmentation - chunking becomes a per-video decision,
    #     not an all-or-nothing one.
    LONG_CLIP_THRESHOLD_SEC = 15.0
    long_clips = [r["label"] for r in records if r["duration_sec"] > LONG_CLIP_THRESHOLD_SEC]

    print(f"\n--- Decision (threshold: a single-sentence clip should be under "
          f"{LONG_CLIP_THRESHOLD_SEC:.0f}s) ---")

    if not long_clips:
        decision = "NOT_NEEDED"
        reasoning = (
            f"All {len(records)} clips are under {LONG_CLIP_THRESHOLD_SEC:.0f} seconds "
            f"(max observed: {durations.max():.2f}s). Each video file already represents "
            "exactly one sign/sentence by construction (one label per file). Splitting a "
            "clip this short into smaller chunks would cut a sign's motion mid-gesture, "
            "destroying the very handshape/motion pattern the embedding needs to recognize. "
            "CONCLUSION: chunking is NOT needed for this dataset. Each clip is used whole as "
            "a single retrievable unit."
        )
        print("   -> Chunking is NOT needed. Each clip is already one retrieval unit.")
    else:
        decision = "NEEDED_FOR_SOME_CLIPS"
        reasoning = (
            f"{len(long_clips)} clip(s) exceed {LONG_CLIP_THRESHOLD_SEC:.0f} seconds: "
            f"{long_clips}. These are long enough that they may contain more than one "
            "sign/sentence, and should be segmented (e.g. by detecting pauses/rest frames "
            "where both hands return to a neutral position) before embedding. Shorter clips "
            "in the same dataset do not need this."
        )
        print(f"   -> {len(long_clips)} clip(s) are long enough to need segmentation: {long_clips}")
        print("      (Everything else is still used whole.)")

    print(f"\nReasoning:\n{reasoning}")

    decision_path = os.path.join(config.ROOT, "chunking_decision.json")
    with open(decision_path, "w") as f:
        json.dump(
            {
                "decision": decision,
                "reasoning": reasoning,
                "num_clips_analyzed": len(records),
                "long_clip_threshold_sec": LONG_CLIP_THRESHOLD_SEC,
                "long_clips": long_clips,
                "duration_stats_sec": {
                    "min": float(durations.min()),
                    "max": float(durations.max()),
                    "mean": float(durations.mean()),
                    "median": float(np.median(durations)),
                },
            },
            f,
            indent=2,
        )
    print(f"\n[save] Decision + evidence written to: {decision_path}")
    print("       (cite the numbers in this file directly in your thesis methodology section)")
    print("\nNote for future work: if you later add long, continuous, multi-sentence ISL")
    print("video (not pre-cut clips), THIS is where a real chunking/segmentation strategy")
    print("would need to be designed - e.g. a sliding window, or detecting rest frames")
    print("between signs. For the pre-segmented dataset you have now, it isn't needed.")
    print("\nNext step: run 03_converting_extractions_to_vector_embeddings.py")


if __name__ == "__main__":
    main()
