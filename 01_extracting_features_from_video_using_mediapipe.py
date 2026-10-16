"""
01_extracting_features_from_video_using_mediapipe.py
-------------------------------------------------------
STEP 1 of the RAG pipeline.

What this does:
  Reads every video in data_raw_videos/, runs MediaPipe Hand + Pose
  landmark detection over every frame, and saves the raw per-frame
  landmark sequence (shape T x 225) to features_extracted/<label>.pkl.

What this does NOT do:
  It does not turn the sequence into a single fixed-size vector yet -
  that's step 03. This script's only job is "pixels in, landmarks out".

Expected input file naming:
  data_raw_videos/How_Are_You.mp4   -> label "How Are You"
  data_raw_videos/Thank_You.mp4     -> label "Thank You"
  (underscore in the filename becomes a space in the label; change
  `filename_to_label` below if you'd rather keep your own labelling
  scheme, e.g. from a CSV like the old corpus_csv.)

Run:
  python3 "01_extracting_features_from_video_using_mediapipe.py"
"""

import argparse
import os
import pickle

import config
from landmark_extraction_utils import extract_landmark_sequence, check_model_files_exist


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def filename_to_label(filename):
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext.replace("_", " ").strip()


def find_labelled_videos(video_dir):
    """Returns [(label, relative_path)] for two layouts, which may be mixed:
      data_raw_videos/Who are you/signer1.mp4, signer2.mp4 ...  -> label = FOLDER name, verbatim
      data_raw_videos/How_Are_You.mp4                           -> label from filename (old layout)
    Several videos per label are expected and fine."""
    found = []
    for entry in sorted(os.listdir(video_dir)):
        if entry.startswith("."):
            continue
        full = os.path.join(video_dir, entry)
        if os.path.isdir(full):
            label = entry.strip()
            for f in sorted(os.listdir(full)):
                if f.lower().endswith(VIDEO_EXTS) and not f.startswith("."):
                    found.append((label, os.path.join(entry, f)))
        elif entry.lower().endswith(VIDEO_EXTS):
            found.append((filename_to_label(entry), entry))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-extract every clip, even ones already done.")
    args = parser.parse_args()
    print("=" * 70)
    print("STEP 1: EXTRACTING MEDIAPIPE LANDMARKS FROM RAW VIDEOS")
    print("=" * 70)

    config.ensure_dirs()

    if not check_model_files_exist():
        print("\nStopping here until the model files are in place.")
        return

    video_dir = config.PATHS["data_raw_videos"]
    out_dir = config.PATHS["features_extracted"]

    video_files = find_labelled_videos(video_dir)

    if not video_files:
        print(f"\n[ERROR] No video files found in: {video_dir}")
        print("Put one FOLDER per sign (folder name = exact label), with every signer's clip inside:")
        print("   data_raw_videos/Who are you/signer1.mp4, signer2.mp4, ...")
        print("   data_raw_videos/Thank you/signer1.mp4, ...")
        print("(Flat files like data_raw_videos/Thank_You.mp4 also still work.)")
        return

    from collections import Counter
    per_label = Counter(l for l, _ in video_files)
    print(f"\nFound {len(video_files)} video(s) across {len(per_label)} label(s):")
    for l, n in per_label.items():
        print(f"   - '{l}': {n} clip(s)")

    summary = []
    skipped = 0

    for i, (label, filename) in enumerate(video_files, start=1):
        video_path = os.path.join(video_dir, filename)
        stem = os.path.splitext(filename)[0].replace(os.sep, "__").replace(" ", "_")
        out_path = os.path.join(out_dir, f"{stem}.pkl")

        if os.path.isfile(out_path) and not args.force:
            skipped += 1
            print(f"[{i}/{len(video_files)}] skip (already extracted): {filename}")
            continue

        print(f"\n--- [{i}/{len(video_files)}] Processing '{label}' ({filename}) ---")

        result = extract_landmark_sequence(video_path, verbose=True)

        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "label": label,
                    "source_video": filename,
                    "features": result["features"],       # (T, 225) float32
                    "num_frames": result["num_frames"],
                    "fps": result["fps"],
                    "duration_sec": result["duration_sec"],
                    "hand_detect_rate": result["hand_detect_rate"],
                    "pose_detect_rate": result["pose_detect_rate"],
                },
                f,
            )

        print(f"[save] Wrote {out_path}")
        print(f"[save]   shape={result['features'].shape}  duration={result['duration_sec']:.2f}s")

        summary.append(
            {
                "label": label,
                "num_frames": result["num_frames"],
                "duration_sec": result["duration_sec"],
                "hand_detect_rate": result["hand_detect_rate"],
                "pose_detect_rate": result["pose_detect_rate"],
            }
        )

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"{'Label':<25}{'Frames':>8}{'Duration(s)':>13}{'Hand %':>9}{'Pose %':>9}")
    for row in summary:
        print(
            f"{row['label']:<25}{row['num_frames']:>8}{row['duration_sec']:>13.2f}"
            f"{row['hand_detect_rate']*100:>8.1f}%{row['pose_detect_rate']*100:>8.1f}%"
        )
    print(f"\nNewly extracted: {len(summary)} | skipped (already done): {skipped}  [use --force to redo all]")
    print(f"Feature files are in: {out_dir}")
    print("Next step: run 03_converting_extractions_to_vector_embeddings.py")
    print("(02_chunking_strategy_analysis.py is optional - see README 'Is chunking needed?')")


if __name__ == "__main__":
    main()
