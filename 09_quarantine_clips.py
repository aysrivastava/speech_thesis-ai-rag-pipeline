"""
09_quarantine_clips.py
--------------------------
Move bad / off-protocol training clips OUT of data_raw_videos/ (into
data_quarantine/, never deleted) and remove their extracted features, so the
next rebuild doesn't see them.

Why a script and not just dragging files in Finder: step 03 embeds every file
in features_extracted/, so a clip you remove from data_raw_videos/ would
STILL be embedded unless its feature file goes too.

Selection rules (combine them; a clip is flagged if ANY applies):
  --max_mb 3        file smaller than 3 MB (your corpus clips are 13-20 MB;
                    the tiny 0.1-1 MB phone re-captures stand out)
  --min_hand 0.6    MediaPipe saw a hand on fewer than 60% of frames
  --names "a.mp4" "b/c.mp4"    explicit relative paths

DRY RUN by default - it only prints what it would move. Add --apply to do it.

Then rebuild:
  python3 03_converting_extractions_to_vector_embeddings.py --force
  python3 04_hosting_vector_db_and_storing.py --rebuild
  python3 08_diagnose_retrieval.py
"""

import argparse
import os
import pickle
import shutil

import config

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def stem_of(rel):
    return os.path.splitext(rel)[0].replace(os.sep, "__").replace(" ", "_")


def list_clips(video_dir):
    out = []
    for entry in sorted(os.listdir(video_dir)):
        full = os.path.join(video_dir, entry)
        if entry.startswith("."):
            continue
        if os.path.isdir(full):
            for f in sorted(os.listdir(full)):
                if f.lower().endswith(VIDEO_EXTS) and not f.startswith("."):
                    out.append((entry, os.path.join(entry, f)))
        elif entry.lower().endswith(VIDEO_EXTS):
            out.append((entry, entry))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_mb", type=float, default=None)
    ap.add_argument("--min_hand", type=float, default=None)
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--apply", action="store_true", help="Actually move files (default: dry run).")
    args = ap.parse_args()
    if args.max_mb is None and args.min_hand is None and not args.names:
        ap.error("give at least one rule: --max_mb, --min_hand or --names")

    video_dir = config.PATHS["data_raw_videos"]
    feat_dir = config.PATHS["features_extracted"]
    qdir = os.path.join(os.path.dirname(os.path.abspath(video_dir)), "data_quarantine")

    flagged = []
    for label, rel in list_clips(video_dir):
        reasons = []
        size_mb = os.path.getsize(os.path.join(video_dir, rel)) / 1e6
        if args.max_mb is not None and size_mb < args.max_mb:
            reasons.append(f"small file ({size_mb:.2f} MB)")
        if args.min_hand is not None:
            pk = os.path.join(feat_dir, stem_of(rel) + ".pkl")
            if os.path.isfile(pk):
                with open(pk, "rb") as fh:
                    rate = pickle.load(fh).get("hand_detect_rate", 1.0)
                if rate < args.min_hand:
                    reasons.append(f"hand detected on only {rate*100:.0f}% of frames")
        if rel in args.names:
            reasons.append("named explicitly")
        if reasons:
            flagged.append((rel, "; ".join(reasons)))

    if not flagged:
        print("No clips matched. Nothing to do.")
        return
    print(f"{'DRY RUN - ' if not args.apply else ''}{len(flagged)} clip(s) flagged:")
    for rel, why in flagged:
        print(f"   {rel:<55} {why}")

    if not args.apply:
        print("\nLooks right? Re-run with --apply to move them to data_quarantine/.")
        return

    for rel, _ in flagged:
        dst = os.path.join(qdir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(video_dir, rel), dst)
        pk = os.path.join(feat_dir, stem_of(rel) + ".pkl")
        if os.path.isfile(pk):
            os.remove(pk)
    print(f"\nMoved {len(flagged)} clip(s) to {qdir} and removed their feature files.")
    print("Now rebuild:\n   python3 03_converting_extractions_to_vector_embeddings.py --force"
          "\n   python3 04_hosting_vector_db_and_storing.py --rebuild\n   python3 08_diagnose_retrieval.py")


if __name__ == "__main__":
    main()
