"""
07_enroll_new_sign.py
------------------------
STEP 7: add a new sign to the vector DB WITHOUT retraining anything.

This is how the "closed 97-sign set" stops being closed. With a classifier,
a 98th sign means retraining. Here the vocabulary is just the rows in
DuckDB, so a new sign is one more row: record it, embed it, store it.

(The out-of-vocabulary fallback in step 6 does the same thing automatically
after a vision model proposes a label and you confirm it. This script is the
manual route, plus tools to inspect and undo enrollments.)

Examples:
  python3 07_enroll_new_sign.py --video test_query_videos/red.mp4 --label "Red"
  python3 07_enroll_new_sign.py                    # interactive: pick a clip, type a label
  python3 07_enroll_new_sign.py --list             # what's in the DB, and where each row came from
  python3 07_enroll_new_sign.py --delete "Red"     # undo enrollments of a label (never touches curated rows)

Tip: enroll 2-3 different recordings of the same sign (run it again with the
same --label). Several rows can share a label; retrieval scores each label
by its best-matching example, which makes a new sign far more robust than a
single clip.
"""

import argparse
import os

import config
import db_utils
import enrollment_utils
from landmark_extraction_utils import check_model_files_exist


def _list_videos():
    d = config.PATHS["test_query_videos"]
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))


def print_db_contents():
    rows = enrollment_utils.list_rows()
    if not rows:
        print("\nThe vector DB is empty (or doesn't exist yet).")
        return
    print(f"\n{'Label':<26}{'Provenance':<15}{'Source video':<34}{'Dur(s)':>7}")
    for label, prov, src, dur, _ in rows:
        print(f"{label:<26}{prov or 'curated':<15}{(src or ''):<34}{dur:>7.2f}")
    labels = {r[0] for r in rows}
    enrolled = sum(1 for r in rows if (r[1] or "curated") != db_utils.PROVENANCE_CURATED)
    print(f"\n{len(rows)} row(s) across {len(labels)} distinct sign(s); {enrolled} enrolled row(s).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="Show everything in the DB and exit.")
    parser.add_argument("--delete", type=str, default=None, metavar="LABEL",
                        help="Remove ENROLLED rows for this label (curated rows are never touched).")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 7: ENROLL A NEW SIGN (no retraining)")
    print("=" * 70)
    config.ensure_dirs()

    if args.list:
        print_db_contents()
        return

    if args.delete:
        n = enrollment_utils.delete_enrolled_label(args.delete)
        print(f"\nRemoved {n} enrolled row(s) for '{args.delete}'." if n else
              f"\nNo enrolled rows found for '{args.delete}' (curated rows can't be deleted here).")
        return

    if not check_model_files_exist():
        print("\nStopping here until the model files are in place.")
        return

    video, label = args.video, args.label
    if not video:
        files = _list_videos()
        if not files:
            print(f"\n[ERROR] No videos in {config.PATHS['test_query_videos']} - drop a clip there or pass --video.")
            return
        print("\nFiles in test_query_videos/:")
        for i, f in enumerate(files, start=1):
            print(f"   [{i}] {f}")
        raw = input("Type a number: ").strip()
        if not (raw.isdigit() and 1 <= int(raw) <= len(files)):
            print("Not a valid choice.")
            return
        video = os.path.join(config.PATHS["test_query_videos"], files[int(raw) - 1])
    if not os.path.isfile(video):
        print(f"\n[ERROR] Video not found: {video}")
        return

    if not label:
        label = input("Label for this sign (e.g. Red): ").strip()
    if not label:
        print("No label given - nothing enrolled.")
        return

    enrollment_utils.enroll_clip(video, label, provenance=db_utils.PROVENANCE_MANUAL)
    print("\nDone. Test it: record/choose a fresh clip of the same sign and run record_webcam_and_test.py "
          "or 05_semantic_search_test.py - it should now be retrievable.")


if __name__ == "__main__":
    main()
