# scripts/02_download_models.py
# ── What this script does ─────────────────────────────────────────────────────
# Downloads the two MediaPipe neural network model files needed for
# hand and body pose landmark detection.
# These are .task files (~30 MB total) from Google's storage servers.
# Safe to re-run — skips if files already exist.
# Run: python3 scripts/02_download_models.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS

import urllib.request

MODELS = {
    'hand_landmarker.task': {
        'url' : 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'desc': 'Detects 21 keypoints on each hand (x, y, z per point = 63 values per hand)',
        'dest': PATHS['hand_model'],
    },
    'pose_landmarker.task': {
        'url' : 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        'desc': 'Detects 33 body pose keypoints (x, y, z per point = 99 values)',
        'dest': PATHS['pose_model'],
    },
}

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1e6
        print(f"\r  {pct:5.1f}%  {mb:.1f} MB downloaded", end='', flush=True)

def download_models():
    print("=" * 55)
    print("DOWNLOADING MEDIAPIPE MODEL FILES")
    print("=" * 55)

    all_ok = True
    for fname, info in MODELS.items():
        dest = Path(info['dest'])
        print(f"\nModel : {fname}")
        print(f"Desc  : {info['desc']}")
        print(f"Dest  : {dest}")

        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            print(f"✅ Already exists ({size_mb:.1f} MB) — skipping")
            continue

        print(f"📥 Downloading...")
        try:
            urllib.request.urlretrieve(info['url'], str(dest), reporthook=show_progress)
            print()  # newline after progress bar
            size_mb = dest.stat().st_size / 1e6
            print(f"✅ Saved ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            all_ok = False

    return all_ok


def verify():
    print("\n" + "=" * 55)
    print("MODEL VERIFICATION")
    print("=" * 55)

    checks = {
        'hand_model' : ('hand_landmarker.task', 'Hand keypoint detector  (21 points × 3 = 63 values)'),
        'pose_model' : ('pose_landmarker.task', 'Body pose detector      (33 points × 3 = 99 values)'),
    }

    all_ok = True
    for key, (fname, desc) in checks.items():
        path = PATHS[key]
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  ✅ {fname:<30} {size_mb:5.1f} MB  —  {desc}")
        else:
            print(f"  ❌ {fname:<30} NOT FOUND")
            all_ok = False

    print("=" * 55)
    if all_ok:
        print("✅ Both models ready — never run this script again")
        print("➡ Next: python3 scripts/03_extract.py")
    else:
        print("⚠ Some models missing — re-run this script")


if __name__ == "__main__":
    ok = download_models()
    verify()