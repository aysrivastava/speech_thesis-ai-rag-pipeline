# scripts/03_extract.py
# ── What this script does ─────────────────────────────────────────────────────
# Loops through every sentence folder → every signer subfolder → every image.
# For each image, runs two MediaPipe neural nets:
#   - Hand landmarker  → 21 points × 3 coords = 63 values (right hand)
#                      → 21 points × 3 coords = 63 values (left hand)
#   - Pose landmarker  → 33 points × 3 coords = 99 values (body)
# Stacks 50 frames → one (50 × 225) matrix per signer per sentence.
# Saves each matrix as a .pkl file in features/
# Progress is saved after every file → safe to interrupt and resume.
# Run: python3 scripts/03_extract.py
# Expected time: ~10-15 minutes on MacBook with MPS
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, SEQ_LEN, FEAT_DIM, STRIDE

import cv2, numpy as np, mediapipe as mp, pickle, json, time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

OUTPUT_DIR    = PATHS['features']
PROGRESS_FILE = PATHS['progress_file']
FRAMES_DIR    = PATHS['frames_sent']
HAND_MODEL    = str(PATHS['hand_model'])
POSE_MODEL    = str(PATHS['pose_model'])

# ── Pre-flight checks ─────────────────────────────────────────────────────────
assert FRAMES_DIR.exists(),       f"❌ Dataset not found: {FRAMES_DIR}\n   Run 01_download.py first"
assert Path(HAND_MODEL).exists(), f"❌ Hand model not found\n   Run 02_download_models.py first"
assert Path(POSE_MODEL).exists(), f"❌ Pose model not found\n   Run 02_download_models.py first"

# ── Initialise MediaPipe detectors ────────────────────────────────────────────
# Both run in IMAGE mode (not VIDEO/LIVE) since we process static frames
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE))

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        min_pose_detection_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE))


def extract_landmarks(image_bgr):
    """
    Takes one BGR image, runs hand + pose detection,
    returns a flat list of 225 floats:
      [right_hand(63), left_hand(63), pose(99)]
    If a hand or body is not detected, that section is all 0.0
    """
    image_bgr = cv2.resize(image_bgr, (320, 240))
    rgb       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Hand landmarks — right and left separated
    hr    = hand_detector.detect(mp_img)
    right = [0.0] * 63
    left  = [0.0] * 63
    for i, hl in enumerate(hr.handedness):
        coords = [v for lm in hr.hand_landmarks[i] for v in [lm.x, lm.y, lm.z]]
        if hl[0].category_name == 'Right':
            right = coords
        else:
            left = coords

    # Pose landmarks — full body skeleton
    pr   = pose_detector.detect(mp_img)
    pose = [v for lm in pr.pose_landmarks[0] for v in [lm.x, lm.y, lm.z]] \
           if pr.pose_landmarks else [0.0] * 99

    return right + left + pose   # 63 + 63 + 99 = 225


def build_jobs():
    """
    Scans Frames_Sentence_Level and returns a list of
    (pkl_name, [frame_paths], sentence_label) tuples.
    One job = one signer performing one sentence.
    97 sentences × 7 signers = ~679 jobs total.
    """
    jobs = []
    sentence_dirs = [d for d in sorted(FRAMES_DIR.iterdir()) if d.is_dir()]
    for sd in sentence_dirs:
        signer_dirs = [s for s in sorted(sd.iterdir()) if s.is_dir()]
        if signer_dirs:
            # Has signer subfolders (normal case)
            for sig in signer_dirs:
                frames = sorted(list(sig.glob('*.jpg')) + list(sig.glob('*.png')))
                if frames:
                    jobs.append((f"{sd.name}__{sig.name}", frames, sd.name))
        else:
            # Flat folder — treat whole sentence as one job
            frames = sorted(list(sd.glob('*.jpg')) + list(sd.glob('*.png')))
            if frames:
                jobs.append((sd.name, frames, sd.name))
    return jobs


def process_job(pkl_name, frames, label):
    """
    Takes a list of image paths for one signer+sentence,
    extracts landmarks from every STRIDE-th frame,
    pads/trims to SEQ_LEN frames,
    saves as .pkl dict with keys 'features' and 'label'.
    """
    selected = sorted(frames)[::STRIDE][:SEQ_LEN]
    features = []
    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        features.append(extract_landmarks(img))

    if not features:
        return False

    # Pad with zero vectors if fewer than SEQ_LEN frames
    if len(features) < SEQ_LEN:
        features += [[0.0] * FEAT_DIM] * (SEQ_LEN - len(features))
    else:
        features = features[:SEQ_LEN]

    out_path = OUTPUT_DIR / f"{pkl_name}.pkl"
    out_path.write_bytes(pickle.dumps({'features': features, 'label': label}))
    return True


def main():
    print("=" * 55)
    print("FEATURE EXTRACTION")
    print("=" * 55)
    print(f"Source    : {FRAMES_DIR}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Seq len   : {SEQ_LEN} frames per sample")
    print(f"Feat dim  : {FEAT_DIM} values per frame")
    print(f"Stride    : every {STRIDE} frames (speeds up without losing motion)")
    print(f"Resume    : {PROGRESS_FILE}")
    print("=" * 55)

    jobs = build_jobs()
    print(f"\nTotal jobs : {len(jobs)}  (~97 sentences × 7 signers)")

    # Load resume state
    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text())
    else:
        progress = {'processed': []}
    processed  = set(progress['processed'])
    to_process = [(n, f, l) for n, f, l in jobs if n not in processed]

    print(f"Already done : {len(processed)}")
    print(f"Remaining    : {len(to_process)}")

    if not to_process:
        print("\n✅ All jobs already done — skipping extraction")
        print("➡ Run: python3 scripts/04_train.py")
        return

    print(f"\nStarting extraction — interrupt anytime, resumes where it left off\n")

    start  = time.time()
    done   = 0
    errors = 0

    for pkl_name, frames, label in tqdm(to_process, desc="Extracting", unit="signer"):
        try:
            success = process_job(pkl_name, frames, label)
            if success:
                done += 1
                processed.add(pkl_name)
                progress['processed'] = list(processed)
                PROGRESS_FILE.write_text(json.dumps(progress))
            else:
                errors += 1
                tqdm.write(f"  ⚠ Empty result: {pkl_name}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                tqdm.write(f"  ⚠ Error [{pkl_name}]: {e}")

    elapsed = time.time() - start
    hand_detector.close()
    pose_detector.close()

    # Final summary
    pkl_count = len(list(OUTPUT_DIR.glob('*.pkl')))
    print("\n" + "=" * 55)
    print("EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"  PKL files created : {pkl_count}")
    print(f"  Errors            : {errors}")
    print(f"  Time taken        : {elapsed/60:.1f} minutes")
    print(f"  Output dir        : {OUTPUT_DIR}")
    print("=" * 55)

    if errors == 0:
        print("✅ All jobs completed with 0 errors")
    else:
        print(f"⚠ {errors} errors — check output above")

    print("➡ Next: python3 scripts/04_train.py")


if __name__ == "__main__":
    main()