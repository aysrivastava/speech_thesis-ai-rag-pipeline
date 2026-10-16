"""
landmark_extraction_utils.py
-----------------------------
Shared MediaPipe extraction logic, used by BOTH:
  - 01_extracting_features_from_video_using_mediapipe.py (bulk extraction
    of your training/reference videos into the vector DB)
  - 05_semantic_search_test.py (extracting features from a NEW clip you
    just recorded, so it can be compared against the DB)

Why this is its own file instead of copy-pasted into both scripts: if the
extraction logic ever differs even slightly between "building the DB" and
"querying the DB", cosine similarity between the two becomes meaningless -
you'd be comparing apples processed one way to oranges processed another.
Importing one shared function guarantees both paths are identical.

Key differences from the OLD 03_extract.py you shared, and why:

1. VIDEO running mode instead of IMAGE mode.
   MediaPipe's Tasks API has a `running_mode`. The old code used IMAGE mode
   frame-by-frame, which treats every frame as a brand-new, unrelated
   picture. VIDEO mode instead takes a timestamp per frame and uses it to
   smooth/track landmarks across time, which is exactly what we want for a
   moving hand - it reduces frame-to-frame jitter a lot.

2. "Hold last known landmark" instead of zero-filling missed detections.
   The old code wrote all-zero vectors for any frame where a hand/pose
   wasn't detected. A real hand position suddenly "teleporting" to (0,0,0)
   for one frame and back creates a huge, fake spike in velocity - and our
   new embedding (see 03_...py) is partly built from velocity statistics.
   One missed frame could quietly dominate the whole embedding. Holding the
   last known position is a far more honest stand-in for "the hand didn't
   move it just wasn't detected".

3. No forced 320x240 downscale. We only shrink frames that are LARGER than
   MAX_FRAME_WIDTH, to preserve handshape detail.

4. No fixed-length padding to SEQ_LEN=50. We keep the sequence at whatever
   length the real clip is. This matters because 03's embedding is a
   statistical pooling over time (mean/std), which works for sequences of
   any length - there's no architecture to pad for. Shorter clips just
   contribute fewer frames to the statistics; nothing is wasted on padding.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


def _build_hand_landmarker():
    # delegate=CPU is passed EXPLICITLY (not just relying on the library
    # default). On macOS, several Tasks-API calculators in this graph
    # (specifically TensorsToDetectionsCalculator, via an internal
    # "DrishtiMetalHelper") try to grab a Metal/GPU graph service even when
    # no GPU was requested, and abort with "Check failed: service_ Service
    # is unavailable" if that service was never registered. Forcing CPU
    # here explicitly is the confirmed workaround for that macOS-specific
    # crash (see https://github.com/google-ai-edge/mediapipe/issues/5568).
    base_options = mp_python.BaseOptions(
        model_asset_path=config.PATHS["hand_model"],
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _build_pose_landmarker():
    # Same explicit CPU delegate, same reason as _build_hand_landmarker()
    # above - see the comment there.
    base_options = mp_python.BaseOptions(
        model_asset_path=config.PATHS["pose_model"],
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_POSE_TRACKING_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _hand_result_to_vectors(result):
    """
    Returns (right_hand_vec_63, left_hand_vec_63, right_detected, left_detected).
    MediaPipe tells us handedness ("Left"/"Right") per detected hand, which
    is the LIVE person's left/right as MediaPipe sees it in a mirrored
    webcam frame - we just trust its label directly, same as the old code.
    """
    right_vec = np.zeros(config.RIGHT_HAND_DIM, dtype=np.float32)
    left_vec = np.zeros(config.LEFT_HAND_DIM, dtype=np.float32)
    right_detected = False
    left_detected = False

    if result.hand_landmarks:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
            ).flatten()
            if label == "Right":
                right_vec = coords
                right_detected = True
            elif label == "Left":
                left_vec = coords
                left_detected = True

    return right_vec, left_vec, right_detected, left_detected


def _pose_result_to_vector(result):
    pose_vec = np.zeros(config.POSE_DIM, dtype=np.float32)
    detected = False
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        pose_vec = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        ).flatten()
        detected = True
    return pose_vec, detected


def extract_landmark_sequence(video_path, verbose=True):
    """
    Runs MediaPipe Hand + Pose landmarkers over every frame of video_path
    in VIDEO mode, and returns a dict with:
        "features"        : np.ndarray of shape (T, 225), float32
        "num_frames"      : T
        "fps"             : detected video FPS
        "duration_sec"    : T / fps
        "hand_detect_rate": fraction of frames where >=1 hand was detected
        "pose_detect_rate": fraction of frames where pose was detected

    T is whatever the real clip length is - NOT padded/truncated to a fixed
    length (see module docstring, point 4).
    """
    if verbose:
        print(f"\n[extract] Opening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # sane fallback if the container doesn't report fps
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"[extract]   reported fps={fps:.2f}, reported frame count hint={total_frames_hint}")

    hand_landmarker = _build_hand_landmarker()
    pose_landmarker = _build_pose_landmarker()

    frame_vectors = []
    hand_detect_flags = []
    pose_detect_flags = []

    # "hold last known" state
    last_right = np.zeros(config.RIGHT_HAND_DIM, dtype=np.float32)
    last_left = np.zeros(config.LEFT_HAND_DIM, dtype=np.float32)
    last_pose = np.zeros(config.POSE_DIM, dtype=np.float32)

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        if w > config.MAX_FRAME_WIDTH:
            scale = config.MAX_FRAME_WIDTH / float(w)
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires a monotonically increasing timestamp in ms.
        timestamp_ms = int((frame_idx / fps) * 1000)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        right_vec, left_vec, right_seen, left_seen = _hand_result_to_vectors(hand_result)
        pose_vec, pose_seen = _pose_result_to_vector(pose_result)

        if right_seen:
            last_right = right_vec
        if left_seen:
            last_left = left_vec
        if pose_seen:
            last_pose = pose_vec

        full_vec = np.concatenate([last_right, last_left, last_pose])
        frame_vectors.append(full_vec)
        hand_detect_flags.append(right_seen or left_seen)
        pose_detect_flags.append(pose_seen)

        if verbose and frame_idx % 15 == 0:
            print(
                f"[extract]   frame {frame_idx:4d} | "
                f"right_hand={'Y' if right_seen else '.'} "
                f"left_hand={'Y' if left_seen else '.'} "
                f"pose={'Y' if pose_seen else '.'}"
            )

        frame_idx += 1

    cap.release()
    hand_landmarker.close()
    pose_landmarker.close()

    elapsed = time.time() - t_start
    features = np.stack(frame_vectors, axis=0) if frame_vectors else np.zeros((0, config.FRAME_FEATURE_DIM))
    num_frames = features.shape[0]
    hand_detect_rate = float(np.mean(hand_detect_flags)) if hand_detect_flags else 0.0
    pose_detect_rate = float(np.mean(pose_detect_flags)) if pose_detect_flags else 0.0

    if verbose:
        print(f"[extract] Done in {elapsed:.1f}s | total frames extracted: {num_frames}")
        print(f"[extract]   feature matrix shape: {features.shape}  (T x 225)")
        print(f"[extract]   hand detected on {hand_detect_rate*100:.1f}% of frames")
        print(f"[extract]   pose detected on {pose_detect_rate*100:.1f}% of frames")
        if hand_detect_rate < 0.5:
            print("[extract]   WARNING: hands were detected on less than half the frames. "
                  "Check lighting/framing - this clip's embedding may be unreliable.")

    return {
        "features": features.astype(np.float32),
        "num_frames": num_frames,
        "fps": fps,
        "duration_sec": num_frames / fps if fps else 0.0,
        "hand_detect_rate": hand_detect_rate,
        "pose_detect_rate": pose_detect_rate,
    }


def check_model_files_exist():
    """Prints a clear, actionable message if the .task model files are missing."""
    missing = []
    for key in ("hand_model", "pose_model"):
        path = config.PATHS[key]
        if not os.path.isfile(path):
            missing.append((key, path))

    if not missing:
        return True

    print("\n[ERROR] Missing MediaPipe model file(s):")
    for key, path in missing:
        print(f"   - {key}: expected at {path}")
    print("\nIf you already downloaded these for the old project, just copy")
    print("hand_landmarker.task and pose_landmarker.task into:")
    print(f"   {os.path.dirname(config.PATHS['hand_model'])}")
    print("\nOtherwise, download them directly with:")
    print("   mkdir -p mediapipe_models")
    print("   curl -L -o mediapipe_models/hand_landmarker.task \\")
    print("     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print("   curl -L -o mediapipe_models/pose_landmarker.task \\")
    print("     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
    return False
