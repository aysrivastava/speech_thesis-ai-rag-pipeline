"""
record_webcam_and_test.py
----------------------------
Records a test clip straight from your Mac's webcam, saves it into
test_query_videos/, and immediately runs it through the step-5 semantic
search test - so you get "record -> see if it matches" in one sitting,
without manually copying phone recordings over.

Why webcam instead of your phone: your training clips are all landscape,
medium-distance shots with your full upper body in frame. Your Mac's
built-in webcam is natively landscape, so this sidesteps the exact framing
mismatch (portrait selfie, hands barely in frame) that caused the weak
match in your first real test.

Not numbered like 01-05: this isn't a new pipeline stage, it's a
convenience tool that wraps the recording step plus steps 1, 3 and 5
together for a fast live-test loop. It imports the exact same
landmark_extraction_utils / embedding_utils / retrieval_utils that those
numbered scripts use, so a webcam-recorded clip is processed identically
to a phone-recorded one - same code path either way.

STEP 6 ADDITION: if ANTHROPIC_API_KEY is set, each round first asks
"Question the signer is answering?" (Enter = none). With a question the
result goes through context_utils.process_result: LLM plausibility
re-ranking (bounded influence), out-of-vocabulary fallback and optional
enrollment. Without a key or question, behaviour is unchanged.

v4 CHANGES:

0. TWO TEST MODES - every round now starts by asking how you want to
   test: record live from the webcam (blind test, as below), or run the
   same test against a file that's already sitting in test_query_videos/
   (with a numbered picker). You said you'll mostly do live webcam demos
   but sometimes want to re-test an existing clip - this is that switch,
   asked fresh each round so you can freely mix the two in one session.
   The webcam itself (and the live overlay's models) are only opened the
   first time you actually pick webcam mode - picking "file" first costs
   you nothing extra.

1. BLIND TEST - no more picking a label before you sign.
   v2 made you choose which stored sign you were about to perform first,
   so it could look up that sign's original duration for the countdown
   timer. The problem: that defeats the point of a retrieval TEST - you're
   telling the system the answer before it guesses. (To be clear: the old
   label was only ever used for the on-screen text and the countdown
   timer, never fed into the similarity comparison itself - the match was
   always computed blind. But asking you to pick first still *felt* like
   giving away the answer, and there was no reason to ask.)

   Now: the target recording duration defaults to a flat 5.0 seconds
   (plenty for a single sign), you can type a custom number of seconds or
   just press Enter to accept the default, and you can always stop early
   with SPACE if you finish sooner. You perform any sign you've already
   stored, the system is never told which one, and the ranked result at
   the end is a genuinely blind guess.

2. LIVE LANDMARK OVERLAY - you can now see exactly what MediaPipe is
   detecting, in real time, on the preview itself:
     - Each detected hand's 21 landmarks + skeleton lines (magenta =
       MediaPipe's "Right", cyan = "Left")
     - The upper-body pose skeleton (green: shoulders/elbows/wrists/hips)
     - A small cluster of face-area REFERENCE points (yellow: nose, eyes,
       ears, mouth corners)

   Important nuance on that last one: this pipeline does NOT run a
   dedicated MediaPipe Face Landmarker / face mesh model (the ~468-point
   dense mesh you may have seen in MediaPipe demo images). The yellow
   points are the 11 face-area landmarks that are already part of the
   33-point POSE topology we extract anyway (indices 0-10: nose, eyes,
   ears, mouth corners) - showing them costs nothing extra, no new model,
   no change to the 225-dim feature vector or the existing vector DB.
   Adding a true dense face mesh would be a bigger, separate decision (new
   model file, a larger feature vector, re-extracting every stored clip) -
   see the README's "Architecture at a glance" section for the trade-off.
   This overlay reuses the EXACT SAME HandLandmarker/PoseLandmarker
   construction (same model files, same confidence thresholds) as the real
   extraction pipeline, so what you see live is representative of what
   actually gets extracted - it's purely a visualization running
   side-by-side, though; it never touches the saved video file or the
   embedding math. If it noticeably slows down the preview on your
   machine, flip SHOW_LANDMARK_OVERLAY to False below.

IMPORTANT - mirroring: the on-screen PREVIEW is mirrored (flipped left-
right) so it feels like looking in a mirror while you position yourself -
completely normal, same as every phone camera app. The overlay points are
mirrored to match. But the frame actually WRITTEN to the saved video file
is NOT flipped, and detection for the overlay itself runs on the
UNMIRRORED raw frame too. This matters: MediaPipe's handedness
("Left"/"Right") is read directly from the image, so if the saved file
were mirrored, your actual right hand would get permanently mislabeled as
"Left" - silently corrupting the right_hand/left_hand split in the 225-dim
feature vector, and making it inconsistent with your (unmirrored,
phone-recorded) training clips. Writing the unmirrored frame avoids that
entirely.

The green guide box is ALWAYS green - it's just a framing reference
("stay inside this box"), not a status indicator. Whether you're actively
recording is shown only in the top status bar text, never by the box
changing color, so there's nothing to misread as an error.

Controls:
  SPACE = skip the get-ready countdown / stop recording early
  Q     = quit (only works while not actively recording)

Run:
  python3 record_webcam_and_test.py
"""

import math
import os

import numpy as np
import time
from datetime import datetime

import cv2
import mediapipe as mp

import config
import context_utils
import llm_utils
from landmark_extraction_utils import (
    check_model_files_exist,
    _build_hand_landmarker,
    _build_pose_landmarker,
)
from retrieval_utils import run_semantic_search_test, list_stored_labels_with_duration

MIN_RECORD_FRAMES = 10              # discard a take shorter than this - too short to be a real sign
DEFAULT_TARGET_DURATION_SEC = 12.0   # auto mode: only a safety ceiling (the sign length is dynamic); timed mode: fixed length
# Hand-triggered recording: the clip starts when a hand first appears and ends when the hands have been
# out of frame for HAND_GONE_SEC (or at the max length). Nothing before/after the sign is kept.
AUTO_HAND_TRIM = True
PREROLL_SEC = 0.3                    # frames kept from just before the hand appeared
STILL_SEC = 1.0                      # hands visible but not moving this long (resting in view) = sign finished
STILL_MOTION_THRESH = 0.006          # mean per-landmark movement per frame (normalised coords) below this = still
TAIL_KEEP_SEC = 0.3                  # frames kept after the last movement / last sighting
HAND_GONE_SEC = 0.7                  # hands absent this long = sign finished
WAIT_FOR_HAND_TIMEOUT_SEC = 20.0     # give up if no hand shows up
MIN_TARGET_DURATION_SEC = 1.0
GET_READY_SECONDS = 3                # length of the "3...2...1...GO" countdown before recording starts
SAFETY_CAP_PADDING_SEC = 5.0         # hard ceiling = target duration + this, in case something hangs

GUIDE_BOX_COLOR = (0, 200, 0)        # constant - never changes with recording state, see module docstring

# Live landmark overlay - purely visual, see module docstring point 2.
# Flip to False if it noticeably slows down your preview framerate.
SHOW_LANDMARK_OVERLAY = True

RIGHT_HAND_COLOR = (255, 0, 255)     # magenta (BGR) - MediaPipe's "Right"
LEFT_HAND_COLOR = (255, 255, 0)      # cyan (BGR)     - MediaPipe's "Left"
POSE_BODY_COLOR = (0, 255, 0)        # green - shoulders/elbows/wrists/hips
POSE_FACE_COLOR = (0, 255, 255)      # yellow - nose/eyes/ears/mouth-corner reference points

# Canonical 21-point MediaPipe hand skeleton (same topology MediaPipe's own
# solutions.hands.HAND_CONNECTIONS uses) - hardcoded here instead of
# depending on that (older, sometimes-absent) API surface.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                                # wrist -> pinky mcp (palm base)
]

# Deliberately a reduced subset of the 33-point pose topology - just the
# upper body, since legs/feet are rarely even in frame for a signing
# recording and would just add visual clutter.
POSE_BODY_CONNECTIONS = [
    (11, 12),            # shoulder to shoulder
    (11, 13), (13, 15),  # left arm: shoulder-elbow-wrist
    (12, 14), (14, 16),  # right arm: shoulder-elbow-wrist
    (11, 23), (12, 24),  # shoulders to hips (torso reference)
    (23, 24),             # hip to hip
]
POSE_BODY_POINT_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)

# Indices 0-10 of the 33-point pose topology are already face-area points
# (nose, eyes, ears, mouth corners) - see module docstring point 2.
POSE_FACE_POINT_INDICES = tuple(range(0, 11))
POSE_FACE_CONNECTIONS = [(9, 10)]  # mouth-corner to mouth-corner


class LiveOverlay:
    """
    Bundles the live-preview-only MediaPipe landmarkers plus a monotonic
    timestamp clock, so the preview can show real-time hand/pose tracking.

    This is COMPLETELY SEPARATE from the landmarkers used by the real
    extraction pipeline (landmark_extraction_utils.extract_landmark_sequence),
    which run AFTER a clip is saved, processing the saved file frame-by-frame
    on its own 0-based clock. These exist purely so you can see your
    hands/pose tracked live on screen; they never touch the saved video file
    or the embedding - that's still computed entirely from the saved file,
    exactly as before.

    Uses the exact same _build_hand_landmarker()/_build_pose_landmarker()
    as the real pipeline (same model files, same thresholds), so what you
    see live is representative of what actually gets extracted.
    """

    def __init__(self):
        self.hand_landmarker = _build_hand_landmarker()
        self.pose_landmarker = _build_pose_landmarker()
        self._start = time.time()
        self._last_ms = -1

    def detect(self, frame_bgr):
        """Returns (hand_result, pose_result), or (None, None) on any issue
        (overlay disabled, a detection hiccup, etc) - this must never crash
        the live preview loop."""
        if not SHOW_LANDMARK_OVERLAY:
            return None, None
        try:
            # VIDEO mode needs a strictly increasing timestamp per call.
            ms = int((time.time() - self._start) * 1000)
            if ms <= self._last_ms:
                ms = self._last_ms + 1
            self._last_ms = ms

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, ms)
            pose_result = self.pose_landmarker.detect_for_video(mp_image, ms)
            return hand_result, pose_result
        except Exception as e:
            print(f"[overlay] WARNING: live detection hiccup, skipping this frame's overlay ({e})")
            return None, None

    def close(self):
        self.hand_landmarker.close()
        self.pose_landmarker.close()


def _landmarks_to_mirrored_points(landmarks, w, h):
    """Converts normalized [0,1] landmarks to pixel coords, mirrored on x to
    match the flipped preview (detection itself always runs on the raw,
    unmirrored frame - this mirroring is for on-screen alignment only)."""
    return [(w - int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def _draw_hand(display, hand_landmarks, w, h, color):
    pts = _landmarks_to_mirrored_points(hand_landmarks, w, h)
    for a, b in HAND_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(display, (px, py), 3, color, -1, cv2.LINE_AA)


def _draw_pose(display, pose_landmarks, w, h):
    pts = _landmarks_to_mirrored_points(pose_landmarks, w, h)
    for a, b in POSE_BODY_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], POSE_BODY_COLOR, 2, cv2.LINE_AA)
    for a, b in POSE_FACE_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], POSE_FACE_COLOR, 1, cv2.LINE_AA)
    for idx in POSE_BODY_POINT_INDICES:
        cv2.circle(display, pts[idx], 4, POSE_BODY_COLOR, -1, cv2.LINE_AA)
    for idx in POSE_FACE_POINT_INDICES:
        cv2.circle(display, pts[idx], 2, POSE_FACE_COLOR, -1, cv2.LINE_AA)


def draw_landmark_overlay(display, hand_result, pose_result, w, h):
    """Never lets a drawing issue crash the live preview."""
    try:
        if hand_result and hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                color = RIGHT_HAND_COLOR if label == "Right" else LEFT_HAND_COLOR
                _draw_hand(display, hand_landmarks, w, h, color)
        if pose_result and pose_result.pose_landmarks:
            _draw_pose(display, pose_result.pose_landmarks[0], w, h)
    except Exception as e:
        print(f"[overlay] WARNING: couldn't draw this frame's overlay ({e})")


def draw_legend(display, h):
    items = [
        ("right hand", RIGHT_HAND_COLOR),
        ("left hand", LEFT_HAND_COLOR),
        ("body", POSE_BODY_COLOR),
        ("face pts", POSE_FACE_COLOR),
    ]
    x, y = 10, h - 12
    for text, color in items:
        cv2.circle(display, (x, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(display, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x += 10 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] + 20


def ask_for_duration(default_duration=DEFAULT_TARGET_DURATION_SEC):
    """
    Terminal prompt (not the cv2 window - OpenCV windows can't take typed
    input). No label is asked for - this is a BLIND test: perform any sign
    you've already stored, and the system guesses from the recording alone.
    Returns target_duration_sec, or None to go back (cancel this take).
    """
    print("\n" + "-" * 70)
    raw = input(
        f"Press Enter = DYNAMIC (starts when your hand appears, ends when it leaves the frame; "
        f"safety max {default_duration:.0f}s), or type a number of seconds for a hard limit, or 'q' to go back: "
    ).strip().lower()

    if raw == "q":
        return None
    if not raw:
        return default_duration
    try:
        return max(float(raw), MIN_TARGET_DURATION_SEC)
    except ValueError:
        print(f"Couldn't parse '{raw}' as a number - using {default_duration:.1f}s.")
        return default_duration


def ask_for_mode():
    """
    Asked fresh at the start of every round, since you may want to mix
    live webcam demos with re-testing a file you already have. Returns
    "webcam", "file", or None to quit.
    """
    print("\n" + "=" * 70)
    print("How do you want to test?")
    print("   [1] Record from webcam (blind test)")
    print("   [2] Test an existing file from test_query_videos/")
    raw = input("Choose 1 or 2, or press Enter to quit: ").strip()
    if not raw:
        return None
    if raw == "1":
        return "webcam"
    if raw == "2":
        return "file"
    print(f"Didn't recognize '{raw}' - defaulting to webcam.")
    return "webcam"


_context_hint_shown = False


def ask_for_question():
    """
    Optional CONTEXT: the question the signer is about to answer (step 6).
    Asked BEFORE recording/picking a file, because in a real exchange the
    question comes first and the sign is the answer. Enter = no question
    (plain retrieval, exactly as before). Only offered when an API key is
    set, so people without one aren't nagged.
    """
    global _context_hint_shown
    if not llm_utils.llm_is_configured():
        if not _context_hint_shown:
            print("\n(Tip: set ANTHROPIC_API_KEY to unlock context mode - you could then give the "
                  "question the signer is answering, and an LLM would re-rank the matches and "
                  "handle out-of-vocabulary answers. See 06_llm_context_rerank.py.)")
            _context_hint_shown = True
        return None
    q = input("Question the signer is answering? (Enter = none, plain retrieval): ").strip()
    return q or None



def slugify_label(label):
    import re
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def ask_for_intended_sign():
    """Asks WHICH sign you are about to perform so the saved file is named after it
    (<sign>__<timestamp>.mp4) and the result can be marked right/wrong afterwards.
    The label is only used for the filename and the printout - never for retrieval."""
    labels = sorted(set(l for l, _ in list_stored_labels_with_duration()))
    if not labels:
        return None
    print("\n" + "-" * 70)
    print("Which sign will you perform? (used ONLY to name the file and score the result;")
    print("the system is not told.)")
    for i, l in enumerate(labels, 1):
        print(f"   [{i}] {l}")
    raw = input("Number (Enter = skip, file is named blind_test_...): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(labels):
        return labels[int(raw) - 1]
    return None


def label_from_filename(path):
    """'comb_your_hair__20261011_1647.mp4' -> the stored label whose slug matches, else None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "__" not in stem:
        return None
    slug = stem.split("__")[0]
    for l, _ in list_stored_labels_with_duration():
        if slugify_label(l) == slug:
            return l
    return None


def print_verdict(result, truth):
    """Right/wrong line against the sign named in the filename."""
    if not result or not truth:
        return
    ranked = []
    for row in result["scored"]:
        if row[0] not in ranked:
            ranked.append(row[0])
    rank = ranked.index(truth) + 1 if truth in ranked else None
    ok = ranked[0] == truth
    print("\n" + "=" * 70)
    print(f"{'CORRECT' if ok else 'WRONG'}  | you signed: '{truth}' | system said: '{ranked[0]}' | "
          f"true sign ranked #{rank}")
    print("=" * 70)
    _TALLY.append((ok, rank))
    n = len(_TALLY)
    print(f"Session tally: top-1 {sum(o for o, _ in _TALLY)}/{n}   "
          f"top-3 {sum(1 for _, r in _TALLY if r and r <= 3)}/{n}")


_TALLY = []


def report_result(result, question, blind_banner_text):
    """Shared by webcam and file mode. With a question (or a calibrated
    out-of-vocabulary floor) hand off to the context layer; otherwise the
    original plain-retrieval banner."""
    if not result:
        return
    if question or config.OOV_SIM_FLOOR is not None:
        context_utils.process_result(result, question)
        return
    print("\n" + "*" * 70)
    print(f"{blind_banner_text}: '{result['best_label']}'")
    if blind_banner_text.startswith("SYSTEM'S BLIND GUESS"):
        print("(It was never told which sign you actually performed - compare this")
        print(" to what you actually just signed.)")
    print("*" * 70)


def list_query_video_files():
    query_dir = config.PATHS["test_query_videos"]
    if not os.path.isdir(query_dir):
        return []
    return sorted(
        f for f in os.listdir(query_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    )


def ask_for_query_file():
    """Lists whatever's actually sitting in test_query_videos/ and lets you
    pick one by number. Returns a full path, or None to go back."""
    files = list_query_video_files()
    print("\n" + "-" * 70)
    if not files:
        print(f"No video files found in {config.PATHS['test_query_videos']}")
        print("Drop a clip in there first, or go back and record from the webcam instead.")
        input("Press Enter to go back...")
        return None

    print("Files in test_query_videos/:")
    for i, f in enumerate(files, start=1):
        print(f"   [{i}] {f}")
    raw = input("Type a number to test that file, or press Enter to go back: ").strip()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(files):
            return os.path.join(config.PATHS["test_query_videos"], files[idx - 1])
    print(f"'{raw}' isn't one of the listed numbers - going back.")
    return None


def open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam (index 0). If another app is using the camera, close it and retry."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3 or fps > 120:
        fps = 30.0
        print(f"[webcam] Camera didn't report a usable fps - assuming {fps:.0f}fps for the saved file.")
    else:
        print(f"[webcam] Camera reports {fps:.1f}fps.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[webcam] Frame size: {actual_w}x{actual_h} (landscape, same orientation as your training clips)")

    return cap, fps, actual_w, actual_h


def guide_box_coords(w, h):
    return int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97)


def draw_top_bar(display_frame, w, text, bar_color=(255, 255, 255)):
    cv2.rectangle(display_frame, (0, 0), (w, 34), (15, 15, 15), -1)
    cv2.putText(display_frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_color, 1, cv2.LINE_AA)


def draw_progress_bar(display_frame, w, h, fraction, label_text):
    """A filling bar just under the top status bar, showing elapsed/target time."""
    bar_x1, bar_y1 = 10, 42
    bar_x2, bar_y2 = w - 10, 58
    cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * max(0.0, min(1.0, fraction)))
    cv2.rectangle(display_frame, (bar_x1, bar_y1), (fill_x2, bar_y2), (0, 220, 255), -1)
    cv2.putText(display_frame, label_text, (bar_x1 + 4, bar_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def run_get_ready_countdown(cap, w, h, overlay, window_name):
    """
    Shows the guide box + live landmark overlay + a big countdown number for
    GET_READY_SECONDS so the user can position themselves before recording
    starts. SPACE skips straight to recording; Q quits. Returns True to
    proceed, False to quit.
    """
    start = time.time()
    x1, y1, x2, y2 = guide_box_coords(w, h)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[webcam] Lost the camera feed.")
            return False

        hand_result, pose_result = overlay.detect(frame)

        remaining = GET_READY_SECONDS - (time.time() - start)

        display = cv2.flip(frame, 1)
        draw_landmark_overlay(display, hand_result, pose_result, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), GUIDE_BOX_COLOR, 2)
        draw_top_bar(display, w, "Get ready to perform your sign...  (SPACE = start now, Q = quit)")
        draw_legend(display, h)

        if remaining > 0:
            count_text = str(math.ceil(remaining))
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
            tx = (w - text_size[0]) // 2
            ty = (h + text_size[1]) // 2
            cv2.putText(display, count_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 220, 255), 8, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False
        if remaining <= 0:
            return True


def _hands_seen(hand_result):
    try:
        return bool(hand_result is not None and hand_result.hand_landmarks)
    except Exception:
        return False


def _hand_points(hand_result):
    """All detected hand landmarks as one (N,2) array sorted by (hand count, x of wrist) so frame-to-frame
    comparison is stable; None if no hands."""
    try:
        hands = hand_result.hand_landmarks
        if not hands:
            return None
        hands = sorted(hands, key=lambda h: h[0].x)
        return np.array([[lm.x, lm.y] for h in hands for lm in h], dtype=np.float64)
    except Exception:
        return None


def record_one_take_auto(cap, fps, w, h, out_path, max_sign_sec, overlay):
    """
    Hand-triggered take. After the countdown the camera waits for a hand; the saved clip runs from
    PREROLL_SEC before the hand appeared until HAND_GONE_SEC after it left the frame (or max_sign_sec after
    it appeared). Returns frames written, or None (quit / no hand).
    """
    from collections import deque
    window_name = "Record test clip (SPACE = stop, Q = quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)
    if not run_get_ready_countdown(cap, w, h, overlay, window_name):
        cv2.destroyWindow(window_name)
        return None

    x1, y1, x2, y2 = guide_box_coords(w, h)
    pre = deque(maxlen=max(int(PREROLL_SEC * fps), 1))
    frames, started, t_start, t_last_seen = [], False, None, None
    prev_pts, t_last_motion, idx_motion, idx_seen = None, None, 0, 0
    t_wait = time.time()
    print("[webcam] Waiting for your hand... (starts when a hand appears, stops when it leaves the frame)")
    result = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[webcam] Lost the camera feed.")
            break
        hand_result, pose_result = overlay.detect(frame)
        seen = _hands_seen(hand_result)
        now = time.time()
        if not started:
            pre.append(frame)
            if seen:
                started, t_start, t_last_seen = True, now, now
                frames = list(pre)
                t_last_motion, idx_motion, idx_seen = now, len(frames), len(frames)
            elif now - t_wait > WAIT_FOR_HAND_TIMEOUT_SEC:
                print("[webcam] No hand seen - cancelled.")
                break
            status = "Raise your hand(s) to start signing..."
            bar = (0, 160, 255)
        else:
            frames.append(frame)
            if seen:
                t_last_seen = now
                idx_seen = len(frames)
                pts = _hand_points(hand_result)
                moved = (pts is None or prev_pts is None or pts.shape != prev_pts.shape
                         or float(np.abs(pts - prev_pts).mean()) > STILL_MOTION_THRESH)
                prev_pts = pts
                if moved:
                    t_last_motion, idx_motion = now, len(frames)
            else:
                prev_pts = None
            el = now - t_start
            status = f"SIGNING  {el:.1f}s  (stop moving or drop your hands to finish)"
            bar = (0, 100, 255)
            keep_after = int(TAIL_KEEP_SEC * fps)
            if now - t_last_seen > HAND_GONE_SEC:
                print(f"[webcam] Hands left the frame - sign was {t_last_seen - t_start:.1f}s.")
                frames = frames[:idx_seen + keep_after]
                break
            if el > 1.0 and now - t_last_motion > STILL_SEC:
                print(f"[webcam] Hands stopped moving - sign was {t_last_motion - t_start:.1f}s.")
                frames = frames[:idx_motion + keep_after]
                break
            if el >= max_sign_sec:
                print(f"[webcam] Reached the {max_sign_sec:.1f}s maximum.")
                break
        display = cv2.flip(frame, 1)
        draw_landmark_overlay(display, hand_result, pose_result, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), GUIDE_BOX_COLOR, 2)
        draw_top_bar(display, w, status, bar_color=bar)
        draw_legend(display, h)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            frames = []
            break
        if key == ord(" ") and started:
            break
    cv2.destroyWindow(window_name)
    if len(frames) < MIN_RECORD_FRAMES:
        return None
    # drop the post-roll frames after the last sighting beyond a short tail
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return len(frames)


def record_one_take(cap, fps, w, h, out_path, target_duration_sec, overlay):
    """
    Runs the get-ready countdown, then records for target_duration_sec
    (auto-stopping when the timer hits zero), writing raw (unmirrored)
    frames to out_path. SPACE during recording stops it early. Returns
    the number of frames written, or None if the user quit instead.
    """
    window_name = "Record test clip (SPACE = skip/stop early, Q = quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    proceed = run_get_ready_countdown(cap, w, h, overlay, window_name)
    if not proceed:
        cv2.destroyWindow(window_name)
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frame_count = 0
    start_time = time.time()
    safety_cap = target_duration_sec + SAFETY_CAP_PADDING_SEC
    x1, y1, x2, y2 = guide_box_coords(w, h)

    print("[webcam] Recording started...")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[webcam] Lost the camera feed.")
            writer.release()
            cv2.destroyWindow(window_name)
            return frame_count if frame_count >= MIN_RECORD_FRAMES else None

        writer.write(frame)  # unmirrored - see module docstring
        frame_count += 1

        hand_result, pose_result = overlay.detect(frame)

        elapsed = time.time() - start_time
        remaining = max(0.0, target_duration_sec - elapsed)

        display = cv2.flip(frame, 1)
        draw_landmark_overlay(display, hand_result, pose_result, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), GUIDE_BOX_COLOR, 2)
        draw_top_bar(
            display, w,
            f"RECORDING  -  {remaining:.1f}s left  (SPACE = stop early)",
            bar_color=(0, 100, 255),
        )
        draw_progress_bar(display, w, h, elapsed / target_duration_sec, f"{frame_count} frames")
        draw_legend(display, h)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            print(f"[webcam] Stopped early by request - {frame_count} frames captured.")
            break
        if elapsed >= target_duration_sec:
            print(f"[webcam] Timer finished - {frame_count} frames captured.")
            break
        if elapsed >= safety_cap:
            print(f"[webcam] Hit the safety cap ({safety_cap:.0f}s) - stopping automatically.")
            break

    writer.release()
    cv2.destroyWindow(window_name)
    return frame_count


def _print_stored_signs_once():
    stored = list_stored_labels_with_duration()
    print("\n" + "-" * 70)
    if stored:
        print("Signs currently in your vector DB (for your reference only - the system")
        print("is never told which one you're about to perform):")
        for label, duration in stored:
            print(f"   - {label}   (originally {duration:.1f}s)")
    else:
        print("Your vector DB is empty (or 04_hosting_vector_db_and_storing.py hasn't run yet).")
        print("You can still record a take, but there'll be nothing to compare it to.")


def main():
    print("=" * 70)
    print("TEST THE VECTOR DB - FROM A LIVE WEBCAM TAKE, OR AN EXISTING FILE")
    print("=" * 70)

    config.ensure_dirs()

    if not check_model_files_exist():
        print("\nStopping here until the model files are in place.")
        return

    # Webcam + the live-overlay models are opened lazily, only the first
    # time "webcam" mode is actually chosen - picking "file" mode doesn't
    # pay that startup cost at all.
    cap = fps = w = h = None
    overlay = None
    shown_stored_signs = False

    try:
        while True:
            mode = ask_for_mode()
            if mode is None:
                print("\nQuitting.")
                break

            # Context (step 6): the question comes BEFORE the sign, like in
            # a real exchange. None = plain retrieval, exactly as before.
            question = ask_for_question()

            if mode == "webcam":
                if cap is None:
                    cap, fps, w, h = open_webcam()
                    print("[webcam] Loading MediaPipe models for the live landmark overlay "
                          "(hands + pose/face points)...")
                    overlay = LiveOverlay()
                    print("[webcam] Overlay ready. (Set SHOW_LANDMARK_OVERLAY = False at the "
                          "top of this file to disable it.)")

                if not shown_stored_signs:
                    _print_stored_signs_once()
                    shown_stored_signs = True

                duration = ask_for_duration()
                if duration is None:
                    continue  # back to mode prompt, not a full quit

                intended = ask_for_intended_sign()
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = (f"{slugify_label(intended)}__{stamp}.mp4" if intended
                            else f"blind_test_{stamp}.mp4")
                out_path = os.path.join(config.PATHS["test_query_videos"], filename)

                print(f"\nAbout to record a blind test take - target duration {duration:.1f}s.")
                print("Perform ANY sign that's already in your vector DB - the system won't be told which one.")
                print("Position yourself so your head and signing hand(s) fit inside the green box.")
                print("A 3-2-1 countdown gives you time to get ready before recording starts.")

                if AUTO_HAND_TRIM and SHOW_LANDMARK_OVERLAY:
                    frame_count = record_one_take_auto(cap, fps, w, h, out_path, duration, overlay)
                else:
                    frame_count = record_one_take(cap, fps, w, h, out_path, duration, overlay)

                if frame_count is None:
                    continue  # Q during countdown/recording - back to mode prompt

                if frame_count < MIN_RECORD_FRAMES:
                    print(f"\n[WARN] Only {frame_count} frames captured (need at least {MIN_RECORD_FRAMES}) - "
                          f"discarding this take.")
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                else:
                    print(f"\n[save] Saved {frame_count} frames to: {out_path}")
                    try:
                        result = run_semantic_search_test(out_path, top_k=5, query_label_hint="blind_test")
                        report_result(result, question, "SYSTEM'S BLIND GUESS")
                        print_verdict(result, intended)
                    except FileNotFoundError as e:
                        print(f"\n[ERROR] {e}")

            elif mode == "file":
                path = ask_for_query_file()
                if path is None:
                    continue  # back to mode prompt

                print(f"\nTesting existing file: {path}")
                try:
                    result = run_semantic_search_test(
                        path, top_k=5,
                        query_label_hint=os.path.splitext(os.path.basename(path))[0],
                    )
                    report_result(result, question, "BEST MATCH")
                    print_verdict(result, label_from_filename(path))
                except FileNotFoundError as e:
                    print(f"\n[ERROR] {e}")
    finally:
        if overlay is not None:
            overlay.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[webcam] Done.")


if __name__ == "__main__":
    main()
