"""
10_preview_landmarks.py
--------------------------
LOOK at what MediaPipe sees in your corpus clips, and COUNT how many people
and hands it finds. Use this when retrieval is near chance: it tells you
whether the landmarks even describe the signer.

Why it matters: the pipeline assumes one signer per clip. If a clip has two
people in frame, the pose model silently picks one and the hand model may
return the other person's hands - the embedding then mixes two people.

Modes
  python3 10_preview_landmarks.py --label "who are you"
        Makes preview_<label>_<n>.jpg contact sheets (8 frames each, landmarks
        drawn) for the first 2 clips of that folder, and prints the counts.
        Send me the JPG and I can see what the extractor sees.
  python3 10_preview_landmarks.py --video "data_raw_videos/who are you/who are you (1).MP4"
  python3 10_preview_landmarks.py --scan
        Checks EVERY clip (sampled frames) and prints, per clip:
        % of frames with 2+ people, % of frames with 3+ hands, % with a hand.
        Sorted worst first, plus a summary line.

Experiment flags (to see if hand detection can be improved, for one clip):
  python3 10_preview_landmarks.py --video "<clip>" --max_width 1920 --hand_conf 0.3
Look at "any hand on X% of frames" - compare with the default run.

Colours: right hand magenta, left hand cyan, first person green, any extra
person ORANGE (that's the one to worry about).
"""

import argparse
import os

# Silence MediaPipe's very chatty C++ logging (must be set before importing mediapipe).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from landmark_extraction_utils import check_model_files_exist

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def build_landmarkers():
    hand = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.PATHS["hand_model"],
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=4,
        min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE))
    pose = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.PATHS["pose_model"],
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=3,
        min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_POSE_TRACKING_CONFIDENCE))
    return hand, pose


def summarize_counts(per_frame):
    """per_frame: list of (n_people, n_hands). Pure function (unit-tested)."""
    n = max(len(per_frame), 1)
    return {
        "frames": len(per_frame),
        "multi_person": sum(1 for p, h in per_frame if p >= 2) / n,
        "many_hands": sum(1 for p, h in per_frame if h >= 3) / n,
        "any_hand": sum(1 for p, h in per_frame if h >= 1) / n,
    }


def _draw(frame, hand_res, pose_res):
    h, w = frame.shape[:2]
    for k, lms in enumerate(pose_res.pose_landmarks or []):
        col = (0, 200, 0) if k == 0 else (0, 140, 255)
        pts = [(int(l.x * w), int(l.y * h)) for l in lms]
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], col, 2, cv2.LINE_AA)
        for i in list(range(0, 11)) + [15, 16]:
            cv2.circle(frame, pts[i], 4, (0, 255, 255) if i < 11 else col, -1, cv2.LINE_AA)
    for lms, handed in zip(hand_res.hand_landmarks or [], hand_res.handedness or []):
        col = (255, 0, 255) if handed[0].category_name == "Right" else (255, 255, 0)
        pts = [(int(l.x * w), int(l.y * h)) for l in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, col, -1, cv2.LINE_AA)


def analyse(video_path, step=1, collect_frames=0):
    """Runs the models over the clip (every `step`-th frame). Returns
    (per_frame_counts, [annotated_frames])."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    wanted = set(np.linspace(0, max(total - 1, 0), collect_frames).round().astype(int).tolist()) if collect_frames else set()
    hand, pose = build_landmarkers()
    counts, shots, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0 or i in wanted:
            if bgr.shape[1] > config.MAX_FRAME_WIDTH:
                s = config.MAX_FRAME_WIDTH / bgr.shape[1]
                bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = int(i / fps * 1000)
            hr, pr = hand.detect_for_video(img, ts), pose.detect_for_video(img, ts)
            counts.append((len(pr.pose_landmarks or []), len(hr.hand_landmarks or [])))
            if i in wanted:
                vis = bgr.copy()
                _draw(vis, hr, pr)
                cv2.putText(vis, f"frame {i}  people={counts[-1][0]}  hands={counts[-1][1]}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                shots.append(vis)
        i += 1
    cap.release(); hand.close(); pose.close()
    return counts, shots


def contact_sheet(shots, cols=4, width=480):
    tiles = []
    for s in shots:
        h, w = s.shape[:2]
        tiles.append(cv2.resize(s, (width, int(h * width / w))))
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def list_clips():
    base = config.PATHS["data_raw_videos"]
    out = []
    for lab in sorted(os.listdir(base)):
        d = os.path.join(base, lab)
        if os.path.isdir(d) and not lab.startswith("."):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(VIDEO_EXTS) and not f.startswith("."):
                    out.append((lab, os.path.join(d, f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--label")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--clips_per_label", type=int, default=2)
    ap.add_argument("--max_width", type=int, default=None,
                    help="Try a larger frame width (e.g. 1920). Default is config.MAX_FRAME_WIDTH.")
    ap.add_argument("--hand_conf", type=float, default=None,
                    help="Try a lower hand confidence threshold (e.g. 0.3). Default is config's 0.5.")
    args = ap.parse_args()
    if args.max_width:
        config.MAX_FRAME_WIDTH = args.max_width
    if args.hand_conf:
        config.MIN_HAND_DETECTION_CONFIDENCE = config.MIN_HAND_PRESENCE_CONFIDENCE = \
            config.MIN_HAND_TRACKING_CONFIDENCE = args.hand_conf
    print(f"[settings] max_width={config.MAX_FRAME_WIDTH}  hand_conf={config.MIN_HAND_DETECTION_CONFIDENCE}")
    if not (args.video or args.label or args.scan):
        ap.error("give --video, --label or --scan")
    if not check_model_files_exist():
        return

    targets = []
    if args.video:
        targets = [("video", args.video)]
    elif args.label:
        c = [x for x in list_clips() if x[0] == args.label]
        if not c:
            print(f"No clips found for label '{args.label}'. Labels: {sorted({x[0] for x in list_clips()})}")
            return
        targets = c[:args.clips_per_label]

    if args.scan:
        rows = []
        clips = list_clips()
        for n, (lab, path) in enumerate(clips, 1):
            counts, _ = analyse(path, step=4)
            rows.append((lab, os.path.relpath(path, config.PATHS["data_raw_videos"]), summarize_counts(counts)))
            print(f"\r[scan] {n}/{len(clips)}", end="", flush=True)
        print("\n")
        rows.sort(key=lambda r: -(r[2]["multi_person"] + r[2]["many_hands"]))
        print(f"{'clip':<62}{'2+ people':>10}{'3+ hands':>10}{'any hand':>10}")
        for _, rel, s in rows:
            print(f"{rel[:61]:<62}{s['multi_person']*100:>9.0f}%{s['many_hands']*100:>9.0f}%{s['any_hand']*100:>9.0f}%")
        flagged = sum(1 for r in rows if r[2]["multi_person"] > 0.2)
        print(f"\n{flagged} of {len(rows)} clips show a second person in more than 20% of sampled frames.")
        return

    for k, (lab, path) in enumerate(targets, 1):
        counts, shots = analyse(path, step=1, collect_frames=8)
        s = summarize_counts(counts)
        out = f"preview_{os.path.basename(os.path.dirname(path)) if lab != 'video' else 'video'}_{k}.jpg".replace(" ", "_")
        sheet = contact_sheet(shots)
        if sheet is not None:
            cv2.imwrite(out, sheet)
        print(f"{path}\n   frames={s['frames']}  2+ people on {s['multi_person']*100:.0f}% of frames  "
              f"3+ hands on {s['many_hands']*100:.0f}%  any hand on {s['any_hand']*100:.0f}%  -> {out}")


if __name__ == "__main__":
    main()
