import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, SEQ_LEN, FEAT_DIM, HIDDEN

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import pickle
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque


class SignClassifier(nn.Module):
    def __init__(self, input_dim=225, hidden=256, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


# ── Load model ────────────────────────────────────────────────────────────────
ck          = torch.load(str(PATHS['ckpt_best']), map_location='cpu')
SEQ         = ck['seq_len']
FDIM        = ck['feat_dim']
CKPT_HIDDEN = ck['model_state']['lstm.weight_ih_l0'].shape[0] // 4

with open(PATHS['label_encoder'], 'rb') as f:
    le = pickle.load(f)

model = SignClassifier(input_dim=FDIM, hidden=CKPT_HIDDEN, num_classes=ck['num_classes'])
model.load_state_dict(ck['model_state'])
model.eval()
print(f"✅ Model loaded — {ck['num_classes']} classes  |  hidden={CKPT_HIDDEN}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(PATHS['hand_model'])),
        num_hands=2, min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        running_mode=vision.RunningMode.IMAGE))

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(PATHS['pose_model'])),
        min_pose_detection_confidence=0.4,
        running_mode=vision.RunningMode.IMAGE))


def extract_landmarks(image_bgr):
    resized = cv2.resize(image_bgr, (320, 240))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hr    = hand_detector.detect(mp_img)
    right = [0.0] * 63
    left  = [0.0] * 63
    for i, hl in enumerate(hr.handedness):
        coords = [v for lm in hr.hand_landmarks[i] for v in [lm.x, lm.y, lm.z]]
        if hl[0].category_name == 'Right': right = coords
        else: left = coords

    pr   = pose_detector.detect(mp_img)
    pose = [v for lm in pr.pose_landmarks[0] for v in [lm.x, lm.y, lm.z]] \
           if pr.pose_landmarks else [0.0] * 99

    hand_detected = len(hr.hand_landmarks) > 0
    return right + left + pose, hand_detected


def predict(frames):
    seq = frames.copy()
    if len(seq) < SEQ:
        seq += [[0.0] * FDIM] * (SEQ - len(seq))
    else:
        seq = seq[:SEQ]
    with torch.no_grad():
        probs = torch.softmax(model(torch.FloatTensor([seq])), dim=1)[0]
        top5  = torch.topk(probs, min(5, ck['num_classes']))
    return [(le.inverse_transform([i.item()])[0], p.item() * 100)
            for p, i in zip(top5.values, top5.indices)]


def draw_ui(frame, state, results, sign_frames, total_frames):
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 80), (15, 15, 15), -1)

    STATE_COLORS = {
        'WAITING'    : (120, 120, 120),
        'SIGNING'    : (0, 220, 255),
        'PREDICTING' : (0, 255, 100),
        'RESULT'     : (0, 255, 100),
    }
    state_color = STATE_COLORS.get(state, (255, 255, 255))

    if state == 'WAITING':
        cv2.putText(frame, "Waiting — show your hands to start",
                    (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(frame, "Position hands clearly in frame",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    elif state == 'SIGNING':
        cv2.putText(frame, f"SIGNING...  ({sign_frames} frames captured)",
                    (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)
        cv2.putText(frame, "Keep signing — prediction runs when you stop",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        # Red REC dot
        cv2.circle(frame, (w - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    elif state in ('PREDICTING', 'RESULT'):
        if results:
            top_label, top_conf = results[0]
            color = (0, 255, 100) if top_conf >= 60 else \
                    (0, 200, 255) if top_conf >= 35 else (100, 130, 255)
            cv2.putText(frame, f"{top_label}",
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, f"Confidence: {top_conf:.1f}%   "
                               f"(show hands again for next sign)",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

    # ── Top-5 bars ────────────────────────────────────────────────────────────
    if results and state in ('RESULT', 'PREDICTING'):
        py = 95
        cv2.putText(frame, "Top 5:", (10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        for rank, (label, conf) in enumerate(results):
            y      = py + 18 + rank * 26
            bar_w  = int(conf * 1.8)
            color  = (0, 255, 100) if rank == 0 else (70, 140, 255)
            cv2.rectangle(frame, (10, y - 13), (10 + 180, y + 3),
                          (45, 45, 45), -1)
            if bar_w > 0:
                cv2.rectangle(frame, (10, y - 13), (10 + bar_w, y + 3),
                              color, -1)
            short = label[:26] + ".." if len(label) > 26 else label
            cv2.putText(frame, f"{short}  {conf:.1f}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (255, 255, 255), 1)

    # ── Bottom controls bar ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 22), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "SPACE = force predict   Q = quit",
                (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (130, 130, 130), 1)

    # ── Frame counter ─────────────────────────────────────────────────────────
    cv2.putText(frame, f"Frame {total_frames}",
                (w - 90, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (80, 80, 80), 1)

    return frame


def main():
    print("=" * 55)
    print("ISL SIGN RECOGNITION — AUTO CAPTURE MODE")
    print("=" * 55)
    print("How to use:")
    print("  1. Position yourself so your upper body is visible")
    print("  2. Raise your hands — recording starts automatically")
    print("  3. Perform the ISL sign")
    print("  4. Lower your hands — prediction runs automatically")
    print("  5. Result shown on screen + printed in terminal")
    print("  SPACE = force predict anytime  |  Q = quit")
    print("=" * 55)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "❌ Cannot open webcam"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("✅ Webcam open\n")

    # ── State machine ─────────────────────────────────────────────────────────
    # WAITING    → hands not detected → do nothing
    # SIGNING    → hands detected     → record frames
    # PREDICTING → hands gone         → run model
    # RESULT     → show result        → wait for next sign

    state          = 'WAITING'
    sign_buffer    = []        # frames captured during current sign
    results        = []        # last prediction results
    no_hand_count  = 0         # consecutive frames without hands
    result_hold    = 0         # frames to hold result on screen
    total_frames   = 0

    # Thresholds
    NO_HAND_TRIGGER = 12       # frames without hands before triggering prediction
    MIN_SIGN_FRAMES = 15       # minimum frames needed to make a valid prediction
    RESULT_HOLD_FRAMES = 90    # hold result on screen for ~6 seconds

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        total_frames += 1

        # Extract landmarks from current frame
        lm, hand_detected = extract_landmarks(frame.copy())

        # ── State transitions ─────────────────────────────────────────────────
        if state == 'WAITING':
            if hand_detected:
                state         = 'SIGNING'
                sign_buffer   = [lm]
                no_hand_count = 0
                print("🤟 Signing detected — recording...")

        elif state == 'SIGNING':
            sign_buffer.append(lm)

            if not hand_detected:
                no_hand_count += 1
                if no_hand_count >= NO_HAND_TRIGGER:
                    # Hands gone long enough — trigger prediction
                    if len(sign_buffer) >= MIN_SIGN_FRAMES:
                        state   = 'PREDICTING'
                        results = predict(sign_buffer)
                        top_label, top_conf = results[0]
                        print(f"\n{'='*50}")
                        print(f"🤟 PREDICTION  ({len(sign_buffer)} frames captured)")
                        print(f"{'='*50}")
                        for rank, (label, conf) in enumerate(results):
                            bar    = "█" * int(conf / 4)
                            marker = " ← TOP" if rank == 0 else ""
                            print(f"  {rank+1}. {label:<35} {conf:5.1f}%  {bar}{marker}")
                        print(f"{'='*50}\n")
                        state         = 'RESULT'
                        result_hold   = RESULT_HOLD_FRAMES
                        sign_buffer   = []
                        no_hand_count = 0
                    else:
                        # Too few frames — was probably just hand passing through
                        state         = 'WAITING'
                        sign_buffer   = []
                        no_hand_count = 0
            else:
                no_hand_count = 0   # reset if hand reappears

            # Safety cap — if signing for too long, predict and reset
            if len(sign_buffer) >= SEQ * 3:
                print(f"⚠ Sign too long — auto-predicting at {len(sign_buffer)} frames")
                results       = predict(sign_buffer)
                state         = 'RESULT'
                result_hold   = RESULT_HOLD_FRAMES
                sign_buffer   = []
                no_hand_count = 0

        elif state == 'RESULT':
            result_hold -= 1
            if result_hold <= 0:
                state   = 'WAITING'
                results = []
                print("⏳ Ready for next sign — raise your hands")

            # If hands appear again during result display → start new sign
            if hand_detected and result_hold < RESULT_HOLD_FRAMES - 20:
                state         = 'SIGNING'
                sign_buffer   = [lm]
                no_hand_count = 0

        # ── Draw UI ───────────────────────────────────────────────────────────
        frame = draw_ui(frame, state, results,
                        len(sign_buffer), total_frames)
        cv2.imshow("ISL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 Quitting...")
            break

        elif key == ord(' '):
            # Force predict on whatever is in buffer right now
            buf = sign_buffer if sign_buffer else []
            if len(buf) >= MIN_SIGN_FRAMES:
                results = predict(buf)
                state   = 'RESULT'
                result_hold = RESULT_HOLD_FRAMES
                top_label, top_conf = results[0]
                print(f"\n🤟 FORCED PREDICTION: {top_label}  ({top_conf:.1f}%)\n")
            else:
                print("⚠ Not enough frames — raise hands and sign first")

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    pose_detector.close()
    print("✅ Done")


if __name__ == "__main__":
    main()