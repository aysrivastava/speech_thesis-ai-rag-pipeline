import pickle
import importlib.util
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.py"

spec = importlib.util.spec_from_file_location(
    "project_config",
    CONFIG_PATH
)
project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)

PATHS = project_config.PATHS


if hasattr(project_config, "DEVICE"):
    DEVICE = project_config.DEVICE
else:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


SEQ_LEN = getattr(
    project_config,
    "SEQ_LEN",
    getattr(project_config, "SEQLEN", 50)
)

FEAT_DIM = getattr(
    project_config,
    "FEAT_DIM",
    getattr(project_config, "FEATDIM", 225)
)


# ============================================================
# PATH HELPERS
# ============================================================

def get_path(*possible_keys):
    for key in possible_keys:
        if key in PATHS:
            return Path(PATHS[key])

    raise KeyError(
        f"None of these keys were found in PATHS: {possible_keys}"
    )


def load_module(filename, module_name):
    module_path = SCRIPT_DIR / filename

    module_spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


mod06 = load_module(
    "06_label_to_gloss.py",
    "mod06_label_to_gloss"
)

mod07 = load_module(
    "07_normalize_gloss.py",
    "mod07_normalize_gloss"
)

mod08 = load_module(
    "08_translate_gloss.py",
    "mod08_translate_gloss"
)


# ============================================================
# MODEL
# ============================================================

class SignClassifier(nn.Module):
    def __init__(
        self,
        input_dim=225,
        hidden=256,
        num_classes=10
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)

        forward_hidden = h[-2]
        backward_hidden = h[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        return self.fc(combined)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def ensure_runtime_paths():
    root = Path(PATHS["root"])

    PATHS["gloss"] = root / "outputs" / "gloss"
    PATHS["translations"] = root / "outputs" / "translations"

    PATHS["gloss"].mkdir(
        parents=True,
        exist_ok=True
    )

    PATHS["translations"].mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_encoder():
    root = Path(PATHS["root"])

    checkpoint_candidates = [
        root / "models" / "checkpoints" / "best_model.pt",
        root / "models" / "checkpoints" / "bestmodel.pt",
        root / "models" / "checkpoints" / "last_model.pt",
        root / "models" / "checkpoints" / "lastmodel.pt",
    ]

    checkpoint_path = next(
        (
            path
            for path in checkpoint_candidates
            if path.exists()
        ),
        None
    )

    if checkpoint_path is None:
        checkpoint_path = get_path(
            "ckpt_best",
            "best_model",
            "checkpoint_best",
            "ckpt",
            "model_ckpt",
            "ckptbest"
        )

    encoder_candidates = [
        root / "label_encoder.pkl",
        root / "models" / "label_encoder.pkl",
    ]

    encoder_path = next(
        (
            path
            for path in encoder_candidates
            if path.exists()
        ),
        None
    )

    if encoder_path is None:
        encoder_path = get_path(
            "label_encoder",
            "labelencoder",
            "encoder",
            "label_enc"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}"
        )

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using label encoder: {encoder_path}")
    print(f"Using device: {DEVICE}")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE
    )

    with open(encoder_path, "rb") as file:
        label_encoder = pickle.load(file)

    hidden_size = (
        checkpoint["model_state"]["lstm.weight_ih_l0"].shape[0] // 4
    )

    model = SignClassifier(
        input_dim=checkpoint["feat_dim"],
        hidden=hidden_size,
        num_classes=checkpoint["num_classes"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, label_encoder, checkpoint


# ============================================================
# MEDIAPIPE DETECTORS
# ============================================================

def create_detectors():
    root = Path(PATHS["root"])

    hand_model = (
        root
        / "models"
        / "mediapipe"
        / "hand_landmarker.task"
    )

    pose_model = (
        root
        / "models"
        / "mediapipe"
        / "pose_landmarker.task"
    )

    if not hand_model.exists():
        raise FileNotFoundError(
            f"Hand landmark model missing: {hand_model}"
        )

    if not pose_model.exists():
        raise FileNotFoundError(
            f"Pose landmark model missing: {pose_model}"
        )

    print(f"Using hand model: {hand_model}")
    print(f"Using pose model: {pose_model}")

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(hand_model)
            ),
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
    )

    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(pose_model)
            ),
            min_pose_detection_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
    )

    return hand_detector, pose_detector


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(
    image_bgr,
    hand_detector,
    pose_detector
):
    resized = cv2.resize(
        image_bgr,
        (320, 240)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    hand_result = hand_detector.detect(mp_image)

    right_hand = [0.0] * 63
    left_hand = [0.0] * 63

    for index, handedness in enumerate(
        hand_result.handedness
    ):
        coordinates = [
            value
            for landmark in hand_result.hand_landmarks[index]
            for value in (
                landmark.x,
                landmark.y,
                landmark.z
            )
        ]

        side = handedness[0].category_name

        if side == "Right":
            right_hand = coordinates
        else:
            left_hand = coordinates

    pose_result = pose_detector.detect(mp_image)

    if pose_result.pose_landmarks:
        pose = [
            value
            for landmark in pose_result.pose_landmarks[0]
            for value in (
                landmark.x,
                landmark.y,
                landmark.z
            )
        ]
    else:
        pose = [0.0] * 99

    hand_detected = len(
        hand_result.hand_landmarks
    ) > 0

    features = right_hand + left_hand + pose

    return features, hand_detected


# ============================================================
# PREDICTION
# ============================================================

def predict(
    sign_buffer,
    model,
    label_encoder,
    checkpoint
):
    sequence = list(sign_buffer)

    target_length = checkpoint["seq_len"]
    feature_dimension = checkpoint["feat_dim"]

    if len(sequence) < target_length:
        missing_frames = target_length - len(sequence)

        sequence += [
            [0.0] * feature_dimension
            for _ in range(missing_frames)
        ]
    else:
        sequence = sequence[:target_length]

    input_tensor = torch.FloatTensor(
        [sequence]
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        top_k = torch.topk(
            probabilities,
            k=min(
                5,
                checkpoint["num_classes"]
            )
        )

    results = []

    for probability, index in zip(
        top_k.values,
        top_k.indices
    ):
        label = label_encoder.inverse_transform(
            [index.item()]
        )[0]

        confidence = probability.item() * 100

        results.append(
            (
                label,
                confidence
            )
        )

    return results


# ============================================================
# GLOSS, GRAMMAR, TRANSLATION PIPELINE
# ============================================================

def run_text_pipeline(top_label):
    # 1. Label -> raw gloss
    gloss_record = mod06.label_to_gloss(top_label)
    gloss_path = mod06.save_output(gloss_record)

    # 2. Raw gloss -> normalized gloss
    normalized_record = mod07.normalize_gloss_record(
        gloss_record
    )

    normalized_path = mod07.save_normalized_json(
        normalized_record
    )

    # 3. Normalized gloss -> English translation
    translation_record = mod08.build_translation_record(
        normalized_record
    )

    translation_path = mod08.save_translation_json(
        translation_record
    )

    return {
        "label": top_label,
        "gloss_record": gloss_record,
        "normalized_record": normalized_record,
        "translation_record": translation_record,
        "gloss_path": str(gloss_path),
        "normalized_path": str(normalized_path),
        "translation_path": str(translation_path),
    }


# ============================================================
# UI HELPERS
# ============================================================

def shorten_text(text, max_length=72):
    text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def put_lines(
    frame,
    lines,
    x,
    y,
    dy=26,
    color=(255, 255, 255),
    scale=0.58,
    thickness=1
):
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            str(line),
            (x, y + index * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def marker_status(
    normalized_record,
    marker,
    field,
    expected_value=None
):
    tokens = {
        str(token).upper()
        for token in normalized_record.get(
            "tokens_normalized",
            []
        )
    }

    field_value = str(
        normalized_record.get(field, "")
    ).lower()

    marker_present = marker in tokens

    if expected_value is not None:
        marker_present = (
            marker_present
            or field_value == expected_value
        )

    return "ON" if marker_present else "OFF"

def draw_text_box(frame, x, y, w, h, alpha=0.72, color=(255, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_ui(
    frame,
    state,
    results,
    pipeline_result,
    sign_len,
    total_frames
):
    height, width = frame.shape[:2]

    top_panel_height = 115
    bottom_panel_height = 235

    # Solid dark background behind top status text.
    cv2.rectangle(
        frame,
        (0, 0),
        (width, top_panel_height),
        (15, 15, 15),
        -1
    )

    # Solid dark background behind grammar and translation text.
    cv2.rectangle(
        frame,
        (0, height - bottom_panel_height),
        (width, height),
        (15, 15, 15),
        -1
    )

    # Dark translucent card behind Top-5 predictions.
    if results:
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, top_panel_height),
            (440, top_panel_height + 155),
            (10, 10, 10),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.82,
            frame,
            0.18,
            0,
            frame
        )

    state_color = {
        "WAITING": (180, 180, 180),
        "SIGNING": (0, 220, 255),
        "PREDICTING": (0, 255, 255),
        "RESULT": (0, 255, 120),
    }.get(state, (255, 255, 255))

    put_lines(
        frame,
        [
            f"State: {state}",
            f"Captured Frames: {sign_len} | Total Frames: {total_frames}",
        ],
        15,
        35,
        dy=30,
        color=state_color,
        scale=0.72,
        thickness=2
    )

    if results:
        top_label, top_confidence = results[0]

        put_lines(
            frame,
            [
                f"Prediction: {shorten_text(top_label, 50)}",
                f"Confidence: {top_confidence:.1f}%",
            ],
            15,
            94,
            dy=25,
            color=(240, 240, 240),
            scale=0.56,
            thickness=1
        )

        cv2.putText(
            frame,
            "TOP 5 PREDICTIONS",
            (15, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 220, 255),
            1,
            cv2.LINE_AA
        )

        for index, (label, confidence) in enumerate(results):
            short_label = shorten_text(label, 37)

            cv2.putText(
                frame,
                f"{index + 1}. {short_label} ({confidence:.1f}%)",
                (15, 172 + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA
            )

    if pipeline_result:
        normalized = pipeline_result["normalized_record"]
        translation = pipeline_result["translation_record"]

        normalized_tokens = {
            str(token).upper()
            for token in normalized.get("tokens_normalized", [])
        }

        polarity = str(normalized.get("polarity", "")).lower()
        question = str(normalized.get("is_question", "")).lower()
        aspect = str(normalized.get("aspect", "")).lower()

        negation = "ON" if (
            "NEG" in normalized_tokens
            or polarity == "negative"
        ) else "OFF"

        wh_marker = "ON" if (
            "G-WH" in normalized_tokens
            or question in {"yes", "true"}
        ) else "OFF"

        completive = "ON" if (
            "COMPL" in normalized_tokens
            or aspect == "completive"
        ) else "OFF"

        wh_type = normalized.get("wh_type", "none")

        applied_rules = normalized.get("applied_rules", [])
        rules_text = ", ".join(applied_rules) if applied_rules else "none"

        grammar_color = (
            (0, 255, 120)
            if any(value == "ON" for value in [negation, wh_marker, completive])
            else (210, 210, 210)
        )

        lines = [
            f"GRAMMAR: NEG={negation} | WH={wh_marker} ({wh_type}) | COMPL={completive}",
            f"Raw Gloss: {shorten_text(normalized.get('raw_gloss', ''), 78)}",
            f"Normalized: {shorten_text(normalized.get('normalized_gloss', ''), 76)}",
            f"Polarity: {normalized.get('polarity', 'unknown')} | Tense: {normalized.get('tense', 'unknown')}",
            f"Rules: {shorten_text(rules_text, 78)}",
            f"English: {shorten_text(translation.get('english_translation', ''), 82)}",
        ]

        put_lines(
            frame,
            lines,
            15,
            height - 190,
            dy=26,
            color=(240, 240, 240),
            scale=0.50,
            thickness=1
        )

        cv2.putText(
            frame,
            lines[0],
            (15, height - 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            grammar_color,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "SPACE = force predict | 1 = WH test | 2 = COMPL test | 3 = NEG test | Q = quit",
        (15, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Frame {total_frames}",
        (width - 100, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_pipeline_result(
    results,
    pipeline_result,
    forced=False
):
    top_label, top_confidence = results[0]

    title = (
        "FORCED PREDICTION"
        if forced
        else "LIVE PREDICTION"
    )

    normalized = pipeline_result[
        "normalized_record"
    ]

    translation = pipeline_result[
        "translation_record"
    ]

    tokens = {
        str(token).upper()
        for token in normalized.get(
            "tokens_normalized",
            []
        )
    }

    negation = (
        "ON"
        if (
            "NEG" in tokens
            or str(
                normalized.get(
                    "polarity",
                    ""
                )
            ).lower() == "negative"
        )
        else "OFF"
    )

    wh = (
        "ON"
        if (
            "G-WH" in tokens
            or str(
                normalized.get(
                    "is_question",
                    ""
                )
            ).lower() in {"yes", "true"}
        )
        else "OFF"
    )

    completive = (
        "ON"
        if (
            "COMPL" in tokens
            or str(
                normalized.get(
                    "aspect",
                    ""
                )
            ).lower() == "completive"
        )
        else "OFF"
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Predicted Label       : {top_label}")
    print(
        f"Prediction Confidence : "
        f"{top_confidence:.1f}%"
    )
    print(
        f"Raw Gloss             : "
        f"{normalized.get('raw_gloss', '')}"
    )
    print(
        f"Normalized Gloss      : "
        f"{normalized.get('normalized_gloss', '')}"
    )
    print()
    print("GRAMMAR MODULE")
    print(
        f"Negation Marker       : {negation}"
    )
    print(
        f"WH Marker             : {wh}"
    )
    print(
        f"WH Type               : "
        f"{normalized.get('wh_type', 'none')}"
    )
    print(
        f"Completive Marker     : {completive}"
    )
    print(
        f"Polarity              : "
        f"{normalized.get('polarity', 'positive')}"
    )
    print(
        f"Tense                 : "
        f"{normalized.get('tense', 'present')}"
    )
    print(
        f"Aspect                : "
        f"{normalized.get('aspect', 'none')}"
    )
    print(
        f"Applied Rules         : "
        f"{', '.join(normalized.get('applied_rules', [])) or 'none'}"
    )
    print()
    print(
        f"English Translation   : "
        f"{translation.get('english_translation', '')}"
    )
    print()
    print(
        f"Saved Gloss JSON      : "
        f"{pipeline_result['gloss_path']}"
    )
    print(
        f"Saved Normalized JSON : "
        f"{pipeline_result['normalized_path']}"
    )
    print(
        f"Saved Translation JSON: "
        f"{pipeline_result['translation_path']}"
    )
    print("=" * 70)
    print()


# ============================================================
# MAIN LIVE DEMO
# ============================================================

def main():
    ensure_runtime_paths()

    print(
        "PATH KEYS:",
        sorted(PATHS.keys())
    )

    print("=" * 68)
    print("REAL-TIME ISL PIPELINE DEMO")
    print("=" * 68)
    print(
        "1. Position yourself so upper body "
        "and hands are visible"
    )
    print(
        "2. Raise hands to start recording "
        "automatically"
    )
    print(
        "3. Perform one trained sentence"
    )
    print(
        "4. Lower hands to trigger prediction"
    )
    print(
        "5. Grammar output and translation "
        "will appear on screen"
    )
    print(
        "Controls: SPACE = force predict | "
        "Q = quit"
    )
    print("=" * 68)

    model, label_encoder, checkpoint = (
        load_model_and_encoder()
    )

    hand_detector, pose_detector = (
        create_detectors()
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1440
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        900
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        15
    )

    window_name = (
        "ISL Real-Time Pipeline Demo"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        1280,
        1000
    )

    state = "WAITING"
    sign_buffer = []
    results = []
    pipeline_result = None

    no_hand_count = 0
    result_hold = 0
    total_frames = 0

    no_hand_trigger = 12
    minimum_sign_frames = 15
    result_hold_frames = 120

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print(
                    "Unable to read a frame "
                    "from the webcam."
                )
                break

            frame = cv2.flip(frame, 1)
            total_frames += 1

            landmarks, hand_detected = (
                extract_landmarks(
                    frame.copy(),
                    hand_detector,
                    pose_detector
                )
            )

            if state == "WAITING":
                if hand_detected:
                    state = "SIGNING"
                    sign_buffer = [landmarks]
                    no_hand_count = 0
                    pipeline_result = None
                    results = []

                    print(
                        "🤟 Signing detected — "
                        "recording..."
                    )

            elif state == "SIGNING":
                sign_buffer.append(landmarks)

                if not hand_detected:
                    no_hand_count += 1

                    if no_hand_count >= no_hand_trigger:
                        if len(sign_buffer) >= minimum_sign_frames:
                            state = "PREDICTING"

                            results = predict(
                                sign_buffer,
                                model,
                                label_encoder,
                                checkpoint
                            )

                            top_label = results[0][0]

                            pipeline_result = (
                                run_text_pipeline(
                                    top_label
                                )
                            )

                            print_pipeline_result(
                                results,
                                pipeline_result,
                                forced=False
                            )

                            state = "RESULT"
                            result_hold = (
                                result_hold_frames
                            )
                            sign_buffer = []
                            no_hand_count = 0

                        else:
                            state = "WAITING"
                            sign_buffer = []
                            no_hand_count = 0

                else:
                    no_hand_count = 0

                if len(sign_buffer) >= (
                    checkpoint["seq_len"] * 3
                ):
                    results = predict(
                        sign_buffer,
                        model,
                        label_encoder,
                        checkpoint
                    )

                    top_label = results[0][0]

                    pipeline_result = (
                        run_text_pipeline(
                            top_label
                        )
                    )

                    print_pipeline_result(
                        results,
                        pipeline_result,
                        forced=False
                    )

                    state = "RESULT"
                    result_hold = result_hold_frames
                    sign_buffer = []
                    no_hand_count = 0

            elif state == "RESULT":
                result_hold -= 1

                if result_hold <= 0:
                    state = "WAITING"
                    results = []
                    pipeline_result = None

                    print(
                        "⏳ Ready for next sign"
                    )

                if (
                    hand_detected
                    and result_hold
                    < result_hold_frames - 20
                ):
                    state = "SIGNING"
                    sign_buffer = [landmarks]
                    no_hand_count = 0
                    results = []
                    pipeline_result = None

                    print(
                        "🤟 New signing detected — "
                        "recording..."
                    )

            frame = draw_ui(
                frame=frame,
                state=state,
                results=results,
                pipeline_result=pipeline_result,
                sign_len=len(sign_buffer),
                total_frames=total_frames
            )

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(
                    "👋 Quitting demo..."
                )
                break

            if key == ord(" "):
                if len(sign_buffer) >= minimum_sign_frames:
                    results = predict(
                        sign_buffer,
                        model,
                        label_encoder,
                        checkpoint
                    )

                    top_label = results[0][0]

                    pipeline_result = (
                        run_text_pipeline(
                            top_label
                        )
                    )

                    print_pipeline_result(
                        results,
                        pipeline_result,
                        forced=True
                    )

                    state = "RESULT"
                    result_hold = result_hold_frames
                    sign_buffer = []
                    no_hand_count = 0

                else:
                    print(
                        "⚠ Not enough frames captured "
                        "yet for prediction"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()

        hand_detector.close()
        pose_detector.close()

        print("✅ Demo finished")


if __name__ == "__main__":
    main()