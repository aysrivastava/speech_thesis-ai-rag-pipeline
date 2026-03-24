from pathlib import Path
import torch

ROOT = Path("/Users/ayushsrivastav/Downloads/MTECH_Docs/MTECH - 2nd Year/Thesis/ISL_SLT_Project")

PATHS = {
    'root'        : ROOT,
    'data_raw'    : ROOT / "data/raw",
    'data_proc'   : ROOT / "data/processed",
    'features'    : ROOT / "features",
    'checkpoints' : ROOT / "models/checkpoints",
    'mp_models'   : ROOT / "models/mediapipe",
    'outputs'     : ROOT / "outputs/predictions",
    'previews'    : ROOT / "outputs/previews",
    'logs'        : ROOT / "logs",
    'label_encoder' : ROOT / "label_encoder.pkl",
    'progress_file' : ROOT / "logs/features_progress.json",
    'ckpt_best'     : ROOT / "models/checkpoints/best_model.pt",
    'ckpt_last'     : ROOT / "models/checkpoints/last_model.pt",
    'hand_model'    : ROOT / "models/mediapipe/hand_landmarker.task",
    'pose_model'    : ROOT / "models/mediapipe/pose_landmarker.task",
    'frames_sent'   : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level",
    'frames_word'   : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level",
    'corpus_csv'    : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/corpus_csv_files",
}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); DEVICE_NAME = "Apple MPS (Metal GPU)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda"); DEVICE_NAME = "CUDA GPU"
else:
    DEVICE = torch.device("cpu"); DEVICE_NAME = "CPU"

SEQ_LEN = 50
FEAT_DIM = 225
HIDDEN   = 128
EPOCHS   = 300
LR       = 5e-4
BATCH    = 16
STRIDE   = 1

if __name__ == "__main__":
    print("=" * 55)
    print("PROJECT CONFIG VERIFICATION")
    print("=" * 55)
    print(f"  Root   : {ROOT}")
    print(f"  Device : {DEVICE_NAME}")
    for k, v in PATHS.items():
        exists = "✅" if Path(v).exists() else "⚪ (not yet)"
        print(f"    {exists}  {k:<15} {v}")
    print("=" * 55)
