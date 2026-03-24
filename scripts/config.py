# config.py
# ── Single source of truth for all paths, device, and hyperparameters ─────────
# Every script imports this file. Never hardcode paths anywhere else.
# Usage in any script: from config import PATHS, DEVICE, SEQ_LEN, FEAT_DIM

from pathlib import Path
import torch

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path("/Users/ayushsrivastav/Downloads/MTECH_Docs/MTECH - 2nd Year/Thesis/ISL_SLT_Project")

# ── All paths ─────────────────────────────────────────────────────────────────
PATHS = {
    # Directories
    'root'        : ROOT,
    'data_raw'    : ROOT / "data/raw",
    'data_proc'   : ROOT / "data/processed",
    'features'    : ROOT / "features",
    'checkpoints' : ROOT / "models/checkpoints",
    'mp_models'   : ROOT / "models/mediapipe",
    'outputs'     : ROOT / "outputs/predictions",
    'previews'    : ROOT / "outputs/previews",
    'logs'        : ROOT / "logs",

    # Files
    'label_encoder' : ROOT / "label_encoder.pkl",
    'progress_file' : ROOT / "logs/features_progress.json",
    'ckpt_best'     : ROOT / "models/checkpoints/best_model.pt",
    'ckpt_last'     : ROOT / "models/checkpoints/last_model.pt",
    'hand_model'    : ROOT / "models/mediapipe/hand_landmarker.task",
    'pose_model'    : ROOT / "models/mediapipe/pose_landmarker.task",

    # Dataset frame folders (exist after 01_download.py)
    'frames_sent'   : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level",
    'frames_word'   : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level",
    'corpus_csv'    : ROOT / "data/raw/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/corpus_csv_files",
}

# ── Device — auto picks MPS → CUDA → CPU ─────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE      = torch.device("mps")
    DEVICE_NAME = "Apple MPS (Metal GPU)"
elif torch.cuda.is_available():
    DEVICE      = torch.device("cuda")
    DEVICE_NAME = "CUDA GPU"
else:
    DEVICE      = torch.device("cpu")
    DEVICE_NAME = "CPU"

# ── Model hyperparameters ─────────────────────────────────────────────────────
SEQ_LEN  = 50     # number of frames per sample fed to LSTM
FEAT_DIM = 225    # features per frame: 63 (right hand) + 63 (left hand) + 99 (pose)
HIDDEN   = 128    # LSTM hidden layer size
EPOCHS   = 300    # max training epochs (early stop at 100% acc)
LR       = 5e-4   # learning rate for AdamW
BATCH    = 16     # training batch size
STRIDE   = 1      # use EVERY frame — extract maximum info from few samples

# ── Verify config on direct run ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("PROJECT CONFIG VERIFICATION")
    print("=" * 55)
    print(f"  Root   : {ROOT}")
    print(f"  Device : {DEVICE_NAME}")
    print(f"  SeqLen : {SEQ_LEN}  |  FeatDim: {FEAT_DIM}  |  Hidden: {HIDDEN}")
    print(f"  Epochs : {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}")
    print()
    print("  Paths:")
    for k, v in PATHS.items():
        exists = "✅" if Path(v).exists() else "⚪ (not yet)"
        print(f"    {exists}  {k:<15} {v}")
    print("=" * 55)
    print("✅ config.py loaded successfully")