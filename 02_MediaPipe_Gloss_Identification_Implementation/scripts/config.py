from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    # Project root
    "root": ROOT,

    # Data directories
    "dataraw": ROOT / "data" / "raw",
    "dataproc": ROOT / "data" / "processed",
    "features": ROOT / "features",

    # Model directories
    "checkpoints": ROOT / "models" / "checkpoints",
    "mpmodels": ROOT / "models" / "mediapipe",

    # Output directories
    "outputs": ROOT / "outputs" / "predictions",
    "previews": ROOT / "outputs" / "previews",
    "logs": ROOT / "logs",

    # Gloss / translation directories
    "annotations": ROOT / "annotations",
    "gloss": ROOT / "outputs" / "gloss",
    "translations": ROOT / "outputs" / "translations",

    # Files
    "labelencoder": ROOT / "label_encoder.pkl",
    "progressfile": ROOT / "logs" / "features_progress.json",
    "ckptbest": ROOT / "models" / "checkpoints" / "bestmodel.pt",
    "ckptlast": ROOT / "models" / "checkpoints" / "lastmodel.pt",
    "handmodel": ROOT / "models" / "mediapipe" / "handlandmarker.task",
    "posemodel": ROOT / "models" / "mediapipe" / "poselandmarker.task",
    "gloss_map": ROOT / "annotations" / "gloss_map.csv",

    # Dataset subfolders
    "framessent": ROOT / "data" / "raw" / "ISLCSLRTCorpus" / "ISLCSLRTCorpus" / "FramesSentenceLevel",
    "framesword": ROOT / "data" / "raw" / "ISLCSLRTCorpus" / "ISLCSLRTCorpus" / "FramesWordLevel",
    "corpuscsv": ROOT / "data" / "raw" / "ISLCSLRTCorpus" / "ISLCSLRTCorpus" / "corpuscsvfiles",
}

# Model / training hyperparameters
SEQLEN = 50
FEATDIM = 225
HIDDEN = 128
EPOCHS = 300
LR = 5e-4
BATCH = 16
STRIDE = 1
