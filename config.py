"""
config.py
---------
Single place for every path and constant used across the pipeline scripts
(01 through 05). Nothing in this project hardcodes a path a second time
anywhere else - if you move the project folder, this is the only file you
need to touch.

Design note for Ayush: the OLD config.py defined HIDDEN = 128 but the model
that actually got trained used hidden=256 (the value never got passed
through). That mismatch is exactly the kind of silent bug that wastes a
debugging afternoon. This new pipeline has no hidden trained-model
dimension to get wrong in the first place (see 03_... .py for why), so that
whole bug class disappears by construction, not by being more careful.
"""

import os

# ---------------------------------------------------------------------------
# ROOT: this file's own folder. Everything else is relative to this, so the
# whole RAG_Pipeline folder can be moved anywhere on your Mac and still work.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    # Put your raw .mp4 / .mov clips here. One file = one sign/sentence,
    # e.g. data_raw_videos/How_Are_You.mp4, data_raw_videos/Thank_You.mp4
    "data_raw_videos": os.path.join(ROOT, "data_raw_videos"),

    # 01 writes one .pkl per video here (raw per-frame landmark sequence).
    "features_extracted": os.path.join(ROOT, "features_extracted"),

    # 03 writes the embedding vectors here (both the DB-ready file and the
    # human-readable dictionary export).
    "embeddings": os.path.join(ROOT, "embeddings"),

    # 11_train_encoder.py saves the trained sequence encoder here.
    "encoder_dir": os.path.join(ROOT, "encoder"),

    # 04 creates the local DuckDB database file here.
    "vector_db": os.path.join(ROOT, "vector_db"),
    "duckdb_file": os.path.join(ROOT, "vector_db", "sign_vectors.duckdb"),

    # Put the clip(s) you want to TEST retrieval with here (05 reads from
    # this folder) - e.g. you record yourself signing "How Are You" again
    # and drop it in here to see if it gets matched back correctly.
    "test_query_videos": os.path.join(ROOT, "test_query_videos"),

    # Re-use the MediaPipe model files you already downloaded for the old
    # project (via its 02_download_models.py). Point these at wherever
    # those two .task files already live on your machine. If you don't
    # have them anymore, 01_...py will print the exact download URLs.
    "hand_model": os.path.join(ROOT, "mediapipe_models", "hand_landmarker.task"),
    "pose_model": os.path.join(ROOT, "mediapipe_models", "pose_landmarker.task"),

    # 07 (and the out-of-vocabulary fallback in 06) copy every newly
    # enrolled clip here, so a DB row's source_video always points at a
    # stable file even if the original test clip is deleted later.
    "enrolled_videos": os.path.join(ROOT, "enrolled_videos"),

    # YOUR OWN recordings of the demo signs, kept SEPARATE from data_raw_videos (which is untouched).
    # Layout: data_raw_videos2/<exact sign label>/take1.mp4, take2.mp4 ...
    # 13_eval_own_clips.py tests these against the DB built from data_raw_videos. They are never added to it.
    "data_raw_videos2": os.path.join(ROOT, "data_raw_videos2"),
    "features_extracted2": os.path.join(ROOT, "features_extracted2"),

    # Every LLM/VLM call (prompt metadata + raw response) is appended to
    # llm_calls.jsonl here, so experiments are auditable/reproducible.
    "llm_logs": os.path.join(ROOT, "llm_logs"),
}

# ---------------------------------------------------------------------------
# Landmark feature layout - IDENTICAL convention to the old pipeline so
# anything you learned about the 225-dim vector still applies:
#   right hand : 21 landmarks x (x, y, z) = 63
#   left hand  : 21 landmarks x (x, y, z) = 63
#   pose       : 33 landmarks x (x, y, z) = 99
#   TOTAL per frame = 225
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
DIMS_PER_LANDMARK = 3

RIGHT_HAND_DIM = NUM_HAND_LANDMARKS * DIMS_PER_LANDMARK   # 63
LEFT_HAND_DIM = NUM_HAND_LANDMARKS * DIMS_PER_LANDMARK    # 63
POSE_DIM = NUM_POSE_LANDMARKS * DIMS_PER_LANDMARK         # 99
FRAME_FEATURE_DIM = RIGHT_HAND_DIM + LEFT_HAND_DIM + POSE_DIM  # 225

# MediaPipe detection/tracking confidence thresholds.
# Old pipeline used 0.3 for everything (quite permissive -> more false
# detections, which is one likely source of the accuracy problems you saw).
# 0.5 is MediaPipe's own documented default and a safer starting point.
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_HAND_TRACKING_CONFIDENCE = 0.5

MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_POSE_TRACKING_CONFIDENCE = 0.5

MAX_NUM_HANDS = 2

# Cap frame width before running MediaPipe. The old pipeline forced every
# frame down to 320x240, which throws away a lot of detail that matters for
# subtle handshapes. We only downscale if a frame is LARGER than this, just
# to keep per-video runtime sane on a laptop. 960px keeps far more detail
# than 320px while still running comfortably on an M2.
MAX_FRAME_WIDTH = 960

# Embedding vector size produced by 03 (see that file for the full
# explanation): 4 statistics (mean position, std position, mean velocity,
# std velocity) x 225 landmark dims = 900.
EMBEDDING_DIM_V1 = FRAME_FEATURE_DIM * 4  # 900

# Embedding version. 2 = body-relative + hand-shape features, block-balanced
# (fixes "one label wins everything" - see embedding_utils.py). 1 = the
# original raw-coordinate version. CHANGING THIS MEANS RE-EMBEDDING:
#   python3 03_...py --force && python3 04_...py --rebuild
EMBEDDING_VERSION = 4          # 1 raw | 2 body-relative stats | 3 stats + trajectory | 4 TRAINED encoder (11_train_encoder.py)
HAND_INVARIANT = False         # v2 only: swap/mirror so the more active hand is always 'right' (v3 always does)
TEMPORAL_K = 12                # v3: number of time points in the trajectory
V2_PER_FRAME_DIM = 126 + 126 + 69   # hand pos (R,L) + hand shape (R,L) + upper-body pose (23 pts)
if EMBEDDING_VERSION == 1:
    EMBEDDING_DIM = EMBEDDING_DIM_V1
elif EMBEDDING_VERSION == 2:
    EMBEDDING_DIM = V2_PER_FRAME_DIM * 4
elif EMBEDDING_VERSION == 4:
    EMBEDDING_DIM = 64
else:
    EMBEDDING_DIM = V2_PER_FRAME_DIM * 4 + TEMPORAL_K * V2_PER_FRAME_DIM

# Subtract the mean of the stored vectors from query and stored vectors
# before cosine (hubness fix). Computed on the fly from the DB, so there are
# no extra files and incremental adds just work.
# Hand-flip handling (EMBEDDING_VERSION 4). A signer who signs with the other dominant hand is a MIRROR image.
MIRROR_AUGMENT = True          # training: show the encoder every clip in both hand orientations
QUERY_BOTH_ORIENTATIONS = True # retrieval: embed the query as signed AND mirrored, keep the better match

RETRIEVAL_CENTERING = False   # flip to True if 08_diagnose_retrieval.py says "+ centering" wins on your data


# ---------------------------------------------------------------------------
# Context layer (steps 06-07): LLM re-ranking, out-of-vocabulary detection,
# VLM fallback. Everything tunable lives here so experiments are one-line
# changes. See context_utils.py for what each knob does mathematically.
# ---------------------------------------------------------------------------

# Read from the environment so you never hardcode it. The API key itself is
# read from ANTHROPIC_API_KEY (never put it in this file or commit it).
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5-5")
LLM_MAX_TOKENS = 700

# How many retrieval candidates the LLM is asked to judge.
RERANK_TOP_K = 5

# Retrieval likelihood = softmax(cosine_similarity / RETRIEVAL_TEMPERATURE).
# Lower = retrieval is trusted more sharply.
RETRIEVAL_TEMPERATURE = 0.03

# Posterior ~ retrieval_likelihood * (plausibility + CONTEXT_EPSILON) ** CONTEXT_LAMBDA.
# EPSILON keeps an "implausible" (plausibility=0) candidate from being
# vetoed outright: it can still win if retrieval is overwhelmingly sure.
# Consequence (printed at runtime): context can only override retrieval
# when the top-1 vs rival similarity gap is below
#   RETRIEVAL_TEMPERATURE * CONTEXT_LAMBDA * ln((1+EPSILON)/EPSILON)  (~0.09 here).
CONTEXT_EPSILON = 0.05
CONTEXT_LAMBDA = 1.0

# If even the MOST plausible of the top-k candidates scores below this, the
# decision is "no candidate fits the question" -> out-of-vocabulary fallback.
NONE_FITS_MAX_PLAUSIBILITY = 0.25

# Optional similarity floor for out-of-vocabulary detection: if the best
# cosine similarity is below this, the clip is treated as "not in the DB".
# None = disabled. It has to be CALIBRATED on your own data (the in-vocab vs
# unseen similarity distributions), so it ships disabled rather than with a
# made-up number.
OOV_SIM_FLOOR = None

# VLM fallback: how many frames to sample from the clip, and their max side.
VLM_NUM_FRAMES = 8
VLM_MAX_SIDE_PX = 768


# Keys in PATHS that point at a FILE (not a folder) - ensure_dirs() must
# create their PARENT folder, never the path itself. Every other key in
# PATHS is a folder.
_FILE_PATH_KEYS = {"duckdb_file", "hand_model", "pose_model"}


def ensure_dirs():
    """Create every output folder in PATHS if it doesn't exist yet."""
    for key, path in PATHS.items():
        if key in _FILE_PATH_KEYS:
            # it's a file path, make sure its parent folder exists instead
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
