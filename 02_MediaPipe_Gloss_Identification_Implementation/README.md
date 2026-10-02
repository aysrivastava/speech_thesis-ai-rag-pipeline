# MediaPipe + Gloss Identification Implementation (MTP-2)

Real-time Indian Sign Language (ISL) sentence recognition with an added **gloss and grammar layer**.

This folder builds on the original MediaPipe + BiLSTM implementation in [`01_MediaPipe_Paper_Implementation`](../01_MediaPipe_Paper_Implementation). The BiLSTM recognizer is unchanged. MTP-2 adds what happens **after** a sentence label is predicted: the label is mapped to an ISL gloss, the gloss is normalized using ISL grammar rules, and a baseline English sentence is generated. All of this is shown live in a webcam demo.

> **Scope note.** The BiLSTM does **not** recognize gloss tokens from video. It predicts one of 97 trained sentence labels. That label is then looked up in a gloss table (`annotations/gloss_map.csv`) and processed by rule-based grammar and translation modules. The system is a **closed-set recognizer**, not an open-vocabulary translator.

---

## Table of Contents

1. [What MTP-2 Adds](#what-mtp-2-adds)
2. [Pipeline Overview](#pipeline-overview)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [How to Run](#how-to-run)
6. [Component Details](#component-details)
7. [Gloss Map Format](#gloss-map-format)
8. [Grammar Normalization Rules](#grammar-normalization-rules)
9. [Live Demo Output](#live-demo-output)
10. [Output Files](#output-files)
11. [Known Limitations](#known-limitations)
12. [Future Work](#future-work)
13. [References](#references)

---

## What MTP-2 Adds

| Layer | MTP-1 (folder 01) | MTP-2 (this folder) |
|---|---|---|
| Landmark extraction | MediaPipe hands + pose → 225 features/frame | Same |
| Recognition | 2-layer BiLSTM → top-5 sentence labels | Same |
| Gloss identification | — | Predicted label → ISL gloss via `gloss_map.csv` |
| Grammar features | — | Question flag, WH type, polarity, tense, aspect |
| Grammar normalization | — | Temporal fronting, WH inference, clause-final NEG / COMPL / G-WH, predicate-final reordering |
| English output | Label text only | Rule-based English sentence + LLM-ready prompt JSON |
| Live demo | Top-5 predictions | Top-5 + gloss + grammar markers + English translation |

---

## Pipeline Overview

```text
Signer performs an ISL sentence
        │
        ▼
Webcam frame (resized to 320×240)
        │
        ▼
MediaPipe Hand Landmarker + Pose Landmarker
  Right hand : 21 × 3 = 63
  Left hand  : 21 × 3 = 63
  Pose       : 33 × 3 = 99
        │
        ▼
225-dim feature vector per frame
        │
        ▼
50-frame sequence (zero-padded / truncated)
        │
        ▼
2-layer BiLSTM classifier ──► Top-5 sentence labels + confidence
        │
        ▼  (top-1 label)
06_label_to_gloss.py      ──► raw gloss + grammar metadata  (gloss_map.csv lookup)
        │
        ▼
07_normalize_gloss.py     ──► normalized gloss + applied rules + NEG / WH / COMPL
        │
        ▼
08_translate_gloss.py     ──► English sentence + LLM prompt JSON
        │
        ▼
09_realtime_pipeline_demo.py  ──► on-screen + terminal output, JSON saved to outputs/
```

**Example**

```text
Predicted label   : where are you from
Raw gloss         : YOU FROM PLACE G-WH
Normalized gloss  : YOU FROM PLACE G-WH
Grammar           : NEG=OFF | WH=ON (where) | COMPL=OFF
Polarity / Tense  : positive / present
English           : Where are you from?
```

---

## Project Structure

```text
02_MediaPipe_Gloss_Identification_Implementation/
├── README.md
├── annotations/
│   └── gloss_map.csv              # Label → gloss + grammar annotation table
├── scripts/
│   ├── config.py                  # Paths and hyperparameters
│   ├── 01_download.py             # Download ISL-CSLTR corpus (MTP-1)
│   ├── 02_download_models.py      # Download MediaPipe .task models (MTP-1)
│   ├── 03_extract.py              # Extract hand + pose landmarks (MTP-1)
│   ├── 04_train.py                # Train BiLSTM classifier (MTP-1)
│   ├── 05_webcam_demo.py          # Recognition-only live demo (MTP-1)
│   ├── 06_label_to_gloss.py       # NEW: label → gloss lookup
│   ├── 07_normalize_gloss.py      # NEW: grammar-aware gloss normalization
│   ├── 08_translate_gloss.py      # NEW: rule-based English + LLM prompt schema
│   └── 09_realtime_pipeline_demo.py  # NEW: full live pipeline demo
└── outputs/                       # Generated at runtime (git-ignored)
    ├── gloss/
    ├── translations/
    └── predictions/
```

Datasets, extracted features, MediaPipe `.task` files, checkpoints (`*.pt`) and `label_encoder.pkl` are **not committed**. Regenerate them with scripts `01`–`04`.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install mediapipe opencv-python pandas numpy tqdm scikit-learn kagglehub matplotlib
```

Tested on macOS (Apple Silicon). The demo picks the device automatically: `mps` → `cuda` → `cpu`.

---

## How to Run

### 1. Build the recognizer (same as MTP-1)

```bash
python3 scripts/01_download.py          # ISL-CSLTR corpus
python3 scripts/02_download_models.py   # hand_landmarker.task, pose_landmarker.task
python3 scripts/03_extract.py           # landmark features
python3 scripts/04_train.py             # BiLSTM → models/checkpoints/
```

### 2. Run the gloss pipeline offline (optional)

Each module can run on its own to test a single label:

```bash
python3 scripts/06_label_to_gloss.py
python3 scripts/07_normalize_gloss.py
python3 scripts/08_translate_gloss.py
```

### 3. Run the full live demo

```bash
python3 scripts/09_realtime_pipeline_demo.py
```

| Action | Result |
|---|---|
| Raise hands into frame | Recording starts (`SIGNING`) |
| Lower hands for 12 frames | Prediction runs (minimum 15 captured frames) |
| `SPACE` | Force a prediction on the current buffer |
| `Q` | Quit |

---

## Component Details

### Recognizer (`04_train.py`, `09_realtime_pipeline_demo.py`)

| Parameter | Value |
|---|---|
| Input | 50 frames × 225 features |
| Model | 2-layer bidirectional LSTM, dropout 0.3 |
| Hidden size | 256 (read from the checkpoint at inference time) |
| Head | Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→num_classes) |
| Classes | 97 sentence labels (from `label_encoder.pkl`) |
| Output | Top-5 labels with softmax confidence |
| Loss / Optimizer | Cross-entropy / AdamW (weight decay 1e-4), LR 5e-4, batch 16 |
| Scheduler | Cosine annealing, T_max = 300 |

> `config.py` sets `HIDDEN = 128`, but `04_train.py` does not pass it to the model, so training uses the class default of 256. The demo reads the real hidden size from `lstm.weight_ih_l0` in the checkpoint, so loading always works.

### `06_label_to_gloss.py` — Gloss Identification

- Loads `annotations/gloss_map.csv` into a dictionary keyed by `label`.
- `label_to_gloss(label)` returns the gloss and grammar metadata for the predicted label.
- **Fallback:** if the label is not in the CSV, it returns an empty gloss with all grammar fields set to `unknown` instead of guessing.
- `save_output()` writes the record as JSON to `outputs/gloss/`.

### `07_normalize_gloss.py` — Grammar Normalization

- `normalize_gloss_record()` tokenizes the raw gloss and applies the rules listed in [Grammar Normalization Rules](#grammar-normalization-rules).
- Produces `tokens_normalized`, `normalized_gloss`, `wh_type`, and an `applied_rules` list showing which rules fired.
- `save_normalized_json()` writes to `outputs/gloss/`.

### `08_translate_gloss.py` — English Generation

- `build_translation_record()`:
  - **Label present:** returns the label in sentence case, ending with `?` or `.` based on `is_question`.
  - **No label:** maps gloss tokens to English through lookup tables (pronouns, time words, ~30 content words) and WH-specific templates.
- `build_llm_prompt()` builds a structured JSON prompt with the label, raw and normalized gloss, grammar features, applied rules, and the baseline translation. It is ready for a future LLM step. **No LLM is called yet.**
- `save_translation_json()` writes to `outputs/translations/`.

### `09_realtime_pipeline_demo.py` — Integrated Live Demo

- Webcam capture with a `WAITING → SIGNING → PREDICTING → RESULT` state machine.
- `predict()` → top-5 labels; `run_text_pipeline(top_label)` chains `06 → 07 → 08`.
- Draws prediction, top-5, grammar markers, raw/normalized gloss, rules and English on screen. Also prints a full report to the terminal.
- Results stay on screen for 120 frames before the demo resets.

---

## Gloss Map Format

`annotations/gloss_map.csv`

| Column | Meaning | Example |
|---|---|---|
| `label` | Exact BiLSTM class name (must match `label_encoder.pkl`) | `where are you from` |
| `gloss` | ISL gloss in uppercase tokens | `YOU FROM PLACE G-WH` |
| `is_question` | `yes` / `no` | `yes` |
| `wh_type` | `who`, `where`, `when`, `what`, `how_many`, `yesno`, `none` | `where` |
| `polarity` | `positive` / `negative` | `positive` |
| `tense` | `past` / `present` / `future` | `present` |
| `aspect` | `none` / `completive` | `none` |
| `notes` | Free text | `starter mapping` |

**Current starter mapping (subset of the 97 labels)**

```csv
label,gloss,is_question,wh_type,polarity,tense,aspect,notes
are you free today,YOU TODAY FREE G-WH,yes,yesno,positive,present,none,starter mapping
help me,ME HELP,no,none,positive,present,none,starter mapping
where are you from,YOU FROM PLACE G-WH,yes,where,positive,present,none,starter mapping
who are you,YOU WHO G-WH,yes,who,positive,present,none,starter mapping
i need water,ME WATER NEED,no,none,positive,present,none,starter mapping
```

Lookup is an exact match on `label` (lowercase, same spelling and punctuation as the label encoder). Labels missing from the CSV are still predicted, but their grammar fields show as `unknown`.

---

## Grammar Normalization Rules

The rules draw on published Indian / Indo-Pakistani Sign Language grammar (Zeshan; Aboh, Pfau & Zeshan). They are applied in this order:

| # | Rule | Mechanism |
|---|---|---|
| 1 | Temporal fronting | Tokens from a 9-word time vocabulary (`YESTERDAY, TODAY, TOMORROW, BEFORE, LATER, NOW, PAST, FUTURE, TIME`) move to the front |
| 2 | WH-type inference | Uses `wh_type` from the CSV if set; otherwise uses associate signs `FACE→who`, `PLACE→where`, `TIME→when`, `NUMBER→how_many`; a bare `G-WH` gives `unknown` |
| 3 | Clause-final particles | `NEG`, `COMPL`, `G-WH` are taken from the tokens and/or inferred from polarity / aspect / is_question, then always re-emitted in the order **NEG → COMPL → G-WH** |
| 4 | Predicate-final reordering | For sequences longer than 2 tokens: first pronoun = subject, verb-like stem (30-stem list) = predicate, rebuilt as subject + object + residual + predicate |

Final order: **temporal tokens → core content → clause-final particles**.

These rules are heuristics. They have not been evaluated against a manually glossed test set and are not a complete ISL grammar parser.

---

## Live Demo Output

**On screen**

```text
State: RESULT
Captured Frames: 0 | Total Frames: <frame count>
Prediction: where are you from
Confidence: <top-1 confidence>%
TOP 5 PREDICTIONS
1. where are you from (<confidence>%)
2-5. <next four labels with confidence>
GRAMMAR: NEG=OFF | WH=ON (where) | COMPL=OFF
Raw Gloss: YOU FROM PLACE G-WH
Normalized: YOU FROM PLACE G-WH
Polarity: positive | Tense: present
Rules: <rules fired by 07_normalize_gloss.py>
English: Where are you from?
```

**Marker logic**

| Marker | ON when |
|---|---|
| `NEG` | `NEG` in normalized tokens **or** polarity = `negative` |
| `WH` | `G-WH` in normalized tokens **or** is_question = `yes` |
| `COMPL` | `COMPL` in normalized tokens **or** aspect = `completive` |

---

## Output Files

Every prediction saves three JSON files:

| File | Location | Contents |
|---|---|---|
| Gloss record | `outputs/gloss/` | label, raw gloss, grammar metadata |
| Normalized record | `outputs/gloss/` | normalized tokens/gloss, WH type, applied rules |
| Translation record | `outputs/translations/` | English sentence + LLM prompt JSON |

---

## Known Limitations

1. **Closed-set recognition.** The model can only output one of its 97 trained labels. A sentence outside that set is mapped to the closest known label, sometimes with high confidence.
2. **Grammar depends on the predicted label.** The gloss layer runs after classification, so it cannot correct a wrong prediction.
3. **Partial gloss coverage.** Only a starter subset of the 97 labels is annotated in `gloss_map.csv`. Unmapped labels return `unknown` grammar fields.
4. **No context awareness.** Each sign is handled on its own, with no dialogue history. Example: if the question is "What is your favourite colour?", the system may still output an unrelated trained sentence such as "I am fine".
5. **Accuracy not yet reported.** In `04_train.py`, augmentation (Gaussian noise, ±3-frame temporal shift, 0.92–1.08 scale) runs **before** the 80/20 stratified split. Versions of the same video can therefore end up in both the training and validation sets, which inflates validation accuracy. A validated figure will only be reported after re-splitting at the original-clip level.
6. **Template-based English.** Negation inserts "not" from a small auxiliary table. Completive aspect appends "completely". Both are rough approximations.

---

## Future Work

- Re-split the data at the original-clip level (separate training and test videos, augmentation on the training set only) and report validated accuracy.
- Extend `gloss_map.csv` to cover all 97 labels.
- Use BiLSTM hidden-state embeddings for top-k retrieval of similar ISL clips. This becomes the starting point for [`03_RAG_Pipeline`](../03_RAG_Pipeline).
- Add a dialogue-context check that flags predictions that don't fit the previous question.
- Send the existing LLM prompt schema to an LLM, grounded in the retrieved evidence.

---

## References

1. Elakkiya, R. & Natarajan, B. — *ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition*, Mendeley Data, 2021. https://data.mendeley.com/datasets/kcmpdxky7p/1
2. Google — *MediaPipe Hand Landmarker*. https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
3. Google — *MediaPipe Pose Landmarker*. https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
4. Zeshan, U. — *Indo-Pakistani Sign Language Grammar: A Typological Outline*, Sign Language Studies, 2003.
5. Aboh, E., Pfau, R. & Zeshan, U. — *When a Wh-Word Is Not a Wh-Word: The Case of Indian Sign Language*, 2005.
6. Napoli, D. J. & Sutton-Spence, R. — *Order of the Major Constituents in Sign Languages: Implications for All Language*, Frontiers in Psychology, 2014.

---

**Author:** Ayush Srivastava — M.Tech Thesis (MTP-2)
