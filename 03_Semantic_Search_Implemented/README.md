# MTP-2 Master README: Replicating the Thesis Results

**Context Aware SLT System: Vector-Retrieval Recognition, Gloss-Aware Translation and Open-Set Contextual Answering for Indian Sign Language**
M.Tech Thesis (MTP-2), Indian Institute of Technology Jodhpur
Ayush Srivastava (M25DE2031) · Advisor: Prof. Sidharth Ranjan · November 2026

This is the single entry point for the whole project. Follow the steps in order to reproduce every result in the thesis report. Each step lists the exact commands, how long it takes, and what it produces.

The thesis compares two ways of recognising Indian Sign Language (ISL) signs from video:

* **MTP-1 (closed-set):** MediaPipe landmarks and a BiLSTM classifier that picks one label from a fixed list. This replicates a published paper.
* **MTP-2 (open-set):** the same kind of landmarks go through a small 1D-CNN encoder that produces a 64-number embedding. Embeddings are stored in a DuckDB table and a query is answered by nearest-neighbour search, so new signs can be added without retraining.

---

## Contents

1. [Folder map](#1-folder-map)
2. [Step 1: Requirements](#2-step-1-requirements)
3. [Step 2: Install the libraries](#3-step-2-install-the-libraries)
4. [Step 3: Download the dataset](#4-step-3-download-the-dataset)
5. [Step 4: Check the setup](#5-step-4-check-the-setup)
6. [Step 5 (optional): Re-extract landmarks](#6-step-5-optional-re-extract-the-landmarks)
7. [Step 6: The 20-sign pilot (thesis 4.1 and 4.2)](#7-step-6-the-20-sign-pilot-thesis-41-and-42)
8. [Step 7: The 98-sign studies (thesis 4.3 to 4.7)](#8-step-7-the-98-sign-studies-thesis-43-to-47)
9. [Step 8: The 10-fold benchmark (thesis 4.8)](#9-step-8-the-10-fold-benchmark-thesis-48)
10. [Step 9: Compare with the expected results](#10-step-9-compare-with-the-expected-results)
11. [Optional A: Demo on your own video](#11-optional-a-demo-on-your-own-video)
12. [Optional B: Context-aware prototype (thesis Chapter 5)](#12-optional-b-context-aware-prototype-thesis-chapter-5)
13. [Optional C: Gloss-aware translation layer](#13-optional-c-gloss-aware-translation-layer)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)

---

## 1. Folder map

| Folder | Contents |
|---|---|
| `0. Dataset (ISL-CSLRT videos)/` | The ISL-CSLRT videos (about 7 GB) |
| `1. MediaPipe BiLSTM/` | MTP-1: download, extraction, training and webcam-demo scripts (`mtp1_original/scripts/`) |
| `2. Vector Retrieval/` | MTP-2: landmark extraction, features, encoder training, DuckDB storage and retrieval, MediaPipe model files |
| `3. Comparison Experiment/` | The 20-sign pilot (`run_*.py`, `results/`, `COMPARISON.md`) |
| `3. Comparison Experiment/scale/` | The 98-sign study: scripts `s0` to `s11`, `corpus.json`, `signers.json`, `folds_10fold_S98.json`, saved landmark features and results in `data/` |
| `4. Open Set Context Aware Detection/` | The context-aware prototype (has its own README) |
| `5. Live Professor Demo/` | Scripts for the live side-by-side demo |
| `6. Final Submission/` | A slim copy of the code (`ISL_MTP2_Replication/` and its zip) without the large data folders |
| `Master Docs/` | Two PDF master references: the written record of the experiments and of the prototype |

**All commands in this guide use the folders in this directory.** Run them in a terminal (macOS or Linux; on Windows use WSL).

> **Do you need the videos?** No, not to reproduce the results. The extracted landmark features and the per-fold results are already saved in `3. Comparison Experiment/` and `3. Comparison Experiment/scale/data/`. The videos are needed only for Step 5 (re-extracting landmarks) and for the demos.

---

## 2. Step 1: Requirements

| Item | Requirement |
|---|---|
| Operating system | macOS or Linux (Windows through WSL) |
| Python | 3.9 to 3.12 (3.12 is needed only for the optional Holistic environment in Step 5) |
| Hardware | A CPU is enough; no GPU is required. A webcam is needed only for the live demos |
| Disk space | About 15 GB for this folder with the videos; the code and saved features alone are about 3 GB |
| Time | Pilot: about 1 to 2 hours. 98-sign five-fold studies: several hours. 10-fold benchmark: about 4 to 6 hours. All times are for a typical laptop CPU and every step can be interrupted and resumed |

---

## 3. Step 2: Install the libraries

Open a terminal in this folder and create a virtual environment:

```bash
python3 -m venv env
source env/bin/activate            # Windows (WSL): same command
python -m pip install --upgrade pip
```

Install the libraries used by all recognition experiments:

```bash
pip install numpy scipy scikit-learn torch opencv-python duckdb matplotlib "mediapipe>=0.10.30,<1.0"
```

| Library | Used for |
|---|---|
| `numpy`, `scipy`, `scikit-learn` | Arrays, statistics, stratified splits |
| `torch` | BiLSTM and the 1D-CNN encoder (CPU build is fine) |
| `opencv-python` | Reading video frames |
| `mediapipe` (0.10.30 or later, below 1.0) | Hand and pose landmarks (Tasks API) |
| `duckdb` | The vector database |
| `matplotlib` | Figures only |

> **Why below 1.0?** MediaPipe 1.0.x has a regression that crashes the hand and pose landmarkers on macOS with Apple Silicon. Versions 0.10.30 to 0.10.x are confirmed to work.

For the optional parts, install these only when you reach them:

| Part | Extra libraries |
|---|---|
| Context-aware prototype (Optional B) | `pip install anthropic` |
| Dataset download through Kaggle (Step 3, option B) | `pip install kagglehub` |
| Paper-recipe landmarks (Step 5) | A **separate** environment with Python 3.12 and `mediapipe==0.10.14` |

Check the installation:

```bash
python -c "import numpy, scipy, sklearn, torch, cv2, duckdb, mediapipe; print('mediapipe', mediapipe.__version__, '| torch', torch.__version__)"
```

---

## 4. Step 3: Download the dataset

The experiments use the **ISL-CSLRT** corpus (Elakkiya and Natarajan, 2021): about 700 sentence-level ISL videos in about 100 sentence folders, signed by 7 people. The copy used in the thesis has 687 videos in 101 folders (about 7 GB).

**Option A: Mendeley Data (original source)**

1. Open <https://data.mendeley.com/datasets/kcmpdxky7p/1> (DOI 10.17632/kcmpdxky7p.1).
2. Download the dataset and unzip it.

**Option B: Kaggle mirror (used by the project's own download script)**

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('drblack00/isl-csltr-indian-sign-language-dataset'))"
```

The command prints the folder where the files were saved. If it asks you to sign in, use a free Kaggle account.

**After downloading**, find the folder named `Videos_Sentence_Level` (it contains one sub-folder per sentence, each with the `.mp4` or `.mov` clips):

```bash
find "<download folder>" -type d -name Videos_Sentence_Level
```

Tell the scripts where it is (use the path the command printed):

```bash
export SCALE_VIDEOS="/path/to/Videos_Sentence_Level"
```

**Which clips are used.** The study uses 98 signs and 490 clips: the 1920×1080 landscape studio recordings (files of 3 MB or more), keeping signs that have at least 4 such clips. The list is fixed in `3. Comparison Experiment/scale/corpus.json`, and the signer of each clip is in `scale/signers.json`. The corpus has no signer labels, so signers were recovered by clustering torso colour (evidence in `scale/signer_study/`). Always use the shipped `corpus.json`; the script `s0_build_corpus.py` can rebuild a list from the videos but may differ slightly.

---

## 5. Step 4: Check the setup

Regenerate the result tables from the saved per-fold results. This needs no training and takes seconds:

```bash
cd "3. Comparison Experiment/scale"
python s5_analyze.py S98
python s10_analyze_new.py
cd ../..
```

The tables appear in `3. Comparison Experiment/scale/data/results/` (`RESULTS_S98.md`, `NEW_RESULTS.md`). Compare them with the thesis tables. If this works, the environment is ready.

---

## 6. Step 5 (optional): Re-extract the landmarks

Skip this step to use the shipped features (recommended when you compare numbers, because BiLSTM accuracy changes slightly with the MediaPipe version). It needs the videos (`SCALE_VIDEOS`, Step 3).

```bash
cd "3. Comparison Experiment/scale"
python s1_extract.py rag 0 1       # MTP-2 pipeline: MediaPipe Tasks, video mode, 225 values per frame
python s1_extract.py mtp1 0 1      # MTP-1 pipeline: image mode, 320x240 frames, 225 values per frame
```

The arguments are `<mode> <shard> <number of shards>`. To use more cores, run several shards at the same time, for example `0 4`, `1 4`, `2 4` and `3 4`.

> These commands write into `scale/data/feat_rag/`, `feat_mtp1_image/` and `feat_holistic/`. To keep the shipped features untouched, first set `export SCALE_DATA="/some/new/folder"`.

The paper recipe (variant C) uses MediaPipe Holistic, 1,662 values per frame, which needs the old "solutions" API in its own environment:

```bash
python3.12 -m venv ../../env_holistic
../../env_holistic/bin/pip install "mediapipe==0.10.14" opencv-python-headless "numpy<2"
../../env_holistic/bin/python s1_extract.py holistic 0 1     # run inside the Holistic environment
cd ../..
```

The Holistic features are already saved in `scale/data/feat_holistic/`.

---

## 7. Step 6: The 20-sign pilot (thesis 4.1 and 4.2)

The pilot compares MTP-1 and the first vector retriever on 20 signs (five folds), and audits the 91.1% accuracy of MTP-1.

```bash
cd "3. Comparison Experiment"
python run_vdb.py 150                 # vector retrieval, 5 folds
python run_bilstm.py V0 60            # MTP-1 as built (image mode, first 50 frames)
python run_bilstm.py V1 60            # + video-mode landmarks
python run_bilstm.py V2 60            # + 30 frames sampled across the whole sign
python run_bilstm.py V2b 150          # V2 without augmentation
python run_bilstm.py V3 60            # + MediaPipe Holistic features
python run_bilstm.py V3b 150          # V3 without augmentation
python run_bilstm.py V4               # + the paper's network and training recipe
python run_bilstm.py V5               # + the paper's landmark normalisation
python run_original_protocol.py 60    # MTP-1's own (leaky) evaluation, for contrast
python run_extras.py                  # new-sign and rejection test
python run_timing.py                  # training time and query latency
python make_report.py                 # rebuilds COMPARISON.md from results/
cd ..
```

Output: `3. Comparison Experiment/results/summary.json` and `COMPARISON.md`.

---

## 8. Step 7: The 98-sign studies (thesis 4.3 to 4.7)

Run all commands in this step from `3. Comparison Experiment/scale/`:

```bash
cd "3. Comparison Experiment/scale"
```

Every script saves one result file per fold and **resumes from the last finished fold**, so an interrupted run can simply be started again. The numbers `3` or `4` at the end are the thread count: use your number of CPU cores.

The arguments are `<variant or system> <sign subset> <epochs> <threads>`. Use `S50` instead of `S98` for the 50-sign subset.

**BiLSTM variants:** A is MTP-1 as built (image mode, first 50 frames, with augmentation). B is video-mode landmarks and 30 frames sampled across the whole sign. C is the published paper's recipe (Holistic features, ReLU-LSTM 64-128-64, Adamax, 300 epochs, no augmentation). D and E mix parts of the paper recipe with the project's landmarks.

### 4.3 Main study: 50 and 98 signs, five-fold

```bash
for SUB in S50 S98; do
  python s3_vdb.py $SUB 150 3             # vector retrieval
  python s2_bilstm.py A $SUB 100 3        # MTP-1 as built
  python s2_bilstm.py B $SUB 100 3        # best BiLSTM
  python s2_bilstm.py C $SUB 300 3        # paper recipe (uses data/feat_holistic)
  python s3b_head_control.py $SUB 150 3   # control: encoder as a classifier (thesis 4.7)
  python s5_analyze.py $SUB               # tables -> data/results/RESULTS_$SUB.md
done
```

### 4.4 A signer never seen in training (leave-one-signer-out)

```bash
SCALE_PROTO=loso python s3_vdb.py S98 150 3
SCALE_PROTO=loso python s2_bilstm.py A S98 100 3
SCALE_PROTO=loso python s2_bilstm.py B S98 100 3
SCALE_PROTO=loso python s2_bilstm.py C S98 300 3
```

### 4.5 Training clips per sign (1, 2 and 3)

```bash
for K in 1 2 3; do
  SCALE_K=$K python s3_vdb.py S98 150 3
  SCALE_K=$K python s2_bilstm.py B S98 100 3
done
```

### 4.6 Unseen signs: rejection and enrolment

```bash
python s9_newsign.py vdb 3
python s9_newsign.py B 3
```

### Tables for 4.4 to 4.6

```bash
python s10_analyze_new.py             # -> data/results/NEW_RESULTS.md and NEW_RESULTS.json
```

### 4.7 Control experiment

Already run in the loop for Section 4.3 (`s3b_head_control.py`). Output: `data/results/summary_CTRL.json`.

### Environment variables used by the scripts

| Variable | Meaning |
|---|---|
| `SCALE_VIDEOS` | Folder `Videos_Sentence_Level` (Step 3) |
| `SCALE_DATA` | Data folder (default: `scale/data`) |
| `SCALE_PIPE` | Location of `2. Vector Retrieval/` (default: found automatically) |
| `SCALE_PROTO=loso` | Leave-one-signer-out instead of five-fold |
| `SCALE_K=1\|2\|3` | Number of training clips per sign |
| `SCALE_FOLDS_FILE` | Use a fixed fold file for any system (used for the 10-fold benchmark) |
| `SCALE_EPC` | Epochs of the paper-recipe variants (default: 300) |

---

## 9. Step 8: The 10-fold benchmark (thesis 4.8)

This is the final comparison. Both systems use the same ten folds, stored in `folds_10fold_S98.json`. The file was built with seed 2026: for each sign the clips are shuffled and dealt round-robin to the folds, so the clips of a sign always fall into different folds. Fold sizes are 49 clips (one fold has 48).

```bash
cd "3. Comparison Experiment/scale"
mkdir -p data/logs

# 1) Vector retrieval, 10 folds (about 3 to 6 minutes per fold)
python s11_kfold_vdb.py 10 150 4

# 2) BiLSTM baselines on the SAME folds (about 10 to 16 minutes per fold)
export SCALE_FOLDS_FILE="$PWD/folds_10fold_S98.json"
python s2_bilstm.py B S98 100 4        # best BiLSTM
python s2_bilstm.py A S98 100 4        # MTP-1 as built
unset SCALE_FOLDS_FILE
```

`s11_kfold_vdb.py` creates the fold file if it is missing and checks that every clip is tested exactly once and that every test sign has training clips in every fold.

Output in `data/results/`: `VDB_S98_10FOLD.json`, `RESULTS_S98_10FOLD.md` (vector), `B_S98_10FOLD.json`, `A_S98_10FOLD.json` and `RESULTS_S98_10FOLD_benchmark.md` (BiLSTM). Two vector-retrieval runs are shipped for comparison: `VDB_S98_10FOLD.json` (laptop) and `VDB_S98_10FOLD_cloudrun.json` (cloud container).

---

## 10. Step 9: Compare with the expected results

Differences of 1 to 3 percentage points between machines are normal (training is seeded per fold, but libraries differ).

| Result | Expected |
|---|---|
| 20-sign pilot, MTP-1's own protocol | 97.5% (held-out clips: 22.7%) |
| 98 signs, 5-fold, vector retrieval, top-1 / top-3 | 47.6% / 69.4% |
| 98 signs, 5-fold, MTP-1 as built (A) / best BiLSTM (B), top-1 | 4.5% / 9.0% |
| Paper recipe (C) on 98 signs, top-1 | 3.9% |
| Leave-one-signer-out, vector retrieval, top-1 | 45.4% |
| One clip per sign, vector retrieval / BiLSTM B, top-1 | 13.5% / 2.2% |
| Unseen signs enrolled from one clip, top-1 / top-3 | 15.3% / 42.6% |
| Control: same encoder as a classifier, top-1 | 49.0% |
| 10-fold, vector retrieval, top-1 / top-3 | 48.2 ± 5.5% / 71.1 ± 6.4% (cloud run); 45.6 ± 6.8% / 67.7 ± 6.6% (laptop run) |
| 10-fold, BiLSTM variant B, top-1 / top-3 | 7.6 ± 3.9% / 17.8 ± 6.0% |

On some machines the similarity step prints numpy `RuntimeWarning` messages (divide by zero, overflow, invalid value). If you see them, compare your 10-fold result with both shipped vector files.

---

## 11. Optional A: Demo on your own video

Build the encoder, vector database and BiLSTM models for all 98 signs (about 10 minutes for the vector part), then classify one video with both systems:

```bash
cd "3. Comparison Experiment/scale"
python s4_final_models.py S98 4 vdb        # encoder + vector database
python s4_final_models.py S98 4 bilstm     # BiLSTM A, B and C (run separately from the line above)
python demo_query.py "/path/to/video.mp4" S98
```

The models are saved in `data/saved_models/S98/`.

**Live webcam demo (20 signs).** Build the 20-sign models once, then start the side-by-side demo:

```bash
cd "3. Comparison Experiment/scale"
python s4_final_models.py D20 4 vdb
python s4_final_models.py D20 4 bilstm
cd "../../5. Live Professor Demo"
python professor_demo.py --vocab 20
```

The menu lets you sign live, run a dry run on a reference video, re-run the last take and show a scoreboard. Press SPACE to start a 3-2-1 countdown and a 3.5-second recording. Turn off any camera auto-framing feature of your operating system first.

---

## 12. Optional B: Context-aware prototype (thesis Chapter 5)

The prototype answers a typed question from a few webcam frames with a vision language model. It needs no training and no sign list. It needs an API key; provide it **only through the environment** and never store it in a file.

```bash
cd "4. Open Set Context Aware Detection"
pip install -r requirements.txt                 # opencv-python, numpy, anthropic
read -s -p "API key: " ANTHROPIC_API_KEY; echo; export ANTHROPIC_API_KEY
python open_set_demo.py --dry-run               # prints the prompts; no API call, no key needed
python open_set_demo.py                         # live webcam round (needs the key and camera permission)
python -m pytest tests/test_offline.py          # offline tests (no key, no camera)
```

The ten logged rounds behind thesis Section 5.4 are in `logs/`. The accuracy of this prototype is not measured. See the README inside the folder for details.

---

## 13. Optional C: Gloss-aware translation layer

The scripts are in `6. Final Submission/ISL_MTP2_Replication/5. Gloss Layer/`. Run `06_label_to_gloss.py`, `07_normalize_gloss.py` and `08_translate_gloss.py` in this order over a gloss table `annotations/gloss_map.csv` with the columns `label, gloss, is_question, wh_type, polarity, tense, aspect, notes`. The scripts need the MTP-1 configuration: copy `config_mtp1.py` to `config.py` and set its `ROOT` to your folder. The language-model prompt is built and saved but not sent.

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| MediaPipe crashes on a Mac with Apple Silicon | Use `mediapipe>=0.10.30,<1.0`. Version 1.0.x has a known regression |
| `ModuleNotFoundError` | Activate the environment (`source env/bin/activate`) and repeat Step 2 |
| Scripts cannot find the videos | Set `SCALE_VIDEOS` (Step 3) |
| A run stopped half way | Run the same command again; finished folds are skipped |
| Slightly different accuracies | Normal (1 to 3 points). Use the shipped features and the fixed fold file |
| MTP-1's `config.py` points to an old path | `1. MediaPipe BiLSTM/mtp1_original/scripts/config.py` has a `ROOT` variable; set it to `1. MediaPipe BiLSTM/mtp1_original` |

---

## 15. Limitations

* All results are on studio recordings of five people (the two native signers' low-resolution clips are excluded), with about four clips per sign. There are no live-webcam accuracy numbers.
* The accuracy of the context-aware prototype is not measured.
* Training times in the result files were recorded while other jobs shared the machine.
* The 20-sign live demo uses signs chosen for the demonstration; the fair numbers are the 98-sign held-out results in Step 7.

---

**Citation:** A. Srivastava, *Context Aware SLT System: Vector-Retrieval Recognition, Gloss-Aware Translation and Open-Set Contextual Answering for Indian Sign Language*, M.Tech Thesis, IIT Jodhpur, 2026.
