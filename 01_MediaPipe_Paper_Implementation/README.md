# ISL Sign Language Recognition

Real-time Indian Sign Language (ISL) sentence recognition using MediaPipe + BiLSTM.

ISL_SLT_Project/
├── config.py # All paths and hyperparameters
├── scripts/
│ ├── 01_download.py # Download ISL-CSLRT dataset (8.3 GB)
│ ├── 02_download_models.py # Download MediaPipe .task models
│ ├── 03_extract.py # Extract hand + pose landmarks → .pkl
│ ├── 04_train.py # Train BiLSTM classifier
│ └── 05_webcam_demo.py # Live webcam demo (auto-capture)
└── notebooks/
└── explore_dataset.ipynb # Dataset exploration


## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install mediapipe opencv-python pandas numpy tqdm scikit-learn kagglehub matplotlib jupyter ipykernel
```

## Run in order
```bash
python3 scripts/01_download.py
python3 scripts/02_download_models.py
python3 scripts/03_extract.py
python3 scripts/04_train.py
python3 scripts/05_webcam_demo.py
```

## Results
- Dataset : ISL-CSLRT Corpus — 97 sentences, 7 signers
- Model   : 2-layer Bidirectional LSTM
- Input   : 50 frames x 225 landmarks
- Val Accuracy : 91.1%
