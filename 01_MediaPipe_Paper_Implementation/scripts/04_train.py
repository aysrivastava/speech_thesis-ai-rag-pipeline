# scripts/04_train.py
# ── What this script does ─────────────────────────────────────────────────────
# Loads all 663 .pkl feature files from features/
# Encodes sentence labels as numeric class IDs using LabelEncoder
# Splits data 80/20 train/val using stratified split
#   (ensures every class appears in both train and val sets)
# Trains a Bidirectional LSTM classifier:
#   Input  : (batch, 50, 225) — 50 frames × 225 landmark values
#   LSTM   : 2 layers, 256 hidden, bidirectional → 512 output
#   FC     : 512 → 256 → 128 → num_classes
# Saves best_model.pt whenever val accuracy improves
# Saves last_model.pt every epoch (resume checkpoint)
# Saves label_encoder.pkl (needed by webcam demo to decode predictions)
# Run: python3 scripts/04_train.py
# Expected time: ~5-10 minutes on Apple MPS
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, DEVICE, DEVICE_NAME, SEQ_LEN, FEAT_DIM, HIDDEN, EPOCHS, LR, BATCH

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class SignDataset(Dataset):
    """
    Wraps numpy arrays X (features) and y (labels) as a PyTorch dataset.
    X shape : (N, SEQ_LEN, FEAT_DIM) = (663, 50, 225)
    y shape : (N,) integer class IDs
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Model definition ──────────────────────────────────────────────────────────
class SignClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for ISL sentence recognition.

    Architecture:
      BiLSTM (2 layers, hidden=256, bidirectional)
        → concatenate final forward + backward hidden states → 512 values
        → FC(512→256) + ReLU + Dropout(0.3)
        → FC(256→128) + ReLU + Dropout(0.2)
        → FC(128→num_classes)
        → output: raw logits per class (softmax applied at inference)
    """
    def __init__(self, input_dim=225, hidden=256, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        # h: (num_layers*2, batch, hidden) — *2 becse bidirectional
        _, (h, _) = self.lstm(x)
        # Concatenate last forward (h[-2]) and last backward (h[-1]) hidden states
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


def augment(features):
    """
    Creates synthetic variations of a feature sequence to multiply training data.
    Augmentation types:
      - Gaussian noise    : adds tiny random jitter to all landmark coords
      - Time shift        : rolls the sequence forward/backward by a few frames
      - Scale             : slightly shrinks/enlarges the landmark positions
    Each call returns ONE augmented copy — called multiple times per sample.
    """
    features = np.array(features, dtype=np.float32)

    # Gaussian noise — simulates slight hand tremor / detection noise
    noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
    aug1  = np.clip(features + noise, 0, 1)

    # Time shift — simulates signing slightly earlier or later
    shift = np.random.randint(-3, 4)
    aug2  = np.roll(features, shift, axis=0)

    # Scale — simulates person being closer/farther from camera
    scale = np.random.uniform(0.92, 1.08)
    aug3  = np.clip(features * scale, 0, 1)

    return [aug1.tolist(), aug2.tolist(), aug3.tolist()]


def load_features():
    pkl_files = sorted(PATHS['features'].glob('*.pkl'))
    assert len(pkl_files) > 0, "❌ No pkl files found. Run 03_extract.py first."

    print(f"Loading {len(pkl_files)} pkl files...")
    X_list, y_list = [], []

    for fp in tqdm(pkl_files, desc="Loading + augmenting", unit="file"):
        data     = pickle.loads(fp.read_bytes())
        features = data['features'] if isinstance(data, dict) else data
        label    = data['label']    if isinstance(data, dict) else fp.stem

        if len(features) < SEQ_LEN:
            features += [[0.0] * FEAT_DIM] * (SEQ_LEN - len(features))
        else:
            features = features[:SEQ_LEN]

        # Original sample
        X_list.append(features)
        y_list.append(label)

        # 3 augmented copies per sample → 4× more training data
        for aug in augment(features):
            X_list.append(aug)
            y_list.append(label)

    X  = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(y_list)

    PATHS['label_encoder'].write_bytes(pickle.dumps(le))
    print(f"✅ Label encoder saved")
    print(f"✅ Augmented: {len(pkl_files)} → {len(X_list)} samples (4×)")

    return X, y, le


def train():
    print("=" * 55)
    print("MODEL TRAINING")
    print("=" * 55)
    print(f"Device    : {DEVICE_NAME}")
    print(f"Seq len   : {SEQ_LEN}  |  Feat dim: {FEAT_DIM}  |  Hidden: {HIDDEN}")
    print(f"Epochs    : {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, le = load_features()
    num_classes = len(le.classes_)
    samples_per_class = len(X) // num_classes

    print(f"\nTotal samples     : {len(X)}")
    print(f"Classes           : {num_classes}")
    print(f"Samples per class : ~{samples_per_class}")

    # ── Train / val split ─────────────────────────────────────────────────────
    # StratifiedShuffleSplit ensures every class has representation in both sets
    if samples_per_class >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_i, va_i = next(sss.split(X, y))
        X_tr, X_va = X[tr_i], X[va_i]
        y_tr, y_va = y[tr_i], y[va_i]
        print(f"Train set         : {len(X_tr)} samples")
        print(f"Val set           : {len(X_va)} samples")
    else:
        # Too few samples per class for a split — train on everything
        X_tr, X_va = X, X
        y_tr, y_va = y, y
        print("⚠ Less than 2 samples/class — training on full set (no val split)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs       = min(BATCH, len(X_tr))
    train_dl = DataLoader(SignDataset(X_tr, y_tr), batch_size=bs,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(SignDataset(X_va, y_va), batch_size=bs,
                          shuffle=False, num_workers=0)

    # ── Model, optimiser, scheduler, loss ────────────────────────────────────
    model = SignClassifier(input_dim=FEAT_DIM, num_classes=num_classes).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingLR smoothly reduces LR from LR → ~0 over EPOCHS
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters  : {total_params:,}")
    print(f"\nTraining started...\n")

    best_acc  = -1.0
    start     = time.time()

    for epoch in range(EPOCHS):
        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            # Clip gradients to prevent exploding gradient in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg_loss = total_loss / len(train_dl)

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                _, pred = torch.max(model(bx), 1)
                correct += (pred == by).sum().item()
                total   += by.size(0)
        acc = correct / total if total > 0 else 0.0
        lr_now = sched.get_last_lr()[0]

        # ── Save last checkpoint every epoch (for resume) ─────────────────────
        torch.save({
            'epoch'       : epoch,
            'model_state' : model.state_dict(),
            'opt_state'   : opt.state_dict(),
            'best_acc'    : best_acc,
            'num_classes' : num_classes,
            'feat_dim'    : FEAT_DIM,
            'seq_len'     : SEQ_LEN,
        }, PATHS['ckpt_last'])

        # ── Save best checkpoint when accuracy improves ───────────────────────
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state' : model.state_dict(),
                'num_classes' : num_classes,
                'feat_dim'    : FEAT_DIM,
                'seq_len'     : SEQ_LEN,
            }, PATHS['ckpt_best'])
            best_marker = f"  ← ✅ BEST saved ({best_acc:.4f})"
        else:
            best_marker = ""

        # Print every 5 epochs or when accuracy jumps
        if (epoch + 1) % 5 == 0 or acc >= 0.5 or best_marker:
            elapsed = (time.time() - start) / 60
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val Acc: {acc:.4f} | "
                  f"LR: {lr_now:.6f} | "
                  f"{elapsed:.1f}m{best_marker}")

        # ── Early stop at perfect accuracy ────────────────────────────────────
        if acc == 1.0:
            print(f"\n🎯 Perfect accuracy at epoch {epoch+1}. Stopping early.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = (time.time() - start) / 60
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Best Val Accuracy : {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"  Total time        : {total_time:.1f} minutes")
    print(f"  Best model saved  : {PATHS['ckpt_best']}")
    print(f"  Last model saved  : {PATHS['ckpt_last']}")
    print(f"  Label encoder     : {PATHS['label_encoder']}")
    print("=" * 55)
    print("➡ Next: python3 scripts/05_webcam_demo.py")


if __name__ == "__main__":
    train()