"""
encoder_utils.py
-------------------
A small TRAINED sequence encoder (1D-CNN) that turns one sign clip into a
64-number embedding. This is the "learned embedding" option (EMBEDDING_VERSION = 4).

Why: the training-free embeddings (mean/std pooling, fixed-length trajectories)
topped out around 25-30% on your 20 signs - signers perform the same sentence
quite differently, and averages can't learn which differences matter. A
network trained on your clips learns that. The retrieval architecture stays
the same: clip -> embedding -> vector DB -> cosine nearest label -> LLM context.

How it works
  1. Landmarks -> body-relative frame features (same as v2/v3, hand-invariant).
  2. Trim idle frames (using BOTH hand and arm motion), resample to T=48 steps.
  3. 1D-CNN over time -> 64-d L2-normalised embedding.
  4. Trained with a cosine-classifier loss (so embeddings of the same sign
     cluster), with heavy augmentation: time warping, random crops, scaling,
     noise, and simulated hand drop-outs (hands vanishing for a few frames,
     which is exactly what MediaPipe does on your clips).

torch is only needed for this file (and EMBEDDING_VERSION = 4).
"""

import os

import numpy as np

import config
from embedding_utils import frame_features_v2, POSE_PTS

T_STEPS = 48
EMB_DIM = 64
FEAT_DIM = 252 + POSE_PTS * 3
_MOTION_COLS = np.r_[0:126, 252:FEAT_DIM]      # hand positions + upper-body pose (arms)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as e:
        raise ImportError("EMBEDDING_VERSION 4 and 11_train_encoder.py need PyTorch: pip install torch") from e


# ----------------------------------------------------------------- preprocessing
def active_frames(features, margin=4, min_len=12, force_swap=None):
    """(T,225) raw landmarks -> (L,303) body-relative frames with idle start/end trimmed."""
    ff = frame_features_v2(features, hand_invariant=True, force_swap=force_swap)
    if ff.shape[0] < 3:
        return ff
    sp = np.abs(np.diff(ff[:, _MOTION_COLS], axis=0)).sum(axis=1)
    k = 5
    sm = np.convolve(sp, np.ones(k) / k, mode="same")
    if sm.max() <= 1e-9:
        return ff
    idx = np.where(sm > 0.15 * sm.max())[0]
    a, b = max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 2 + margin, ff.shape[0])
    return ff[a:b] if b - a >= min_len else ff


def resample(ff, T=T_STEPS, warp=None):
    L = ff.shape[0]
    if warp is None:
        pos = np.linspace(0, L - 1, T)
    else:
        pos = warp * (L - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, L - 1)
    hi = np.clip(lo + 1, 0, L - 1)
    w = (pos - lo)[:, None]
    return ((1 - w) * ff[lo] + w * ff[hi]).astype(np.float32)


def augment(ff, rng, T=T_STEPS):
    """ff: (L,303) active frames. Returns an augmented (T,303) sequence."""
    L = ff.shape[0]
    a = int(rng.uniform(0, 0.12) * L)
    b = L - int(rng.uniform(0, 0.12) * L)
    if b - a < 8:
        a, b = 0, L
    seg = ff[a:b].copy()
    # random hand drop-outs: hands vanish for a few frames (zeros = "not detected")
    for _ in range(int(rng.integers(0, 3))):
        n = int(rng.integers(3, 10))
        s = int(rng.integers(0, max(seg.shape[0] - n, 1)))
        seg[s:s + n, 0:252] = 0.0
    # monotone random time warp
    inc = rng.uniform(0.6, 1.4, size=T)
    t = np.cumsum(inc)
    t = (t - t[0]) / (t[-1] - t[0])
    out = resample(seg, T, warp=t)
    # amplitude jitter on positional blocks
    s = rng.uniform(0.9, 1.1)
    out[:, 0:126] *= s
    out[:, 252:FEAT_DIM] *= s
    return out


# ------------------------------------------------------------------------ model
def build_model(n_classes):
    torch, nn, F = _import_torch()

    class Encoder(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.c1 = nn.Conv1d(FEAT_DIM, 96, 5, padding=2)
            self.b1 = nn.BatchNorm1d(96)
            self.c2 = nn.Conv1d(96, 128, 5, padding=2)
            self.b2 = nn.BatchNorm1d(128)
            self.c3 = nn.Conv1d(128, 128, 3, padding=1)
            self.b3 = nn.BatchNorm1d(128)
            self.drop = nn.Dropout(0.3)
            self.proj = nn.Linear(256, EMB_DIM)
            self.cls = nn.Parameter(torch.randn(max(n_classes, 1), EMB_DIM) * 0.1)
            self.scale = 16.0

        def embed(self, x):               # x: (B, T, 303) standardised
            h = x.transpose(1, 2)
            h = self.drop(F.relu(self.b1(self.c1(h))))
            h = F.max_pool1d(F.relu(self.b2(self.c2(h))), 2)
            h = F.relu(self.b3(self.c3(h)))
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return F.normalize(self.proj(self.drop(h)), dim=1)

        def logits(self, x):
            return self.scale * self.embed(x) @ F.normalize(self.cls, dim=1).T

    return Encoder(n_classes)


def train_encoder(feature_list, labels, epochs=150, seed=0, verbose=False):
    """feature_list: list of (T,225) raw landmark arrays. Returns a bundle dict."""
    torch, nn, F = _import_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    classes = sorted(set(labels))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[l] for l in labels])
    act = [active_frames(f) for f in feature_list]               # automatic dominant-hand choice
    if getattr(config, "MIRROR_AUGMENT", True):
        # both hand orientations of every clip: the encoder learns that a mirrored sign is the SAME sign
        act_a = [active_frames(f, force_swap=False) for f in feature_list]
        act_b = [active_frames(f, force_swap=True) for f in feature_list]
    else:
        act_a = act_b = act
    plain = np.stack([resample(a) for a in act_a + act_b])       # (2N,T,303)
    mean = plain.reshape(-1, FEAT_DIM).mean(axis=0)
    std = plain.reshape(-1, FEAT_DIM).std(axis=0)
    std[std < 1e-3] = 1.0

    model = build_model(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    repeats, bs = 3, 32
    steps = epochs * int(np.ceil(len(act) * repeats / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    n = len(act)
    for ep in range(epochs):
        model.train()
        order = np.tile(np.arange(n), repeats)
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            if len(ids) < 2:
                continue
            xb = np.stack([augment(act_a[i] if rng.random() < 0.5 else act_b[i], rng) for i in ids])
            xb = (xb - mean) / std + rng.normal(0, 0.08, xb.shape).astype(np.float32)
            xb_t = torch.from_numpy(xb.astype(np.float32))
            yb = torch.from_numpy(y[ids])
            loss = F.cross_entropy(model.logits(xb_t), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach())
        if verbose and (ep + 1) % 25 == 0:
            print(f"   epoch {ep+1}/{epochs}  loss {tot:.3f}")
    model.eval()
    return {"model": model, "mean": mean.astype(np.float32), "std": std.astype(np.float32), "classes": classes}


def embed_with(bundle, features, force_swap=None):
    torch, _, _ = _import_torch()
    x = (resample(active_frames(features, force_swap=force_swap)) - bundle["mean"]) / bundle["std"]
    with torch.no_grad():
        e = bundle["model"].embed(torch.from_numpy(x[None].astype(np.float32)))
    return e[0].numpy().astype(np.float32)


def embed_both(bundle, features):
    """(2,64): the clip embedded with hands kept as detected and with hands mirrored."""
    return np.stack([embed_with(bundle, features, force_swap=False),
                     embed_with(bundle, features, force_swap=True)])


# ------------------------------------------------------------------ save / load
def encoder_path():
    return os.path.join(config.PATHS["encoder_dir"], "encoder.pt")


def save_bundle(bundle, path=None):
    torch, _, _ = _import_torch()
    path = path or encoder_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": bundle["model"].state_dict(), "mean": bundle["mean"], "std": bundle["std"],
                "classes": bundle["classes"]}, path)
    return path


_CACHE = {}


def load_bundle(path=None):
    torch, _, _ = _import_torch()
    path = path or encoder_path()
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trained encoder at {path}. Run: python3 11_train_encoder.py --final")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(len(ck["classes"]))
    model.load_state_dict(ck["state"])
    model.eval()
    _CACHE[path] = {"model": model, "mean": ck["mean"], "std": ck["std"], "classes": ck["classes"]}
    return _CACHE[path]


def embed_features(features):
    return embed_with(load_bundle(), features)
