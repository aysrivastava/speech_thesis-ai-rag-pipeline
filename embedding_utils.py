"""
embedding_utils.py
---------------------
The actual embedding function, pulled out into its own importable module
(instead of living only inside 03_...py) so that BOTH:
  - 03_converting_extractions_to_vector_embeddings.py (building the DB)
  - 05_semantic_search_test.py (embedding a fresh query clip)
call the exact same code. This matters for the same reason
landmark_extraction_utils.py is shared: if the query were embedded even
slightly differently from how the DB was built, cosine similarity between
them would be comparing two different things.

(Python module names can't start with a digit when imported with a plain
`import 03_...` statement, which is why this logic lives here rather than
being imported "from" the numbered script directly - the numbered scripts
stay as the clear, run-these-in-order entry points, and both of them
import this shared helper.)

See the big comment block in 03_converting_extractions_to_vector_embeddings.py
for the full reasoning on WHY this particular embedding method (training-free
statistical motion-pooling) was chosen for this first pass of the pipeline.
"""

import numpy as np


def compute_embedding_v1(features, verbose=True, label=""):
    """
    features: np.ndarray of shape (T, 225) - one row per frame, 225 landmark
              values per frame (63 right hand + 63 left hand + 99 pose).
    returns : np.ndarray of shape (900,), L2-normalized float32.

    The 900 dims are, in order:
      [0:225]   mean position of each of the 225 landmark values over time
      [225:450] std of position of each landmark value over time
      [450:675] mean frame-to-frame velocity of each landmark value
      [675:900] std of that velocity (how erratic the motion is)
    """
    T = features.shape[0]
    if T < 2:
        raise ValueError(f"Need at least 2 frames to compute velocity, got {T}")

    mean_pos = features.mean(axis=0)                  # (225,)
    std_pos = features.std(axis=0)                     # (225,)

    velocity = np.diff(features, axis=0)                # (T-1, 225)
    mean_vel = velocity.mean(axis=0)                    # (225,)
    std_vel = velocity.std(axis=0)                       # (225,)

    raw_vector = np.concatenate([mean_pos, std_pos, mean_vel, std_vel])  # (900,)

    norm = np.linalg.norm(raw_vector)
    if norm < 1e-8:
        print(f"[embed]   WARNING: near-zero vector for '{label}' (no detected motion at all?)")
        normalized = raw_vector
    else:
        normalized = raw_vector / norm

    if verbose:
        print(f"[embed] '{label}': T={T} frames")
        print(f"[embed]   mean_pos  stats -> min={mean_pos.min():.4f} max={mean_pos.max():.4f}")
        print(f"[embed]   std_pos   stats -> min={std_pos.min():.4f} max={std_pos.max():.4f}")
        print(f"[embed]   mean_vel  stats -> min={mean_vel.min():.4f} max={mean_vel.max():.4f}")
        print(f"[embed]   std_vel   stats -> min={std_vel.min():.4f} max={std_vel.max():.4f}")
        print(f"[embed]   raw vector norm (before normalizing): {norm:.4f}")
        print(f"[embed]   final embedding shape: {normalized.shape}, "
              f"L2 norm after normalize: {np.linalg.norm(normalized):.4f}")
        print(f"[embed]   first 8 values: {np.round(normalized[:8], 4).tolist()}")

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# VERSION 2 - why it exists
# ---------------------------------------------------------------------------
# v1 pooled RAW image coordinates. Raw positions are ~0.3-0.9 while per-frame
# velocities are ~0.001-0.01, so after L2 normalisation the 225 mean-position
# values were ~99% of the vector. Every clip then looked almost the same
# (cosine ~0.97 to everything) and the winner was decided by who stood where
# in the frame, not by what was signed. One stored clip that happened to sit
# near "the average posture" became a "hub" that won most queries.
#
# v2 changes three things:
#   1. Body-relative coordinates: hands/pose are expressed relative to the
#      shoulder midpoint and divided by shoulder width, per frame. Standing
#      closer, further, or off-centre no longer changes the vector.
#   2. Hand SHAPE: each hand relative to its own wrist, scaled by hand size.
#      Handshape is a core part of a sign and was only implicit before.
#   3. Block balancing: the four statistics blocks (mean pos, std pos, mean
#      velocity, std velocity) are each L2-normalised before concatenation,
#      so velocity/motion gets a voice instead of being drowned out.
# Retrieval additionally mean-centres vectors (see retrieval_utils.py).

_R = slice(0, 63)
_L = slice(63, 126)
_P = slice(126, 225)
_BLOCK_WEIGHTS = (1.0, 1.0, 1.0, 0.7)   # mean pos, std pos, mean vel, std vel


def _body_frame(pose):
    """pose: (T, 99). Returns (center (T,3), scale (T,), valid (T,)) per frame,
    with frames where pose is missing filled from the clip's median."""
    T = pose.shape[0]
    pts = pose.reshape(T, 33, 3)
    valid = np.any(pose != 0, axis=1)
    ls, rs = pts[:, 11, :], pts[:, 12, :]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm((ls - rs)[:, :2], axis=1)
    valid = valid & (scale > 1e-3)
    if valid.any():
        c_fill = np.median(center[valid], axis=0)
        s_fill = float(np.median(scale[valid]))
    else:
        c_fill = np.array([0.5, 0.5, 0.0])
        s_fill = 0.25
    center = np.where(valid[:, None], center, c_fill[None, :])
    scale = np.where(valid, scale, s_fill)
    center = center.copy()
    center[:, 2] = 0.0
    return center, scale, valid


def _hand_features(hand, center, scale):
    """hand: (T, 63). Returns (pos (T,63), shape (T,63)); zeros where undetected."""
    T = hand.shape[0]
    pts = hand.reshape(T, 21, 3)
    det = np.any(hand != 0, axis=1)
    pos = (pts - center[:, None, :]) / scale[:, None, None]
    wrist = pts[:, 0:1, :]
    size = np.linalg.norm(pts[:, 9, :] - pts[:, 0, :], axis=1)
    size = np.maximum(size, 1e-3)
    shp = (pts - wrist) / size[:, None, None]
    pos = np.where(det[:, None, None], pos, 0.0).reshape(T, 63)
    shp = np.where(det[:, None, None], shp, 0.0).reshape(T, 63)
    return pos, shp


def _hand_motion_energy(pos):
    """How much the wrist moves, over frames where the hand was seen."""
    det = np.any(pos != 0, axis=1)
    if det.sum() < 3:
        return 0.0
    w = pos.reshape(pos.shape[0], 21, 3)[det, 0, :2]
    return float(w.std(axis=0).sum())


POSE_PTS = 23   # nose/eyes/ears/mouth (0-10), shoulders..wrists (11-16) AND the pose model's coarse
                # pinky/index/thumb points (17-22) - these still give a rough hand position when
                # the dedicated hand model misses the hand.
_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]


def _mirror_perm_matrix(a):
    """Swap left/right pose points (for a (T, 3*POSE_PTS) block) - used with x-flip when mirroring."""
    b = a.reshape(a.shape[0], POSE_PTS, 3).copy()
    for i, j in _MIRROR_PAIRS:
        b[:, [i, j], :] = b[:, [j, i], :]
    return b.reshape(a.shape[0], -1)


def _negate_x(a):
    """Mirror left<->right: flip the x coordinate of a (T, 3k) block."""
    b = a.copy().reshape(a.shape[0], -1, 3)
    b[:, :, 0] *= -1
    return b.reshape(a.shape[0], -1)


def frame_features_v2(features, hand_invariant=False, force_swap=None):
    """(T, 225) raw landmarks -> (T, 321) body-relative frame features.

    hand_invariant=True makes the features independent of WHICH hand is which
    and of mirroring: if the left hand moves more than the right, the two
    hand blocks are swapped and x is flipped (so the more active hand is
    always in the 'right' slot, as if the clip were mirrored back). This
    covers left-handed signers AND front-camera clips saved mirrored, which
    otherwise swap the right/left blocks and make identical signs look
    different."""
    features = np.asarray(features, dtype=np.float64)
    T = features.shape[0]
    center, scale, valid = _body_frame(features[:, _P])
    r_pos, r_shp = _hand_features(features[:, _R], center, scale)
    l_pos, l_shp = _hand_features(features[:, _L], center, scale)
    pose_pts = features[:, _P].reshape(T, 33, 3)[:, :POSE_PTS, :]
    pose_rel = (pose_pts - center[:, None, :]) / scale[:, None, None]
    pose_rel = np.where(valid[:, None, None], pose_rel, 0.0).reshape(T, POSE_PTS * 3)
    if force_swap is not None:          # True = treat as mirrored, False = keep as is (overrides auto)
        do_swap = bool(force_swap)
    else:
        do_swap = hand_invariant and _hand_motion_energy(l_pos) > _hand_motion_energy(r_pos)
    if do_swap:
        r_pos, l_pos = _negate_x(l_pos), _negate_x(r_pos)
        r_shp, l_shp = _negate_x(l_shp), _negate_x(r_shp)
        pose_rel = _negate_x(_mirror_perm_matrix(pose_rel))
    return np.concatenate([r_pos, l_pos, r_shp, l_shp, pose_rel], axis=1)


def compute_embedding_v2(features, verbose=True, label="", hand_invariant=False):
    ff = frame_features_v2(features, hand_invariant=hand_invariant)
    T = ff.shape[0]
    if T < 2:
        raise ValueError(f"Need at least 2 frames to compute velocity, got {T}")
    vel = np.diff(ff, axis=0)
    blocks = [ff.mean(axis=0), ff.std(axis=0), vel.mean(axis=0), vel.std(axis=0)]
    parts = []
    for b, w in zip(blocks, _BLOCK_WEIGHTS):
        n = np.linalg.norm(b)
        parts.append(w * (b / n if n > 1e-8 else b))
    raw = np.concatenate(parts)
    norm = np.linalg.norm(raw)
    out = raw / norm if norm > 1e-8 else raw
    if verbose:
        print(f"[embed] '{label}': T={T} frames -> v2 embedding {out.shape} (body-relative, block-balanced)")
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# VERSION 3 - keeps the ORDER of the sign
# ---------------------------------------------------------------------------
# Mean/std pooling (v1, v2) throws away time order: "hand goes up then out"
# and "out then up" pool to the same numbers. v3 adds a time-ordered part:
#   1. trim the idle start/end (frames where the hands barely move), so
#      signers who start late or hold the pose at the end line up,
#   2. resample the body-relative frame features to K evenly spaced points,
#   3. flatten (K x 303) -> a vector that encodes the trajectory in order.
# It is concatenated with the v2 statistics. Always hand-invariant.

def _active_segment(ff, min_len=8):
    speed = np.abs(np.diff(ff[:, :126], axis=0)).sum(axis=1)
    if speed.size < min_len or speed.max() <= 1e-9:
        return 0, ff.shape[0]
    k = 5
    sm = np.convolve(speed, np.ones(k) / k, mode="same")
    idx = np.where(sm > 0.2 * sm.max())[0]
    if idx.size == 0:
        return 0, ff.shape[0]
    start, end = int(idx[0]), int(idx[-1]) + 2
    if end - start < min_len:
        return 0, ff.shape[0]
    return start, min(end, ff.shape[0])


_PARTS = {
    "all": slice(0, 252 + POSE_PTS * 3),
    "pose": slice(252, 252 + POSE_PTS * 3),
    "hands": slice(0, 252),
}


def temporal_vector(features, K=12, trim=True, hand_invariant=True, parts="all"):
    ff = frame_features_v2(features, hand_invariant=hand_invariant)
    if trim:
        a, b = _active_segment(ff)
        ff = ff[a:b]
    ff = ff[:, _PARTS[parts]]
    T = ff.shape[0]
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, K)
    res = np.stack([np.interp(dst, src, ff[:, d]) for d in range(ff.shape[1])], axis=1)  # (K, 303)
    v = res.reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def compute_embedding_v3(features, verbose=True, label="", K=12, use_stats=True, use_temporal=True, parts="all"):
    parts_list = []
    if use_stats:
        parts_list.append(compute_embedding_v2(features, verbose=False, hand_invariant=True).astype(np.float64))
    if use_temporal:
        parts_list.append(temporal_vector(features, K=K, parts=parts))
    raw = np.concatenate(parts_list)
    n = np.linalg.norm(raw)
    out = raw / n if n > 1e-8 else raw
    if verbose:
        print(f"[embed] '{label}': v3 embedding {out.shape} (stats={use_stats}, temporal K={K})")
    return out.astype(np.float32)


def compute_query_embeddings(features, verbose=True, label=""):
    """Embeddings to use for a QUERY clip: a list of vectors.
    With the trained encoder (v4) and config.QUERY_BOTH_ORIENTATIONS, the clip is embedded in BOTH
    hand orientations (as signed, and mirrored: left<->right hand), and retrieval keeps the better
    match. This covers signers whose dominant hand differs from the stored signer's, including cases
    where the automatic 'which hand is dominant' guess is wrong. Otherwise a single vector."""
    import config
    if getattr(config, "EMBEDDING_VERSION", 1) == 4 and getattr(config, "QUERY_BOTH_ORIENTATIONS", True):
        import encoder_utils
        both = encoder_utils.embed_both(encoder_utils.load_bundle(), features)
        return [both[0], both[1]]
    return [compute_motion_pooled_embedding(features, verbose=verbose, label=label)]


def compute_motion_pooled_embedding(features, verbose=True, label=""):
    """The embedding used everywhere (03, 05, 06, 07, webcam tool).
    Version is chosen by config.EMBEDDING_VERSION (1, 2 or 3)."""
    import config
    v = getattr(config, "EMBEDDING_VERSION", 1)
    if v == 4:
        import encoder_utils
        return encoder_utils.embed_features(features)
    if v == 3:
        return compute_embedding_v3(features, verbose=verbose, label=label, K=config.TEMPORAL_K)
    if v == 2:
        return compute_embedding_v2(features, verbose=verbose, label=label,
                                    hand_invariant=getattr(config, "HAND_INVARIANT", False))
    return compute_embedding_v1(features, verbose=verbose, label=label)
