"""
llm_utils.py
--------------
Everything that talks to an LLM/VLM, in one place, so the rest of the
pipeline stays testable without network access or an API key.

Three jobs:

1. rerank_plausibility(question, labels)
   TEXT-ONLY call. Given the question put to the signer and the labels of
   the top retrieval candidates, returns a 0-1 plausibility per candidate:
   "how natural is it to answer THIS question with THIS sign?"

   Two deliberate design choices:
   - The LLM is NOT shown the similarity scores. context_utils.py combines
     retrieval and plausibility itself (as a product of two independent
     pieces of evidence). If the LLM saw the scores it would anchor on
     them and the two signals would no longer be independent.
   - Candidates are presented in SHUFFLED order under letters (A, B, C...),
     then mapped back. LLMs have position bias; shuffling stops "whatever
     retrieval ranked first" from quietly getting a head start.

2. sample_frames_b64(video_path)
   Evenly-spaced JPEG frames from a short clip, base64-encoded. Used
   because image-based APIs take images, not video files.

3. vlm_describe_clip(...) then vlm_propose_label(...)  (TWO passes)
   Pass 1 looks at the frames WITHOUT being told the question, and just
   describes what it sees. Pass 2 (text only) takes that description plus
   the question and proposes a short label. Splitting it this way reduces
   the main failure mode of asking a VLM "what did they answer?": it
   tends to see what the question makes likely instead of what's there.

   IMPORTANT limitation, stated plainly: a VLM reading sampled frames can
   recognise ICONIC / pantomime / pointing content ("holding something to
   the mouth" -> eating). It cannot reliably read arbitrary lexical signs.
   Treat its output as a HYPOTHESIS for a human to confirm, never as
   ground truth. Every enrollment it triggers goes through a confirmation
   prompt for exactly this reason.

Configuration: set ANTHROPIC_API_KEY in your shell (never in a file you
commit). Model name: config.LLM_MODEL (override with env RAG_LLM_MODEL).
Every call is appended to llm_logs/llm_calls.jsonl (image data is logged
only as a count, not the pixels).
"""

import base64
import json
import os
import random
import re
import time
from datetime import datetime

import cv2

import config

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Test hook: when set, _call() uses this instead of the real API.
# Signature: fn(system: str, content_blocks: list) -> str
_BACKEND = None


class LLMUnavailable(RuntimeError):
    """Raised when no API key / SDK is available. Callers catch this and
    degrade gracefully (retrieval-only) instead of crashing the demo."""


def set_backend_for_testing(fn):
    global _BACKEND
    _BACKEND = fn


def llm_is_configured():
    return _BACKEND is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMUnavailable(
            "ANTHROPIC_API_KEY is not set. In your terminal run:\n"
            "   export ANTHROPIC_API_KEY='your-key-here'\n"
            "then re-run (the key stays in your shell session, not in any file)."
        )
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("The 'anthropic' package isn't installed. Run: pip install anthropic") from e
    return anthropic.Anthropic()


def _log_call(kind, system, content_blocks, response_text, seconds, error=None):
    try:
        os.makedirs(config.PATHS["llm_logs"], exist_ok=True)
        text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        n_images = sum(1 for b in content_blocks if b.get("type") == "image")
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "model": config.LLM_MODEL,
            "system": system,
            "user_text": "\n".join(text_parts),
            "num_images": n_images,
            "response": response_text,
            "seconds": round(seconds, 2),
            "error": error,
        }
        with open(os.path.join(config.PATHS["llm_logs"], "llm_calls.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break a demo


def _call(kind, system, content_blocks):
    t0 = time.time()
    try:
        if _BACKEND is not None:
            text = _BACKEND(system, content_blocks)
        else:
            client = _get_client()
            resp = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": content_blocks}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception as e:
        _log_call(kind, system, content_blocks, None, time.time() - t0, error=f"{type(e).__name__}: {e}")
        raise
    _log_call(kind, system, content_blocks, text, time.time() - t0)
    return text


def parse_json_loose(text):
    """Models sometimes wrap JSON in ```json fences or add a sentence around
    it. Pull out the outermost {...} and parse that."""
    if text is None:
        raise ValueError("empty response")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# 1. Text-only plausibility rerank
# ---------------------------------------------------------------------------

RERANK_SYSTEM = (
    "You judge whether a candidate sign is a plausible reply to a question put to a signer. "
    "You cannot see any video. You only see the question and a list of candidate signs "
    "(glosses from a closed Indian Sign Language vocabulary). "
    "Plausible means a person could naturally answer that question with that sign. "
    "Ignore the order of the candidates. Do not reward a candidate just because it is a "
    "common phrase. If no candidate is a sensible reply, give every candidate a low score. "
    "Respond with JSON only."
)


def rerank_plausibility(question, labels, seed=None):
    """
    Returns (plausibilities, reason): a list of floats in [0, 1] aligned
    with `labels`, plus the model's one-line reason.
    Raises ValueError on an unparseable/incomplete answer, LLMUnavailable if
    no key/SDK - callers decide how to degrade.
    """
    n = len(labels)
    if n == 0:
        return [], ""
    if n > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} candidates supported, got {n}")

    order = list(range(n))
    random.Random(seed).shuffle(order)  # position-bias guard
    lines = [f"{LETTERS[i]}. {labels[idx]}" for i, idx in enumerate(order)]
    keys = ", ".join(f'"{LETTERS[i]}"' for i in range(n))

    user_text = (
        f"Question put to the signer: {question}\n\n"
        f"The signer replied with ONE sign. Candidates:\n" + "\n".join(lines) + "\n\n"
        f"For each candidate give a plausibility between 0.0 (no sensible person would "
        f"answer that way) and 1.0 (a very natural answer).\n"
        f'Reply with JSON only, in exactly this shape:\n'
        f'{{"plausibility": {{{keys}}}, "reason": "<one short sentence>"}}\n'
        f"(each key maps to a number)."
    )

    text = _call("rerank", RERANK_SYSTEM, [{"type": "text", "text": user_text}])
    data = parse_json_loose(text)
    plaus_by_letter = data.get("plausibility")
    if not isinstance(plaus_by_letter, dict):
        raise ValueError(f"missing 'plausibility' object in response: {text[:200]!r}")

    result = [None] * n
    for i, idx in enumerate(order):
        letter = LETTERS[i]
        if letter not in plaus_by_letter:
            raise ValueError(f"response missing candidate {letter}: {text[:200]!r}")
        result[idx] = _clamp01(plaus_by_letter[letter])
    return result, str(data.get("reason", "")).strip()


# ---------------------------------------------------------------------------
# 2. Frame sampling
# ---------------------------------------------------------------------------

def sample_frames_b64(video_path, num_frames=None, max_side=None):
    """
    Returns a list of base64-encoded JPEG strings, evenly spaced across the
    clip in time order. Reads the clip sequentially (no seeking - seeking is
    unreliable on some phone codecs like HEVC .mov) and downsizes each frame
    as it goes, so memory stays small. Clips are only a few seconds long.
    """
    num_frames = num_frames or config.VLM_NUM_FRAMES
    max_side = max_side or config.VLM_MAX_SIDE_PX

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    if len(frames) <= num_frames:
        picks = list(range(len(frames)))
    else:
        picks = sorted({round(i * (len(frames) - 1) / (num_frames - 1)) for i in range(num_frames)})

    out = []
    for idx in picks:
        ok, buf = cv2.imencode(".jpg", frames[idx], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            out.append(base64.standard_b64encode(buf.tobytes()).decode("ascii"))
    return out


# ---------------------------------------------------------------------------
# 3. Two-pass VLM: describe blind, then propose a label with the question
# ---------------------------------------------------------------------------

VLM_DESCRIBE_SYSTEM = (
    "You describe what a person in a short video clip is doing with their hands, face and body. "
    "The clip may show sign language, a pantomime (miming an action), or pointing at something. "
    "Describe only what is visible in the frames. Do not guess words you cannot actually see. "
    "State whether the movement looks like an iconic/pantomime action or like an arbitrary "
    "lexical sign you cannot interpret. Respond with JSON only."
)

VLM_PROPOSE_SYSTEM = (
    "You turn a description of a signer's video into a short English label for the sign or "
    "action they conveyed. Be conservative: if the description is ambiguous, or describes "
    "movement you cannot interpret, give a low confidence and a null label. If the description "
    "matches a label from the known vocabulary, use that exact label. Respond with JSON only."
)


def vlm_describe_clip(video_path, num_frames=None):
    frames = sample_frames_b64(video_path, num_frames=num_frames)
    blocks = []
    for b64 in frames:
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    blocks.append({
        "type": "text",
        "text": (
            f"These are {len(frames)} frames in time order from one short clip. "
            "Reply with JSON only, in exactly this shape:\n"
            '{"description": "<what the person does, in order>", '
            '"hands_used": "one|both|none", '
            '"pointing_at": "<what is pointed at, or null>", '
            '"iconic_action": "<the real-world action being mimed, or null>", '
            '"looks_like_pantomime": true|false, '
            '"confidence": 0.0-1.0}'
        ),
    })
    text = _call("vlm_describe", VLM_DESCRIBE_SYSTEM, blocks)
    data = parse_json_loose(text)
    data["confidence"] = _clamp01(data.get("confidence", 0.0))
    data["num_frames_sent"] = len(frames)
    return data


def vlm_propose_label(description, question, known_labels):
    """Text-only second pass. `description` is the dict from vlm_describe_clip."""
    desc_for_prompt = {k: v for k, v in description.items() if k != "num_frames_sent"}
    q_line = f"Question put to the signer: {question}\n\n" if question else "No question was asked.\n\n"
    vocab = ", ".join(known_labels) if known_labels else "(empty)"
    user_text = (
        q_line
        + "Blind description of the signer's clip (written without seeing the question):\n"
        + json.dumps(desc_for_prompt, indent=2)
        + f"\n\nKnown vocabulary (labels already in the system): {vocab}\n\n"
        "Reply with JSON only, in exactly this shape:\n"
        '{"proposed_label": "<1-3 word English label, or null>", '
        '"answer_text": "<a short natural answer to the question, or null>", '
        '"in_known_vocabulary": true|false, '
        '"confidence": 0.0-1.0, '
        '"reason": "<one short sentence>"}'
    )
    text = _call("vlm_propose", VLM_PROPOSE_SYSTEM, [{"type": "text", "text": user_text}])
    data = parse_json_loose(text)
    data["confidence"] = _clamp01(data.get("confidence", 0.0))
    label = data.get("proposed_label")
    data["proposed_label"] = str(label).strip() if label not in (None, "", "null") else None
    return data
