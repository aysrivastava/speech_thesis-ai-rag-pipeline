"""
context_utils.py
------------------
The context-aware layer that sits ON TOP of retrieval (steps 01-05 are not
touched). Given the ranked retrieval candidates and (optionally) the
question the signer was answering, it decides what the answer was.

THE DECISION RULE (read this - it is the method section of a paper)
===================================================================
Two independent pieces of evidence about "which sign was performed":

  1. Retrieval evidence (from the video):
        r_i = softmax_i( cosine_sim_i / T )            T = RETRIEVAL_TEMPERATURE
  2. Context evidence (from the question, via an LLM that NEVER sees the
     video or the similarity scores):
        p_i = plausibility of answering the question with candidate i, in [0,1]

Posterior over candidates:
        post_i  ∝  r_i * (p_i + eps) ** lambda

i.e. a product of experts / Bayes rule with the question acting as a prior
over plausible answers and retrieval acting as the likelihood.

Why this form (and not "ask the LLM to pick"): it makes the LLM's power
over the final answer BOUNDED and explainable. A candidate with p=0 can
still win if retrieval is overwhelmingly sure, but context can only
override retrieval when the similarity gap between the retrieval winner and
the context favourite is smaller than

        max_override_gap = T * lambda * ln((1 + eps) / eps)      (~0.09 by default)

so "a plausible-sounding answer" can never beat a clearly better visual
match. That bound is printed on every decision. It directly answers the
failure mode of LLM re-ranking (plausibility bias: the LLM prefers what
sounds right over what was signed).

Possible outcomes (res["decision"]):
  RETRIEVAL_ONLY        no question given; retrieval's top-1 is the answer
  ACCEPT_TOP1           context agrees with retrieval's top-1
  OVERRIDE              context moved a different candidate to the top
                        (printed prominently - this is the "best match 2
                        makes more sense" case)
  NO_PLAUSIBLE_CANDIDATE  even the most plausible candidate is below
                        NONE_FITS_MAX_PLAUSIBILITY -> the answer is probably
                        NOT in the vocabulary -> out-of-vocabulary fallback
  BELOW_SIM_FLOOR       (only if config.OOV_SIM_FLOOR is calibrated/set) the
                        best match is too weak to trust -> fallback
  CONTEXT_UNAVAILABLE   no API key / call failed / unparseable answer; falls
                        back to retrieval's top-1 and says so

The out-of-vocabulary fallback (run_oov_fallback) asks a VLM to describe the
clip and propose a label, then asks YOU to confirm before anything is
written to the DB (provenance 'vlm_confirmed'). After that the sign is
recognised by plain retrieval - the VLM is only a bootstrap.
"""

import math

import numpy as np

import config
import db_utils
import llm_utils

USE_CONFIG = object()  # sentinel: "read the value from config at call time"


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def max_override_gap(temperature=None, epsilon=None, lam=None):
    """Largest similarity gap (retrieval winner minus context favourite) that
    context is still able to overturn."""
    T = temperature if temperature is not None else config.RETRIEVAL_TEMPERATURE
    eps = epsilon if epsilon is not None else config.CONTEXT_EPSILON
    lam = lam if lam is not None else config.CONTEXT_LAMBDA
    return T * lam * math.log((1.0 + eps) / eps)


def fuse(sims, plausibilities, temperature=None, epsilon=None, lam=None):
    """Returns (retrieval_p, posterior), both normalised to sum to 1."""
    T = temperature if temperature is not None else config.RETRIEVAL_TEMPERATURE
    eps = epsilon if epsilon is not None else config.CONTEXT_EPSILON
    lam = lam if lam is not None else config.CONTEXT_LAMBDA
    retrieval_p = softmax(np.asarray(sims, dtype=np.float64) / T)
    unnorm = retrieval_p * (np.asarray(plausibilities, dtype=np.float64) + eps) ** lam
    return retrieval_p, unnorm / unnorm.sum()


def resolve_with_context(scored, question, top_k=None, sim_floor=USE_CONFIG, plausibility_fn=None):
    """
    scored: list of (label, source_video, cosine_sim) sorted best-first, one
            entry per label (retrieval_utils guarantees this).
    Returns a result dict (see module docstring for decisions).
    """
    top_k = top_k or config.RERANK_TOP_K
    floor = config.OOV_SIM_FLOOR if sim_floor is USE_CONFIG else sim_floor
    cands = scored[:top_k]

    res = {
        "decision": None,
        "question": question,
        "retrieval_top1": cands[0][0] if cands else None,
        "retrieval_top1_sim": float(cands[0][2]) if cands else None,
        "final_label": None,
        "candidates": [],
        "reason": "",
        "needs_fallback": False,
        "max_override_gap": max_override_gap(),
    }
    if not cands:
        res["decision"] = "EMPTY_DB"
        res["reason"] = "The vector DB has no stored signs."
        return res

    labels = [c[0] for c in cands]
    sims = np.array([c[2] for c in cands], dtype=np.float64)
    retrieval_p = softmax(sims / config.RETRIEVAL_TEMPERATURE)

    def fill(plaus=None, post=None):
        res["candidates"] = [
            {
                "label": labels[i],
                "sim": float(sims[i]),
                "retrieval_p": float(retrieval_p[i]),
                "plausibility": None if plaus is None else float(plaus[i]),
                "posterior": float(retrieval_p[i] if post is None else post[i]),
            }
            for i in range(len(labels))
        ]

    if floor is not None and sims[0] < floor:
        fill()
        res["decision"] = "BELOW_SIM_FLOOR"
        res["needs_fallback"] = True
        res["reason"] = (f"Best cosine similarity {sims[0]:.3f} is below the out-of-vocabulary "
                         f"floor {floor:.3f}, so '{labels[0]}' is not trusted as a match.")
        return res

    if not question:
        fill()
        res["decision"] = "RETRIEVAL_ONLY"
        res["final_label"] = labels[0]
        res["reason"] = "No question given, so retrieval's top-1 is used as is."
        return res

    plausibility_fn = plausibility_fn or llm_utils.rerank_plausibility
    try:
        plaus, llm_reason = plausibility_fn(question, labels)
    except Exception as e:  # LLMUnavailable, parse errors, network errors...
        fill()
        res["decision"] = "CONTEXT_UNAVAILABLE"
        res["final_label"] = labels[0]
        res["reason"] = f"Context step skipped ({type(e).__name__}: {e}). Using retrieval's top-1."
        return res

    plaus = np.asarray(plaus, dtype=np.float64)
    _, post = fuse(sims, plaus)
    fill(plaus, post)

    if plaus.max() < config.NONE_FITS_MAX_PLAUSIBILITY:
        res["decision"] = "NO_PLAUSIBLE_CANDIDATE"
        res["needs_fallback"] = True
        res["reason"] = (f"No retrieved candidate is a plausible answer to the question "
                         f"(best plausibility {plaus.max():.2f} < {config.NONE_FITS_MAX_PLAUSIBILITY:.2f}). "
                         f"LLM: {llm_reason}")
        return res

    winner = int(np.argmax(post))
    res["final_label"] = labels[winner]
    if winner == 0:
        res["decision"] = "ACCEPT_TOP1"
        res["reason"] = f"Context agrees with retrieval. LLM: {llm_reason}"
    else:
        res["decision"] = "OVERRIDE"
        gap = sims[0] - sims[winner]
        res["reason"] = (f"Retrieval ranked '{labels[0]}' first (sim {sims[0]:.3f}), but given the question "
                         f"'{labels[winner]}' (sim {sims[winner]:.3f}, gap {gap:.3f}) is far more plausible. "
                         f"LLM: {llm_reason}")
    return res


def print_context_decision(res):
    print("\n" + "=" * 70)
    print("CONTEXT-AWARE DECISION")
    print("=" * 70)
    if res.get("question"):
        print(f"Question put to the signer: {res['question']}")
    if res["candidates"]:
        print(f"\n{'Rank':<6}{'Label':<24}{'Sim':>7}{'Retrieval%':>12}{'Plausible':>11}{'Final%':>9}")
        for i, c in enumerate(res["candidates"], start=1):
            plaus = "-" if c["plausibility"] is None else f"{c['plausibility']:.2f}"
            mark = "  <-- final" if c["label"] == res["final_label"] else ""
            print(f"{i:<6}{c['label']:<24}{c['sim']:>7.3f}{c['retrieval_p']*100:>11.1f}%"
                  f"{plaus:>11}{c['posterior']*100:>8.1f}%{mark}")

    d = res["decision"]
    print()
    if d == "OVERRIDE":
        print("*" * 70)
        print(f"CONTEXT OVERRIDE: retrieval's #1 was '{res['retrieval_top1']}', "
              f"but the answer chosen is '{res['final_label']}'.")
        print("*" * 70)
    print(f"Decision: {d}")
    print(f"Why     : {res['reason']}")
    if d in ("OVERRIDE", "ACCEPT_TOP1", "NO_PLAUSIBLE_CANDIDATE"):
        print(f"(Context can only overturn retrieval when the similarity gap is under "
              f"{res['max_override_gap']:.3f}; a clearly better visual match always wins.)")


def run_oov_fallback(result, question, known_labels, input_fn=input):
    """
    Out-of-vocabulary fallback: VLM describes the clip (blind), proposes a
    label, and the USER confirms before anything is stored.
    Returns None if the fallback couldn't run, else a dict:
      {"proposal": {...}, "description": {...}, "enrolled": bool, "label": str|None}
    """
    print("\n" + "-" * 70)
    print("OUT-OF-VOCABULARY FALLBACK: the retrieved signs don't explain this clip.")
    if not llm_utils.llm_is_configured():
        print("(Set ANTHROPIC_API_KEY to let a vision model propose what the clip shows.)")
        return None

    video = result.get("query_video_path")
    try:
        print(f"[fallback] Sampling {config.VLM_NUM_FRAMES} frames and asking the vision model "
              f"(pass 1: blind description)...")
        description = llm_utils.vlm_describe_clip(video)
        print(f"[fallback]   sees: {description.get('description')}")
        print(f"[fallback]   pantomime-like: {description.get('looks_like_pantomime')} | "
              f"iconic action: {description.get('iconic_action')} | "
              f"pointing at: {description.get('pointing_at')} | "
              f"confidence {description['confidence']:.2f}")
        print("[fallback] (pass 2: proposing a label, now using the question)...")
        proposal = llm_utils.vlm_propose_label(description, question, known_labels)
    except Exception as e:
        print(f"[fallback] Could not complete the vision step ({type(e).__name__}: {e}).")
        return None

    print(f"[fallback]   proposed label : {proposal['proposed_label']}")
    print(f"[fallback]   answer text    : {proposal.get('answer_text')}")
    print(f"[fallback]   confidence     : {proposal['confidence']:.2f}")
    print(f"[fallback]   reason         : {proposal.get('reason')}")
    print("[fallback] NOTE: a vision model reads iconic / pantomime / pointing content well, but it")
    print("[fallback] cannot reliably read arbitrary lexical signs. Treat this as a hypothesis.")

    out = {"proposal": proposal, "description": description, "enrolled": False, "label": None}
    if not proposal["proposed_label"]:
        print("[fallback] The model could not propose a label, so nothing to enroll.")
        return out

    ans = input_fn(
        f"\nEnroll this clip as '{proposal['proposed_label']}'? "
        f"[y = yes, n = no, or type a different label]: "
    ).strip()
    if ans.lower() in ("", "n", "no"):
        print("[fallback] Not enrolled.")
        return out

    from enrollment_utils import enroll_from_result  # lazy: pulls in mediapipe-free code only
    if ans.lower() in ("y", "yes"):
        label, provenance = proposal["proposed_label"], db_utils.PROVENANCE_VLM_CONFIRMED
    else:
        label, provenance = ans, db_utils.PROVENANCE_MANUAL  # the human chose the label
    try:
        enroll_from_result(result, label, provenance)
    except Exception as e:
        print(f"[fallback] Enrollment failed ({type(e).__name__}: {e}).")
        return out
    out["enrolled"], out["label"] = True, label
    return out


def process_result(result, question=None, allow_fallback=True, input_fn=input):
    """One call that does everything after retrieval: context decision,
    optional out-of-vocabulary fallback, final answer banner."""
    res = resolve_with_context(result["scored"], question)
    print_context_decision(res)

    outcome = {"context": res, "fallback": None, "final_label": res["final_label"]}

    if res["needs_fallback"] and allow_fallback:
        known = [s[0] for s in result["scored"]]
        fb = run_oov_fallback(result, question, known, input_fn=input_fn)
        outcome["fallback"] = fb
        if fb and fb["enrolled"]:
            outcome["final_label"] = fb["label"]

    print("\n" + "*" * 70)
    if outcome["final_label"]:
        print(f"FINAL ANSWER: '{outcome['final_label']}'")
    else:
        hint = ""
        fb = outcome["fallback"]
        if fb and fb["proposal"].get("proposed_label"):
            hint = f"  (unconfirmed vision-model hypothesis: '{fb['proposal']['proposed_label']}')"
        print(f"FINAL ANSWER: not in the vocabulary{hint}")
    print("*" * 70)
    return outcome
