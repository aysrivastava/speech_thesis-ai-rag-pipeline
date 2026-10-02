import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PATHS


TEMPORAL_SIGNS = {
    "YESTERDAY", "TODAY", "TOMORROW", "BEFORE", "LATER", "NOW", "PAST", "FUTURE", "TIME"
}

CLAUSE_FINAL_PARTICLES = {
    "NEG", "COMPL", "G-WH"
}

WH_ASSOCIATES = {
    "FACE": "who",
    "PLACE": "where",
    "TIME": "when",
    "NUMBER": "how_many",
}

PRONOUNS = {
    "ME", "I", "YOU", "HE", "SHE", "WE", "THEY", "INDEX1", "INDEX2", "INDEX3"
}


def tokenize_gloss(gloss: str):
    if not gloss:
        return []
    gloss = re.sub(r"\s+", " ", gloss.strip())
    return gloss.split(" ")


def infer_wh_type(tokens, fallback_wh_type):
    if fallback_wh_type and fallback_wh_type.lower() != "none":
        return fallback_wh_type

    for tok in tokens:
        if tok in WH_ASSOCIATES:
            return WH_ASSOCIATES[tok]

    if "G-WH" in tokens:
        return "unknown"

    return "none"


def extract_temporals(tokens):
    temporals = [t for t in tokens if t in TEMPORAL_SIGNS]
    remaining = [t for t in tokens if t not in TEMPORAL_SIGNS]
    return temporals, remaining


def extract_clause_final_particles(tokens, polarity, aspect, is_question):
    particles = []
    remaining = []

    for t in tokens:
        if t in CLAUSE_FINAL_PARTICLES:
            particles.append(t)
        else:
            remaining.append(t)

    if polarity == "negative" and "NEG" not in particles:
        particles.append("NEG")

    if aspect.lower() == "completive" and "COMPL" not in particles:
        particles.append("COMPL")

    if str(is_question).lower() in {"yes", "true"} and "G-WH" not in particles:
        particles.append("G-WH")

    ordered = []
    if "NEG" in particles:
        ordered.append("NEG")
    if "COMPL" in particles:
        ordered.append("COMPL")
    if "G-WH" in particles:
        ordered.append("G-WH")

    return remaining, ordered


def reorder_predicate_final(tokens):
    if len(tokens) <= 2:
        return tokens

    subjects = []
    objects = []
    predicates = []
    others = []

    verb_like_suffixes = (
        "GO", "COME", "SLEEP", "EAT", "HELP", "TRUST", "THINK", "TELL", "NEED",
        "WANT", "DO", "LEAVE", "CRY", "HURT", "DIE", "SIT", "TAKE", "MAKE",
        "UNDERSTAND", "APPRECIATE", "REPEAT", "SPEAK", "PREPARE", "SERVE",
        "BECOME", "PLAN", "CHAT", "FEEL", "STOP", "WEAR", "POUR", "TURN"
    )

    for i, tok in enumerate(tokens):
        upper_tok = tok.upper()

        if upper_tok in PRONOUNS and not subjects:
            subjects.append(tok)
        elif any(upper_tok.endswith(v) for v in verb_like_suffixes) or upper_tok in verb_like_suffixes:
            predicates.append(tok)
        elif subjects and not predicates:
            objects.append(tok)
        else:
            others.append(tok)

    if not predicates:
        return tokens

    normalized = subjects + objects + others + predicates
    return normalized


def normalize_gloss_record(record: dict):
    raw_gloss = record.get("gloss", "").strip()
    tokens = tokenize_gloss(raw_gloss)

    temporals, tokens_wo_temporal = extract_temporals(tokens)

    wh_type = infer_wh_type(tokens_wo_temporal, record.get("wh_type", "none"))

    core_tokens, clause_final_particles = extract_clause_final_particles(
        tokens_wo_temporal,
        record.get("polarity", "positive"),
        record.get("aspect", "none"),
        record.get("is_question", "no")
    )

    predicate_final_core = reorder_predicate_final(core_tokens)

    normalized_tokens = temporals + predicate_final_core + clause_final_particles
    normalized_gloss = " ".join(normalized_tokens).strip()

    applied_rules = []
    if temporals:
        applied_rules.append("temporal_to_clause_initial")
    if predicate_final_core != core_tokens:
        applied_rules.append("predicate_final_reordering")
    if clause_final_particles:
        applied_rules.append("clause_final_particle_ordering")
    if "G-WH" in clause_final_particles:
        applied_rules.append("wh_final_enforcement")
    if "NEG" in clause_final_particles and "G-WH" in clause_final_particles:
        applied_rules.append("neg_before_gwh")

    return {
        "label": record.get("label", ""),
        "raw_gloss": raw_gloss,
        "normalized_gloss": normalized_gloss,
        "tokens_raw": tokens,
        "tokens_normalized": normalized_tokens,
        "is_question": record.get("is_question", "no"),
        "wh_type": wh_type,
        "polarity": record.get("polarity", "positive"),
        "tense": record.get("tense", "present"),
        "aspect": record.get("aspect", "none"),
        "notes": record.get("notes", ""),
        "applied_rules": applied_rules
    }


def load_gloss_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Gloss JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_normalized_json(result: dict):
    out_dir = PATHS["gloss"]
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = result["label"].replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"{safe_name}_normalized.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def resolve_input_to_json(user_input: str):
    candidate = Path(user_input)

    if candidate.exists():
        return candidate

    safe_name = user_input.strip().replace(" ", "_").replace("/", "_")
    guessed = PATHS["gloss"] / f"{safe_name}.json"
    if guessed.exists():
        return guessed

    raise FileNotFoundError(
        f"Could not resolve input '{user_input}' to a gloss JSON file."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/07_normalize_gloss.py "where are you from"')
        print('   or: python3 scripts/07_normalize_gloss.py outputs/gloss/where_are_you_from.json')
        sys.exit(1)

    input_arg = " ".join(sys.argv[1:]).strip()
    input_json_path = resolve_input_to_json(input_arg)

    record = load_gloss_json(input_json_path)
    normalized = normalize_gloss_record(record)
    out_file = save_normalized_json(normalized)

    print("=" * 60)
    print("GLOSS NORMALIZATION")
    print("=" * 60)
    print("Label            :", normalized["label"])
    print("Raw Gloss        :", normalized["raw_gloss"])
    print("Normalized Gloss :", normalized["normalized_gloss"])
    print("Question         :", normalized["is_question"])
    print("WH Type          :", normalized["wh_type"])
    print("Polarity         :", normalized["polarity"])
    print("Tense            :", normalized["tense"])
    print("Aspect           :", normalized["aspect"])
    print("Applied Rules    :", ", ".join(normalized["applied_rules"]) if normalized["applied_rules"] else "none")
    print("Saved to         :", out_file)
    print("=" * 60)
