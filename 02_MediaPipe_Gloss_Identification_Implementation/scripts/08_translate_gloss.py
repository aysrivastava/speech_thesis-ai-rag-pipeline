import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PATHS


PRONOUN_MAP = {
    "ME": "I",
    "I": "I",
    "YOU": "you",
    "HE": "he",
    "SHE": "she",
    "WE": "we",
    "THEY": "they",
    "INDEX1": "I",
    "INDEX2": "you",
    "INDEX3": "he/she",
}

WH_WORD_MAP = {
    "where": "where",
    "who": "who",
    "when": "when",
    "how_many": "how many",
    "what": "what",
    "unknown": "what",
    "none": "",
}

TIME_MAP = {
    "YESTERDAY": "yesterday",
    "TODAY": "today",
    "TOMORROW": "tomorrow",
    "NOW": "now",
    "BEFORE": "before",
    "LATER": "later",
    "PAST": "in the past",
    "FUTURE": "in the future",
}

LEXICAL_MAP = {
    "FROM": "from",
    "PLACE": "place",
    "FACE": "person",
    "TIME": "time",
    "NUMBER": "number",
    "FREE": "free",
    "HELP": "help",
    "WATER": "water",
    "NEED": "need",
    "SLEEP": "sleep",
    "FRIEND": "friend",
    "FATHER": "father",
    "DIE": "die",
    "UNDERSTAND": "understand",
    "MAN": "man",
    "COME": "come",
    "GO": "go",
    "TRAIN": "train",
    "FOOD": "food",
    "EAT": "eat",
    "HUNGRY": "hungry",
    "NAME": "name",
    "GOOD": "good",
    "BAD": "bad",
    "THANK": "thank",
    "SORRY": "sorry",
    "CRY": "cry",
    "HURT": "hurt",
    "ANGRY": "angry",
    "TRUTH": "truth",
    "SPEAK": "speak",
    "SLOW": "slow",
    "REPEAT": "repeat",
    "TRUST": "trust",
    "CAREER": "career",
    "PHONE": "phone",
    "BEAUTIFUL": "beautiful",
    "OUTSIDE": "outside",
}

AUXILIARY_VERBS = {
    "I": "do",
    "you": "do",
    "we": "do",
    "they": "do",
    "he": "does",
    "she": "does",
    "he/she": "does",
}

BE_VERBS = {
    "I": "am",
    "you": "are",
    "we": "are",
    "they": "are",
    "he": "is",
    "she": "is",
    "he/she": "is",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_input_to_json(user_input: str):
    candidate = Path(user_input)
    if candidate.exists():
        return candidate

    safe_name = user_input.strip().replace(" ", "_").replace("/", "_")
    guessed = PATHS["gloss"] / f"{safe_name}_normalized.json"
    if guessed.exists():
        return guessed

    raise FileNotFoundError(
        f"Could not resolve input '{user_input}' to a normalized gloss JSON file."
    )


def gloss_token_to_english(token: str):
    if token in PRONOUN_MAP:
        return PRONOUN_MAP[token]
    if token in TIME_MAP:
        return TIME_MAP[token]
    if token in LEXICAL_MAP:
        return LEXICAL_MAP[token]
    if token in {"NEG", "COMPL", "G-WH"}:
        return ""
    return token.lower()


def clean_words(words):
    return [w for w in words if w and w.strip()]


def sentence_case(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_grammar_summary(record: dict):
    summary = {
        "is_question": record.get("is_question", "no"),
        "wh_type": record.get("wh_type", "none"),
        "polarity": record.get("polarity", "positive"),
        "tense": record.get("tense", "present"),
        "aspect": record.get("aspect", "none"),
        "applied_rules": record.get("applied_rules", []),
    }
    return summary


def rule_based_translation(record: dict):
    label = record.get("label", "").strip()
    normalized_gloss = record.get("normalized_gloss", "").strip()
    tokens = record.get("tokens_normalized", [])
    is_question = str(record.get("is_question", "no")).lower() in {"yes", "true"}
    wh_type = record.get("wh_type", "none")
    polarity = record.get("polarity", "positive").lower()
    aspect = record.get("aspect", "none").lower()

    if label:
        simple_label = label.strip()
        if is_question and not simple_label.endswith("?"):
            return sentence_case(simple_label) + "?"
        elif not is_question:
            return sentence_case(simple_label) + "."

    temporal_words = [gloss_token_to_english(t) for t in tokens if t in TIME_MAP]
    content_tokens = [t for t in tokens if t not in {"NEG", "COMPL", "G-WH"} and t not in TIME_MAP]
    words = clean_words([gloss_token_to_english(t) for t in content_tokens])

    subject = None
    remaining = words[:]
    if remaining:
        first = remaining[0]
        if first in {"I", "you", "he", "she", "we", "they", "he/she"}:
            subject = first
            remaining = remaining[1:]

    if is_question:
        wh_word = WH_WORD_MAP.get(wh_type, "what")

        if wh_type == "where" and subject and "from" in remaining:
            be = BE_VERBS.get(subject, "are")
            return f"{wh_word.capitalize()} {be} {subject} from?"

        if wh_type == "who" and remaining:
            return f"{wh_word.capitalize()} {' '.join(remaining)}?"

        if wh_type == "when" and remaining:
            aux = AUXILIARY_VERBS.get(subject, "do") if subject else "does"
            base = " ".join(([subject] if subject else []) + remaining)
            return f"{wh_word.capitalize()} {aux} {base}?"

        if subject:
            aux = AUXILIARY_VERBS.get(subject, "do")
            predicate = " ".join(remaining).strip()
            if predicate:
                return f"{wh_word.capitalize()} {aux} {subject} {predicate}?"
            return f"{wh_word.capitalize()} {aux} {subject}?"

        return f"{wh_word.capitalize()} {' '.join(remaining)}?".strip()

    pieces = []
    if temporal_words:
        pieces.extend(temporal_words)

    if subject:
        pieces.append(subject)

    if remaining:
        pieces.extend(remaining)

    sentence = " ".join(clean_words(pieces)).strip()

    if polarity == "negative":
        if subject:
            be = BE_VERBS.get(subject, "is")
            if "hungry" in remaining or "angry" in remaining or "free" in remaining:
                sentence = f"{subject} {be} not {' '.join(remaining)}"
            else:
                aux = AUXILIARY_VERBS.get(subject, "do")
                sentence = f"{subject} {aux} not {' '.join(remaining)}"
        else:
            sentence = sentence + " not"

    if aspect == "completive":
        if sentence:
            sentence = sentence + " completely"

    sentence = sentence_case(sentence)
    if not sentence.endswith("."):
        sentence += "."

    return sentence


def build_llm_prompt(record: dict, translation: str):
    prompt = {
        "task": "Translate normalized ISL/IndSL gloss into fluent English.",
        "label": record.get("label", ""),
        "raw_gloss": record.get("raw_gloss", ""),
        "normalized_gloss": record.get("normalized_gloss", ""),
        "grammar_features": {
            "is_question": record.get("is_question", "no"),
            "wh_type": record.get("wh_type", "none"),
            "polarity": record.get("polarity", "positive"),
            "tense": record.get("tense", "present"),
            "aspect": record.get("aspect", "none"),
            "applied_rules": record.get("applied_rules", []),
        },
        "baseline_translation": translation,
        "instruction": (
            "Generate a natural English sentence that preserves the meaning of the "
            "normalized gloss and uses the grammar features as constraints."
        ),
    }
    return prompt


def build_translation_record(record: dict):
    translation = rule_based_translation(record)
    grammar_summary = build_grammar_summary(record)
    llm_prompt = build_llm_prompt(record, translation)

    return {
        "label": record.get("label", ""),
        "raw_gloss": record.get("raw_gloss", ""),
        "normalized_gloss": record.get("normalized_gloss", ""),
        "grammar_summary": grammar_summary,
        "english_translation": translation,
        "translation_method": "rule_based_baseline",
        "llm_prompt_placeholder": llm_prompt,
    }


def save_translation_json(result: dict):
    out_dir = PATHS["translations"]
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = result["label"].replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"{safe_name}_translation.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/08_translate_gloss.py "where are you from"')
        print('   or: python3 scripts/08_translate_gloss.py outputs/gloss/where_are_you_from_normalized.json')
        sys.exit(1)

    input_arg = " ".join(sys.argv[1:]).strip()
    input_json_path = resolve_input_to_json(input_arg)

    normalized_record = load_json(input_json_path)
    translation_record = build_translation_record(normalized_record)
    out_file = save_translation_json(translation_record)

    print("=" * 60)
    print("GLOSS TO ENGLISH TRANSLATION")
    print("=" * 60)
    print("Label               :", translation_record["label"])
    print("Raw Gloss           :", translation_record["raw_gloss"])
    print("Normalized Gloss    :", translation_record["normalized_gloss"])
    print("English Translation :", translation_record["english_translation"])
    print("Method              :", translation_record["translation_method"])
    print("Saved to            :", out_file)
    print("=" * 60)
