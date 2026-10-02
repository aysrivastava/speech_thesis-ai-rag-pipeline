import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PATHS


def load_gloss_map():
    gloss_map = {}
    csv_path = PATHS["gloss_map"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Gloss map not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            gloss_map[label] = {
                "label": label,
                "gloss": row["gloss"].strip(),
                "is_question": row["is_question"].strip(),
                "wh_type": row["wh_type"].strip(),
                "polarity": row["polarity"].strip(),
                "tense": row["tense"].strip(),
                "aspect": row["aspect"].strip(),
                "notes": row["notes"].strip(),
            }
    return gloss_map


def label_to_gloss(label: str):
    gloss_map = load_gloss_map()
    if label not in gloss_map:
        return {
            "label": label,
            "gloss": "",
            "is_question": "unknown",
            "wh_type": "unknown",
            "polarity": "unknown",
            "tense": "unknown",
            "aspect": "unknown",
            "notes": "label not found in gloss_map.csv",
        }
    return gloss_map[label]


def save_output(result: dict):
    out_dir = PATHS["gloss"]
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result["label"].replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"{safe_name}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/06_label_to_gloss.py "your sentence label"')
        sys.exit(1)

    label = " ".join(sys.argv[1:]).strip()
    result = label_to_gloss(label)
    out_file = save_output(result)

    print("=" * 55)
    print("LABEL TO GLOSS")
    print("=" * 55)
    print("Label      :", result["label"])
    print("Gloss      :", result["gloss"])
    print("Question   :", result["is_question"])
    print("WH Type    :", result["wh_type"])
    print("Polarity   :", result["polarity"])
    print("Tense      :", result["tense"])
    print("Aspect     :", result["aspect"])
    print("Notes      :", result["notes"])
    print("Saved to   :", out_file)
    print("=" * 55)
