# scripts/01_download.py
# ── What this script does ─────────────────────────────────────────────────────
# Downloads the ISL-CSLRT dataset (8.3 GB) from Kaggle to data/raw/
# Safe to re-run — skips download if dataset already exists
# Run from project root: python3 scripts/01_download.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Add project root to path so config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.config import PATHS

import kagglehub, shutil, os

DATA_DIR = PATHS['data_raw']

def download_dataset():
    # ── Check if already downloaded ───────────────────────────────────────────
    if (DATA_DIR / 'ISL_CSLRT_Corpus').exists():
        sent = PATHS['frames_sent']
        word = PATHS['frames_word']
        s_count = len(list(sent.iterdir())) if sent.exists() else 0
        w_count = len(list(word.iterdir())) if word.exists() else 0
        print("✅ Dataset already exists — skipping download.")
        print(f"   Sentence folders : {s_count}")
        print(f"   Word folders     : {w_count}")
        return

    # ── Download via kagglehub to temp cache ──────────────────────────────────
    print("=" * 55)
    print("DOWNLOADING ISL-CSLRT DATASET")
    print("=" * 55)
    print("Size     : ~8.3 GB")
    print("Source   : Kaggle — drblack00/isl-csltr-indian-sign-language-dataset")
    print("Dest     : data/raw/")
    print("Note     : One-time download — never needed again after this")
    print("=" * 55)

    tmp_cache = Path('/tmp/isl_kaggle_cache')
    tmp_cache.mkdir(parents=True, exist_ok=True)
    os.environ['KAGGLE_CACHE_DIR'] = str(tmp_cache)

    print("\n📥 Downloading...")
    path = kagglehub.dataset_download("drblack00/isl-csltr-indian-sign-language-dataset")
    print(f"✅ Downloaded to temp: {path}")

    # ── Copy to project data/raw with progress ────────────────────────────────
    src_root       = Path(path)
    all_files      = [f for f in src_root.rglob('*') if f.is_file()]
    total          = len(all_files)
    total_gb       = sum(f.stat().st_size for f in all_files) / 1e9

    print(f"\n📂 Copying {total:,} files ({total_gb:.1f} GB) to data/raw/")
    print("   Progress shown every 1000 files...\n")

    copied = 0
    errors = 0
    for src in all_files:
        rel = src.relative_to(src_root)
        dst = DATA_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src), str(dst))
            copied += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠ Copy error: {rel} — {e}")

        if copied % 1000 == 0:
            pct = copied / total * 100
            print(f"  [{pct:5.1f}%]  {copied:,} / {total:,} files copied")

    print(f"\n✅ Copy complete — {copied:,} files  |  Errors: {errors}")

    # ── Clean up temp cache ───────────────────────────────────────────────────
    shutil.rmtree(str(tmp_cache), ignore_errors=True)
    print("✅ Temp cache cleaned")


def verify():
    # ── Verify final structure ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DATASET VERIFICATION")
    print("=" * 55)

    checks = {
        "Sentence Level" : PATHS['frames_sent'],
        "Word Level"     : PATHS['frames_word'],
        "CSV files"      : PATHS['corpus_csv'],
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            subdirs  = [d for d in path.iterdir() if d.is_dir()]
            images   = list(path.rglob('*.jpg')) + list(path.rglob('*.png'))
            size_mb  = sum(f.stat().st_size for f in images) / 1e6
            print(f"  ✅ {name:<20} {len(subdirs):>4} folders  |  {len(images):>6,} images  |  {size_mb:>7,.0f} MB")
        else:
            print(f"  ❌ {name:<20} NOT FOUND at {path}")
            all_ok = False

    print("=" * 55)
    if all_ok:
        print("✅ Dataset fully ready — never run this script again")
        print("➡ Next: python3 scripts/02_download_models.py")
    else:
        print("⚠ Some folders missing — re-run this script")


if __name__ == "__main__":
    download_dataset()
    verify()