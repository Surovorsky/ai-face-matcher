# скрипт для скачивания FairFace через HuggingFace
# запускай ОТДЕЛЬНО: python download_fairface.py
#

import sys, csv, shutil
from pathlib import Path

DATA = Path("data")
DATA.mkdir(exist_ok=True)

FAIRFACE_CLASSES = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian", "Indian", "Middle Eastern"
]

try:
    import datasets
    print(f"HuggingFace datasets {datasets.__version__} загружен")
except ImportError:
    print("Устанавливаем datasets...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
    import datasets
    print(f"HuggingFace datasets {datasets.__version__} установлен")

out     = DATA / "fairface"
csv_tr  = DATA / "fairface_label_train.csv"
csv_val = DATA / "fairface_label_val.csv"

# проверяем что уже всё есть
if out.exists() and csv_tr.exists() and csv_val.exists():
    n = sum(1 for _ in out.rglob("*.jpg"))
    if n > 50000:
        print(f"FairFace уже готов ({n} фото) — ничего делать не нужно")
        sys.exit(0)

out.mkdir(exist_ok=True)

for split in ["train", "val"]:
    hf_split = "validation" if split == "val" else "train"
    csv_path = csv_tr if split == "train" else csv_val

    # пропускаем если уже скачан
    existing = sum(1 for _ in (out / split).rglob("*.jpg")) if (out / split).exists() else 0
    if csv_path.exists() and existing > 10000:
        print(f"{split}: уже скачан ({existing} фото), пропускаем")
        continue

    print(f"\nЗагружаем FairFace/{hf_split}...")
    print("(~2.5GB, 20-40 минут)")

    ds = datasets.load_dataset(
        "HuggingFaceM4/FairFace",
        "0.25",        # margin=0.25 — стандартная версия датасета
        split=hf_split
    )


    img_key    = "image"
    race_key   = "race"
    age_key    = "age"
    gender_key = "gender"

    # race хранится как целое число: 0=White 1=Black 2=Latino_Hispanic
    # 3=East Asian 4=Southeast Asian 5=Indian 6=Middle Eastern
    RACE_INT_MAP = {i: name for i, name in enumerate(FAIRFACE_CLASSES)}

    csv_rows = []
    for i, item in enumerate(ds):
        race_raw = item.get(race_key)
        age      = str(item.get(age_key,    "")).strip()
        gender   = str(item.get(gender_key, "")).strip()
        img      = item.get(img_key)

        if img is None or race_raw is None:
            continue

        # конвертируем int → название класса
        race_norm = RACE_INT_MAP.get(int(race_raw))
        if race_norm is None:
            continue

        fname   = f"{split}_{i:06d}.jpg"
        dst_dir = out / split / race_norm
        dst_dir.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(img, "save"):
                img.save(str(dst_dir / fname))
            else:
                from PIL import Image
                import io
                if isinstance(img, bytes):
                    Image.open(io.BytesIO(img)).save(str(dst_dir / fname))
                else:
                    Image.fromarray(img).save(str(dst_dir / fname))
        except Exception as e:
            print(f"  Ошибка сохранения фото {i}: {e}")
            continue

        csv_rows.append({"file": fname, "age": age,
                         "gender": gender, "race": race_norm})

        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(ds)} фото", flush=True)

    # сохраняем CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file","age","gender","race"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"{split}: {len(csv_rows)} фото, CSV сохранён → {csv_path}")

n = sum(1 for _ in out.rglob("*.jpg"))
print(f"\nFairFace готов: {n} фото → {out}")
print("\nТеперь запускай:")
print("  python train_pt.py --all --epochs 80")