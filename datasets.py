# качаем и готовим датасеты для обучения
# запуск: python datasets.py
# что скачивается:
#   1. FairFace 108k фото, 7 классов этничности (ГЛАВНЫЙ датасет)
#   2. LFW 13k фото для эмбеддингов
#   3. FER-2013 5k фото для эмоций
#   4. UTKFace резервный если FairFace не скачался

import os, sys, ssl, time, zipfile, tarfile, shutil, csv, socket, json
import urllib.request
from pathlib import Path

DATA = Path("data")
DATA.mkdir(exist_ok=True)
ssl._create_default_https_context = ssl._create_unverified_context

def show_progress(name, count, total):
    # рисует прогресс-бар в консоли
    if total <= 0:
        return
    pct      = min(count / total, 1.0)
    filled   = int(40 * pct)
    mb       = count / 1_048_576
    total_mb = total / 1_048_576
    sys.stdout.write(
        f"\r  {name}: [{'█'*filled}{'░'*(40-filled)}]"
        f" {pct*100:.1f}%  {mb:.1f}/{total_mb:.1f} MB"
    )
    sys.stdout.flush()

def download_file(urls, dest, label=""):
    # Пробует каждый URL из списка пока один не скачается

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"{dest.name} уже скачан")
        return True

    if isinstance(urls, str):
        urls = [urls]

    label = label or dest.name

    for attempt, url in enumerate(urls, 1):
        print(f"\nскачивание {attempt}/{len(urls)}: {url[:70]}...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
            urllib.request.install_opener(opener)

            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(
                url, tmp,
                lambda c, b, t: show_progress(label, min(c*b, t), max(t, 1))
            )
            tmp.rename(dest)
            print(f"\nСкачано: {dest.name}  ({dest.stat().st_size/1_048_576:.1f} MB)")
            return True

        except Exception as e:
            print(f"\nскачивание {attempt} не прошло: {type(e).__name__}: {e}")
            tmp = dest.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
            time.sleep(1)

    return False

def check_internet():
    #проверяем интернет перед скачиванием
    try:
        socket.setdefaulttimeout(5)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except:
        return False

#  1. FairFace

#  этничности:
#  0=White  1=Black  2=Latino_Hispanic  3=East Asian
#  4=Southeast Asian  5=Indian  6=Middle Eastern

FAIRFACE_CLASSES = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian", "Indian", "Middle Eastern"
]

def _download_fairface_huggingface(out, csv_tr, csv_val):
    # Запасной вариант: скачиваем FairFace
    # Вызывается автоматически если прямые ссылки не сработали.
    print("\n  Пробуем HuggingFace datasets как запасной вариант...")

    # устанавливаем библиотеку если нет
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "datasets"],
        capture_output=True, text=True
    )

    # загружаем HuggingFace datasets напрямую из site-packages
    # нельзя делать просто "import datasets" — конфликт с нашим datasets.py
    import importlib.util, importlib.machinery
    hf_datasets = None
    for path in sys.path:
        candidate = Path(path) / "datasets" / "__init__.py"
        if candidate.exists() and "site-packages" in str(candidate):
            spec   = importlib.util.spec_from_file_location("hf_datasets", str(candidate))
            hf_datasets = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hf_datasets)
            break

    if hf_datasets is None or not hasattr(hf_datasets, "load_dataset"):
        print("  Не удалось загрузить HuggingFace datasets из site-packages")
        print("  Попробуй переименовать datasets.py во что-то другое и запустить снова")
        return False

    import csv as csv_mod

    out.mkdir(exist_ok=True)

    for split in ["train", "val"]:
        # в HuggingFace val называется validation
        hf_split = "validation" if split == "val" else "train"
        csv_path = csv_tr if split == "train" else csv_val

        # пропускаем если уже есть
        if csv_path.exists() and sum(1 for _ in (out/split).rglob("*.jpg")) > 1000:
            print(f"  {split}: уже скачан, пропускаем")
            continue

        print(f"\n  Загружаем FairFace/{hf_split} через HuggingFace...")
        print(f"  (это займёт 20-40 минут, ~2.5GB на split)")

        try:
            ds = hf_datasets.load_dataset(
                "HuggingFaceM4/FairFace",
                split=hf_split,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"  HuggingFace тоже не сработал: {e}")
            return False

        csv_rows = []
        for i, item in enumerate(ds):
            race   = item.get("race",   "")
            age    = item.get("age",    "")
            gender = item.get("gender", "")
            img    = item.get("image")

            if img is None or race not in FAIRFACE_CLASSES:
                continue

            fname   = f"{split}_{i:06d}.jpg"
            dst_dir = out / split / race
            dst_dir.mkdir(parents=True, exist_ok=True)
            img.save(str(dst_dir / fname))

            csv_rows.append({"file": fname, "age": age,
                              "gender": gender, "race": race})

            if (i+1) % 2000 == 0:
                print(f"  {split}: {i+1}/{len(ds)} фото", flush=True)

        # сохраняем CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(
                f, fieldnames=["file","age","gender","race"])
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"  ✓ {split}: {len(csv_rows)} фото скачано")

    return True

def prepare_fairface():

    #Скачиваем FairFace (~5.5GB) и раскладываем по папкам.
    #Если уже скачан — пропускаем.

    out     = DATA / "fairface"
    csv_tr  = DATA / "fairface_label_train.csv"
    csv_val = DATA / "fairface_label_val.csv"

    # проверяем что уже всё есть
    if out.exists() and csv_tr.exists() and csv_val.exists():
        n = sum(1 for _ in out.rglob("*.jpg"))
        if n > 50000:
            print(f"FairFace уже готов ({n} фото)")
            return True

    print("FairFace Главный датасет")
    print("108,501 фото 7 классов ~5.5 GB")

    # сначала скачиваем CSV
    csv_mirrors = [
        ("https://drive.google.com/uc?export=download&id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH", csv_tr),
        ("https://drive.google.com/uc?export=download&id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D",   csv_val),
    ]
    for url, dest in csv_mirrors:
        if not dest.exists():
            download_file(url, dest, dest.name)

    if not csv_tr.exists():
        _print_fairface_manual_instructions()
        # пробуем HuggingFace если прямые ссылки не сработали
        return _download_fairface_huggingface(out, csv_tr, csv_val)

    # теперь скачиваем картинки
    zip_path = DATA / "fairface_margin025.zip"
    img_mirrors = [
        "https://dataverse.harvard.edu/api/access/datafile/7412840",
        "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/main/",
        "data/fairface-img-margin025-trainval.zip"

    ]

    if not zip_path.exists():
        print("\nСкачиваем картинки FairFace (~5.5 GB)...")
        if not download_file(img_mirrors, zip_path, "FairFace"):
            # прямые ссылки не сработали — пробуем HuggingFace
            print("\n  Прямые ссылки не работают, переключаемся на HuggingFace...")
            return _download_fairface_huggingface(out, csv_tr, csv_val)

    # распаковываем
    print(f"\nРаспаковываем {zip_path.name}...")
    raw = DATA / "fairface_raw"
    if not raw.exists():
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(DATA / "fairface_raw")
            print("Распаковано")
        except Exception as e:
            print(f"Ошибка распаковки: {e}")
            return False

    # раскладываем по папкам
    print("Раскладываем по папкам...")
    out.mkdir(exist_ok=True)
    _organize_fairface(csv_tr,  raw, out, "train")
    _organize_fairface(csv_val, raw, out, "val")

    n = sum(1 for _ in out.rglob("*.jpg"))
    print(f"\nFairFace готов: {n} фото → {out}")
    return True

def _organize_fairface(csv_path, raw_root, out, split):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = 0
    for i, row in enumerate(rows):
        try:
            eth   = row.get("race", row.get("ethnicity", ""))
            if eth not in FAIRFACE_CLASSES:
                continue
            fname = Path(row["file"]).name

            src = raw_root / split / fname
            if not src.exists():
                src = raw_root / fname
            if not src.exists():
                continue

            dst_dir = out / split / eth
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / fname)
            ok += 1
        except Exception:
            continue

        if i % 2000 == 0:
            show_progress(f"FairFace/{split}", i, len(rows))

    print(f"\nFairFace/{split}: {ok} фото разложено")

def _print_fairface_manual_instructions():
    print()
    print("РУЧНАЯ ЗАГРУЗКА")
    print("Запусти в консоли python download_fairface.py ")


#  2. LFW — эмбеддинги личностей

def prepare_lfw():
    out = DATA / "lfw"
    if out.exists() and any(out.iterdir()):
        n = sum(1 for _ in out.rglob("*.jpg"))
        print(f"LFW уже готов ({n} фото)")
        return True

    # проверяем нет ли уже скачанного архива
    existing = (list(DATA.glob("lfw*.tgz")) +
                list(DATA.glob("lfw*.zip")) +
                list(DATA.glob("lfw*.tar.gz")))
    archive = existing[0] if existing else None

    if not archive:
        print("\n[LFW] Скачиваем LFW ~173 MB")
        mirrors = [
            "https://ndownloader.figshare.com/files/5976015",
            "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        ]
        for url in mirrors:
            fname = "lfw.tgz" if "tgz" in url else "lfw.zip"
            dest  = DATA / fname
            if download_file(url, dest, "LFW"):
                archive = dest
                break

    if not archive:
        print("\nРучная загрузка LFW:")
        print("https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
        print(f"Распакуй в: {out.resolve()}/Person_Name/*.jpg")
        return False

    print(f"Распаковываем {archive.name}...")
    try:
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                t.extractall(DATA)
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(DATA)
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

    # ищем куда распаковалось
    for candidate in [DATA/"lfw", DATA/"lfw_funneled", DATA/"lfw-deepfunneled"]:
        if candidate.exists() and any(candidate.iterdir()):
            if candidate != out:
                candidate.rename(out)
            n = sum(1 for _ in out.rglob("*.jpg"))
            print(f"LFW готов: {n} фото → {out}")
            return True

    print(f"Распаковано но папка не найдена. Проверь {DATA.resolve()}")
    return False

#  3. FER-2013 — эмоции

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def prepare_fer():
    out = DATA / "fer2013"
    if out.exists() and any(out.iterdir()):
        print("FER-2013 уже готов")
        return True

    # ищем папки с картинками
    candidates = [DATA/"fer2013_raw", DATA/"train", DATA/"archive", Path("train")]
    for cand in candidates:
        train_dir = cand/"train" if (cand/"train").exists() else cand
        if train_dir.exists() and any((train_dir/e).exists() for e in EMOTION_CLASSES):
            print(f"\nНайдены папки: {train_dir}")
            _copy_fer_folders(cand, out)
            return True

    # ищем старый CSV формат
    csv_path = DATA / "fer2013.csv"
    if csv_path.exists():
        print(f"\nНайден CSV: {csv_path}")
        _split_fer_csv(csv_path, out)
        return True

    print("\nFER-2013 не найден. Ручная загрузка:")
    print("1.https://www.kaggle.com/datasets/msambare/fer2013")
    print("2.Скачай и распакуй")
    print("3.Папку train/ положи в {DATA.resolve()}/fer2013_raw/train/")
    print("4.Снова запусти python datasets.py")
    return False

def _copy_fer_folders(src_root, out):
    mapping = {}
    for name in ("train", "Training", "TRAIN"):
        p = src_root / name
        if p.exists():
            mapping["train"] = p; break
    for name in ("test", "Test", "val", "validation"):
        p = src_root / name
        if p.exists():
            mapping["val"] = p; break

    total = 0
    for split, src_dir in mapping.items():
        for emo in EMOTION_CLASSES:
            src_emo = src_dir / emo
            dst_emo = out / split / emo
            if not src_emo.exists():
                continue
            dst_emo.mkdir(parents=True, exist_ok=True)
            for f in src_emo.glob("*.*"):
                if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                    shutil.copy2(f, dst_emo / f.name)
                    total += 1
    print(f"FER-2013: {total} фото → {out}")

def _split_fer_csv(csv_path, out):
    import numpy as np
    import cv2
    print("Разбиваем CSV на картинки ~2 мин...")
    for split in ("train", "val"):
        for emo in EMOTION_CLASSES:
            (out / split / emo).mkdir(parents=True, exist_ok=True)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        try:
            emo   = EMOTION_CLASSES[int(row["emotion"])]
            split = "train" if row["Usage"] == "Training" else "val"
            px    = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48,48)
            cv2.imwrite(str(out / split / emo / f"{i}.png"), px)
        except Exception:
            pass
        if i % 200 == 0:
            show_progress("FER-2013", i, len(rows))
    print(f"\nFER-2013: {len(rows)} фото → {out}")

#  4. UTKFace — резервный (если FairFace не скачался)

def prepare_utk():
    out = DATA / "utkface"
    tgz = DATA / "UTKFace.tar.gz"

    if out.exists() and any(out.iterdir()):
        print("UTKFace уже готов")
        return True

    if not tgz.exists():
        print("\nРучная загрузка UTKFace ~110 MB:")
        print("1.https://susanqq.github.io/UTKFace/")
        print("2.Aligned&Cropped Faces → Google Drive")
        print(f"3.Скачай .tar.gz → {tgz.resolve()}")
        print("Или через Kaggle:")
        print("https://www.kaggle.com/datasets/jangedoo/utkface-new")
        return False

    print("Распаковываем UTKFace...")
    out.mkdir(exist_ok=True)
    try:
        with tarfile.open(tgz) as t:
            t.extractall(out)
        print(f"UTKFace готов → {out}")
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

#  ИТОГ

def print_summary(results):
    print("ИТОГ")
    for name, ok in results.items():
        mark = "Готов" if ok else "Нужна ручная загрузка"
        print(f"  {name:<15} {mark}")

    ready = [k for k,v in results.items() if v]

    if "FairFace" in ready:
        print("\nFairFace готов — запускай обучение!")
        print("python train_pt.py --all --epochs 80")
    elif "UTKFace" in ready:
        print("\nUTKFace есть, но FairFace точнее для этничности")
        print("Можно обучать:")
        print("python train_pt.py --all --epochs 80")
    else:
        print("\nНет датасетов. Скачай либо UTKFace либо FairFace")

if __name__ == "__main__":
    print("подготовка датасетов")

    if not check_internet():
        print("Нет интернета — проверь подключение")
        sys.exit(1)
    print("Интернет есть\n")

    results = {
        "FairFace":  prepare_fairface(),
        "LFW":       prepare_lfw(),
        "FER-2013":  prepare_fer(),
        "UTKFace":   prepare_utk(),
    }

    print_summary(results)