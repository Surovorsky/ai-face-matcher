# обучение всех моделей Face AI

import os, sys, json, time, random, copy
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import (Dataset, DataLoader, random_split,
                                   WeightedRandomSampler, Subset)
    from torchvision import transforms, models
    import torch.optim as optim
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    print("[ОШИБКА] PyTorch не установлен.")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ОШИБКА] OpenCV не установлен: pip install opencv-python")
    sys.exit(1)

# папки
DATA   = Path("data")
MODELS = Path("models/pt")
MODELS.mkdir(parents=True, exist_ok=True)

# размеры
IMG_SIZE  = 112    # размер входа для всех моделей
EMBED_DIM = 256    # размерность эмбеддинга лица
BATCH     = 64     # большой батч работает с FP16

# классы FairFace — 7 штук, правильные лейблы
# ВАЖНО: совпадают с main_camera.py и telegram_bot.py
FAIRFACE_CLASSES = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian", "Indian", "Middle Eastern"
]
N_ETH = len(FAIRFACE_CLASSES)   # 7

# определяем девайс и включаем FP16 если есть CUDA
if torch.cuda.is_available():
    DEVICE  = torch.device("cuda")
    USE_AMP = True   # FP16 mixed precision — работает на RTX 4050
    print(f"[PT] Девайс: CUDA ({torch.cuda.get_device_name(0)})")
    print(f"[PT] PyTorch {torch.__version__} | FP16: ВКЛ")
elif torch.backends.mps.is_available():
    DEVICE  = torch.device("mps")
    USE_AMP = False  # Apple MPS пока не поддерживает AMP
    print(f"[PT] Девайс: MPS (Apple Silicon)")
else:
    DEVICE  = torch.device("cpu")
    USE_AMP = False
    print(f"[PT] Девайс: CPU | FP16: ОТКЛ")
    if "+cpu" in torch.__version__:
        print("     CPU PyTorch — GPU не используется!")
        print("     pip uninstall torch torchvision -y")
        print("     pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")

#  АУГМЕНТАЦИИ


TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5,
                           saturation=0.4, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet статистика
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

# валидация без аугментации
VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


EMO_TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((56, 56)),
    transforms.RandomCrop(48),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

EMO_VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

#  ДАТАСЕТЫ

class FairFaceDataset(Dataset):
    # FairFace — 108k фото, 7 классов этничности.
    # Структура: data/fairface/train/White/*.jpg
    # Кеширует всё в RAM (~3GB) — после первой загрузки каждый эпох быстрый.

    def __init__(self, root, split="train", transform=TRAIN_TF, cache=True):
        self.tf     = transform
        self.items  = []    # (путь, возраст, пол, индекс_этничности)
        self._cache = {}    # кеш картинок в RAM

        root    = Path(root)
        split_d = root / split
        if not split_d.exists():
            raise FileNotFoundError(f"Папка не найдена: {split_d}")

        # загружаем мета-данные (возраст, пол) из CSV
        meta = self._load_meta(root, split)

        # собираем список файлов по папкам
        for eth_idx, eth_name in enumerate(FAIRFACE_CLASSES):
            eth_dir = split_d / eth_name
            if not eth_dir.exists():
                continue
            for img_path in eth_dir.glob("*.jpg"):
                age, gender = meta.get(img_path.name, (25, 0))
                self.items.append((img_path, age, gender, eth_idx))

        # статистика
        ages   = [a for _,a,_,_ in self.items]
        young  = sum(1 for a in ages if a < 25)
        middle = sum(1 for a in ages if 25 <= a < 50)
        old    = sum(1 for a in ages if a >= 50)
        print(f"  [FairFace/{split}] {len(self.items)} фото  "
              f"(до 25: {young}  25-49: {middle}  50+: {old})")

        # кешируем в RAM — первый запуск долгий, потом летает
        if cache and len(self.items) > 0:
            print(f"  [FairFace/{split}] Кешируем в RAM...", flush=True)
            t0     = time.time()
            errors = 0
            for i, (p, _, _, _) in enumerate(self.items):
                img = cv2.imread(str(p))
                if img is not None:
                    self._cache[str(p)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    errors += 1
                if (i+1) % 5000 == 0 or (i+1) == len(self.items):
                    pct = (i+1) / len(self.items) * 100
                    print(f"  [FairFace/{split}] {i+1}/{len(self.items)} "
                          f"({pct:.0f}%) {time.time()-t0:.0f}с", flush=True)
            print(f"  [FairFace/{split}] Кеш готов: {len(self._cache)} фото"
                  + (f" ({errors} ошибок)" if errors else ""))

    def _load_meta(self, root, split):
        # загружаем возраст и пол из CSV FairFace
        meta = {}
        # FairFace хранит возраст как диапазон "0-2", "10-19" и т.д.
        age_bucket_map = {
            "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25,
            "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65,
            "more than 70": 75
        }
        csv_path = root.parent / f"fairface_label_{split}.csv"
        if not csv_path.exists():
            csv_path = root / f"fairface_label_{split}.csv"
        if not csv_path.exists():
            return meta

        import csv as csv_mod
        with open(csv_path, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                fname  = Path(row.get("file", "")).name
                age    = age_bucket_map.get(row.get("age", ""), 25)
                gender = 0 if "Male" in row.get("gender", "Male") else 1
                meta[fname] = (age, gender)
        return meta

    def get_sample_weights(self):
        # Веса для WeightedRandomSampler.
        # Подростки (до 20) получают 4-5x — их мало в датасете.
        # Без этого модель плохо узнаёт молодых.
        weights = []
        for _, age, _, _ in self.items:
            if age < 15:    weights.append(5.0)
            elif age < 20:  weights.append(4.0)
            elif age < 25:  weights.append(2.5)
            elif age < 40:  weights.append(1.0)
            elif age < 60:  weights.append(1.5)
            else:           weights.append(2.5)
        return weights

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, age, gender, eth = self.items[idx]
        img = self._cache.get(str(path))
        if img is None:
            img = cv2.imread(str(path))
            if img is None:
                return (torch.zeros(3, IMG_SIZE, IMG_SIZE),
                        torch.tensor(25.0), gender, eth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (self.tf(img),
                torch.tensor(float(age), dtype=torch.float32),
                gender, eth)

class UTKDataset(Dataset):
    # UTKFace — резервный если FairFace не скачан.
    # 5 классов, маппируем в 7-классовое пространство FairFace:
    # White→0, Black→1, Asian→3(East Asian), Indian→5, Other→6

    # UTK индекс → FairFace индекс
    UTK_TO_FAIRFACE = {0: 0, 1: 1, 2: 3, 3: 5, 4: 6}

    def __init__(self, root, transform=TRAIN_TF, cache=True):
        self.tf     = transform
        self.items  = []
        self._cache = {}

        root = Path(root)
        for p in list(root.glob("**/*.jpg")) + list(root.glob("**/*.png")):
            parts = p.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                age     = int(parts[0])
                gender  = int(parts[1])
                utk_eth = int(parts[2])
            except Exception:
                continue
            ff_eth = self.UTK_TO_FAIRFACE.get(utk_eth, 6)
            if 1 <= age <= 100 and 0 <= gender <= 1:
                self.items.append((p, age, gender, ff_eth))

        ages  = [a for _,a,_,_ in self.items]
        young = sum(1 for a in ages if a < 25)
        print(f"  UTKFace {len(self.items)} фото (до 25: {young})")
        print("  UTKFace хуже FairFace — скачай FairFace для точности!")

        if cache:
            print("  UTKFace Кешируем в RAM...", flush=True)
            t0 = time.time()
            for i, (p, _, _, _) in enumerate(self.items):
                img = cv2.imread(str(p))
                if img is not None:
                    self._cache[str(p)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (i+1) % 3000 == 0:
                    print(f"  UTKFace {i+1}/{len(self.items)} "
                          f"{time.time()-t0:.0f}с", flush=True)
            print("  UTKFace Кеш готов")

    def get_sample_weights(self):
        weights = []
        for _, age, _, _ in self.items:
            if age < 18:    weights.append(4.0)
            elif age < 25:  weights.append(3.0)
            elif age < 40:  weights.append(1.0)
            elif age < 60:  weights.append(1.5)
            else:           weights.append(2.5)
        return weights

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, age, gender, eth = self.items[idx]
        img = self._cache.get(str(path))
        if img is None:
            img = cv2.imread(str(path))
            if img is None:
                return (torch.zeros(3, IMG_SIZE, IMG_SIZE),
                        torch.tensor(25.0), gender, eth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (self.tf(img),
                torch.tensor(float(age), dtype=torch.float32),
                gender, eth)

class LFWDataset(Dataset):
    # LFW  для обучения эмбеддингов через triplet loss

    def __init__(self, root, transform=TRAIN_TF, min_imgs=2):
        self.root    = Path(root)
        self.tf      = transform
        self.classes = {}
        for d in self.root.iterdir():
            if d.is_dir():
                imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if len(imgs) >= min_imgs:
                    self.classes[d.name] = imgs
        self.names = list(self.classes.keys())
        print(f"  LFW {len(self.names)} личностей с ≥{min_imgs} фото")

    def _load_img(self, path):
        img = cv2.imread(str(path))
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return self.tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __len__(self): return len(self.names) * 30

    def __getitem__(self, idx):
        # triplet: anchor + positive (та же личность) + negative (другая)
        anchor_name = self.names[idx % len(self.names)]
        neg_name    = random.choice([n for n in self.names if n != anchor_name])
        a_path, p_path = random.sample(self.classes[anchor_name], 2)
        n_path          = random.choice(self.classes[neg_name])
        return self._load_img(a_path), self._load_img(p_path), self._load_img(n_path)

class FERDataset(Dataset):
    # FER-2013 — датасет эмоций
    EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

    def __init__(self, root, split="train"):
        root   = Path(root)
        self.tf    = EMO_TRAIN_TF if split == "train" else EMO_VAL_TF
        self.items = []

        # ищем папку с нужным split
        for candidate in [split, split.capitalize()]:
            sd = root / candidate
            if sd.exists():
                for i, emo in enumerate(self.EMOTIONS):
                    ed = sd / emo
                    if ed.exists():
                        for p in list(ed.glob("*.png")) + list(ed.glob("*.jpg")):
                            self.items.append((p, i))
                break

        # если val не нашли — пробуем test
        if not self.items and split == "val":
            for candidate in ["test", "Test"]:
                sd = root / candidate
                if sd.exists():
                    for i, emo in enumerate(self.EMOTIONS):
                        ed = sd / emo
                        if ed.exists():
                            for p in list(ed.glob("*.png")) + list(ed.glob("*.jpg")):
                                self.items.append((p, i))
                    break

        print(f"  [FER/{split}] {len(self.items)} фото")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = cv2.imread(str(path))
        if img is None:
            return torch.zeros(3, 48, 48), label
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), label

#  АРХИТЕКТУРЫ НЕЙРОСЕТЕЙ

class FaceEmbedder(nn.Module):

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        base          = models.efficientnet_b2(weights="IMAGENET1K_V1")
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        x = self.pool(self.backbone(x)).flatten(1)
        return F.normalize(self.head(x), dim=1)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        dp = F.pairwise_distance(anchor, pos)
        dn = F.pairwise_distance(anchor, neg)
        return F.relu(dp - dn + self.margin).mean()

class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        base          = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)).flatten(1))

class AgeGenderEthNet(nn.Module):

    def __init__(self, n_eth=N_ETH):
        super().__init__()
        base          = models.efficientnet_b2(weights="IMAGENET1K_V1")
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # общий слой компрессии признаков
        self.shared = nn.Sequential(
            nn.Linear(1408, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # отдельная ветка для возраста — точнее
        self.age_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.gender_head = nn.Linear(256, 2)
        self.eth_head    = nn.Linear(256, n_eth)

    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        h    = self.shared(feat)
        return {
            "age":       self.age_branch(h).squeeze(1),
            "gender":    self.gender_head(h),
            "ethnicity": self.eth_head(h),
        }

#  ФУНКЦИЯ ПОТЕРЬ ДЛЯ ВОЗРАСТА

class AgeLoss(nn.Module):

    def forward(self, pred, target):
        err = torch.abs(pred - target)
        # веса по возрасту — чем моложе тем больше вес
        w = torch.ones_like(target)
        w[target < 15]  = 5.0
        w[(target >= 15) & (target < 20)] = 4.0
        w[(target >= 20) & (target < 30)] = 2.0
        mae = (w * err).mean()
        mse = (w * err**2).mean()
        return mae + 0.15 * mse

#  1. ЭМБЕДДЕР ЛИЦ

def train_embedder(epochs=30):
    # ищем LFW в стандартных местах
    lfw_path = None
    for candidate in [DATA/"lfw", DATA/"lfw_funneled", DATA/"lfw-deepfunneled"]:
        if candidate.exists() and any(candidate.iterdir()):
            lfw_path = candidate; break

    if lfw_path is None:
        print("LFW не найден — пропускаем")
        return

    print(f"\nОбучаем FaceEmbedder на LFW ({lfw_path})...")

    ds = LFWDataset(lfw_path)
    if len(ds.names) < 2:
        print("Мало личностей"); return

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    model   = FaceEmbedder(EMBED_DIM).to(DEVICE)
    loss_fn = TripletLoss(margin=0.3)
    opt     = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched   = OneCycleLR(opt, max_lr=3e-4, epochs=epochs,
                         steps_per_epoch=len(loader))
    scaler  = GradScaler(enabled=USE_AMP)

    best = float("inf")
    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for i, (a, p, n) in enumerate(loader, 1):
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=USE_AMP):
                ea, ep_, en = model(a), model(p), model(n)
                loss = loss_fn(ea, ep_, en)
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer=opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            total += loss.item()
            if i % 30 == 0 or i == len(loader):
                print(f"  ep{ep}/{epochs} шаг{i}/{len(loader)} "
                      f"loss={total/i:.4f}", end="\r", flush=True)

        avg = total / len(loader)
        print(f"Эпоха {ep}/{epochs}  avg_loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), MODELS/"embedder_best.pt")

    torch.save(model.state_dict(), MODELS/"embedder.pt")
    print(f"FaceEmbedder сохранён  best_loss={best:.4f}")
    return model

#  2. ЭМОЦИИ

def train_emotion(epochs=40):
    fer_path = DATA / "fer2013"
    if not fer_path.exists():
        print("FER-2013 не найден — пропускаем"); return

    print(f"\nОбучаем EmotionNet на FER-2013...")

    train_ds = FERDataset(fer_path, "train")
    val_ds   = FERDataset(fer_path, "val")

    if len(train_ds) == 0:
        print("Тренировочный набор пуст"); return
    if len(val_ds) == 0:
        n_val = max(1, int(0.1 * len(train_ds)))
        train_ds, val_ds = random_split(train_ds, [len(train_ds)-n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    model  = EmotionNet().to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched  = OneCycleLR(opt, max_lr=3e-4, epochs=epochs,
                        steps_per_epoch=len(train_dl))
    scaler = GradScaler(enabled=USE_AMP)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing предотвращает overconfidence

    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=USE_AMP):
                loss = crit(model(imgs), labels)
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer=opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

        # валидация
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast(enabled=USE_AMP):
                    preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        acc = correct / total
        print(f"  Эпоха {ep}/{epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODELS/"emotion_best.pt")

    torch.save(model.state_dict(), MODELS/"emotion.pt")
    print(f"EmotionNet сохранён  best_acc={best_acc:.3f}")
    return model

#  3. ВОЗРАСТ / ПОЛ / ЭТНИЧНОСТЬ

def train_age_gender_eth(epochs=80, resume=False):

    CKPT     = MODELS / "age_gen_eth_ckpt.pt"
    CKPT_MID = MODELS / "age_gen_eth_ckpt_mid.pt"

    # выбираем датасет
    ff_path  = DATA / "fairface"
    utk_path = DATA / "utkface"

    using_fairface = False
    if ff_path.exists() and any(ff_path.rglob("*.jpg")):
        print(f"\nИспользуем FairFace (7 классов этничности)")
        train_ds = FairFaceDataset(ff_path, split="train",
                                   transform=TRAIN_TF, cache=True)
        val_ds   = FairFaceDataset(ff_path, split="val",
                                   transform=VAL_TF,   cache=True)
        using_fairface = True
    elif utk_path.exists() and any(utk_path.rglob("*.jpg")):
        print(f"\n[PT] ⚠ FairFace не найден — используем UTKFace")
        full_ds = UTKDataset(utk_path, transform=TRAIN_TF, cache=True)
        if len(full_ds) == 0:
            print("[age] Нет данных — пропускаем"); return
        n_val   = max(1, int(0.1 * len(full_ds)))
        n_train = len(full_ds) - n_val
        rng = torch.Generator(); rng.manual_seed(42)
        train_ds, raw_val = random_split(full_ds, [n_train, n_val], generator=rng)
        val_ds_obj    = copy.copy(full_ds)
        val_ds_obj.tf = VAL_TF
        val_ds        = Subset(val_ds_obj, raw_val.indices)
    else:
        print("Нет датасетов. Запусти: python datasets.py")
        return

    if using_fairface:
        weights = train_ds.get_sample_weights()
    else:
        all_w   = full_ds.get_sample_weights()
        weights = [all_w[i] for i in train_ds.indices]

    sampler  = WeightedRandomSampler(weights, num_samples=len(weights),
                                     replacement=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                          num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0)

    print(f"\nОбучаем AgeGenderEthNet ({epochs} эпох)...")
    print(f"     Датасет     : {'FairFace ★' if using_fairface else 'UTKFace'}")
    print(f"     Классов эт. : {N_ETH}")
    print(f"     Backbone    : EfficientNet-B2")
    print(f"     FP16 AMP    : {'ВКЛ' if USE_AMP else 'ОТКЛ'}")
    print(f"     Батч        : {BATCH}")

    model    = AgeGenderEthNet(n_eth=N_ETH).to(DEVICE)
    opt      = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched    = OneCycleLR(opt, max_lr=2e-4, epochs=epochs,
                          steps_per_epoch=len(train_dl))
    scaler   = GradScaler(enabled=USE_AMP)
    age_loss = AgeLoss()
    ce       = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_ep = 1
    best_mae = float("inf")

    # восстановление из чекпоинта
    if resume:
        ckpt_to_load = CKPT if CKPT.exists() else (CKPT_MID if CKPT_MID.exists() else None)
        if ckpt_to_load:
            try:
                ckpt = torch.load(str(ckpt_to_load), map_location=DEVICE,
                                  weights_only=False)
                model.load_state_dict(ckpt["model"])
                opt.load_state_dict(ckpt["opt"])
                sched.load_state_dict(ckpt["sched"])
                start_ep = ckpt["epoch"] + (0 if ckpt_to_load == CKPT_MID else 1)
                best_mae = ckpt.get("best_mae", float("inf"))
                print(f"  Продолжаем с эпохи {start_ep}  "
                      f"best_mae={best_mae:.1f}лет")
            except Exception as e:
                print(f"  Чекпоинт не загрузился ({e}) — стартуем заново")

    if start_ep > epochs:
        print(f"  Уже обучено {epochs} эпох.")
        return model

    # главный цикл обучения
    for ep in range(start_ep, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_steps = len(train_dl)

        for step, (imgs, ages, genders, eths) in enumerate(train_dl, 1):
            imgs    = imgs.to(DEVICE)
            ages    = ages.to(DEVICE)
            genders = genders.to(DEVICE)
            eths    = eths.to(DEVICE)

            opt.zero_grad()
            with autocast(enabled=USE_AMP):
                out  = model(imgs)
                loss = (age_loss(out["age"], ages) +
                        2.0 * ce(out["gender"],    genders) +
                        2.0 * ce(out["ethnicity"], eths))
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer=opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            ep_loss += loss.item()

            if step % 50 == 0 or step == n_steps:
                print(f"  ep{ep}/{epochs}  шаг{step}/{n_steps}  "
                      f"loss={ep_loss/step:.4f}", end="\r", flush=True)

            # чекпоинт — каждые 100 шагов можно прерваться
            if step % 100 == 0:
                torch.save({
                    "epoch":     ep,
                    "step":      step,
                    "mid_epoch": True,
                    "model":     model.state_dict(),
                    "opt":       opt.state_dict(),
                    "sched":     sched.state_dict(),
                    "best_mae":  best_mae,
                }, CKPT_MID)

        # валидация
        model.eval()
        mae = gen_acc = eth_acc = 0.0
        n = 0
        with torch.no_grad():
            for imgs, ages, genders, eths in val_dl:
                imgs    = imgs.to(DEVICE)
                ages    = ages.to(DEVICE)
                genders = genders.to(DEVICE)
                eths    = eths.to(DEVICE)
                with autocast(enabled=USE_AMP):
                    out = model(imgs)
                pred_age = out["age"].clamp(1, 100)
                mae     += torch.abs(pred_age - ages).mean().item()
                gen_acc += (out["gender"].argmax(1) == genders).float().mean().item()
                eth_acc += (out["ethnicity"].argmax(1) == eths).float().mean().item()
                n += 1

        mae /= n; gen_acc /= n; eth_acc /= n
        print(f"  Эпоха {ep}/{epochs}  "
              f"age_mae={mae:.1f}лет  gen={gen_acc:.3f}  eth={eth_acc:.3f}")

        # сохраняем лучшие веса
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), MODELS/"age_gen_eth_best.pt")

        # конец эпохи — полный чекпоинт
        torch.save({
            "epoch":    ep,
            "model":    model.state_dict(),
            "opt":      opt.state_dict(),
            "sched":    sched.state_dict(),
            "best_mae": best_mae,
        }, CKPT)
        # удаляем mid-чекпоинт — эпоха завершена
        if CKPT_MID.exists():
            CKPT_MID.unlink()

        # обновляем рабочую модель — камера может тестировать во время обучения
        torch.save(model.state_dict(), MODELS/"age_gen_eth.pt")

    print(f"AgeGenderEthNet сохранён  best_mae={best_mae:.1f}лет")
    return model

#  МАНИФЕСТ

def save_manifest():
    manifest = {
        "version":     "3",
        "framework":   "pytorch",
        "device":      str(DEVICE),
        "img_size":    IMG_SIZE,
        "embed_dim":   EMBED_DIM,
        "n_eth":       N_ETH,
        "eth_classes": FAIRFACE_CLASSES,
        "models": {
            "embedder":    str(MODELS/"embedder.pt"),
            "emotion":     str(MODELS/"emotion.pt"),
            "age_gen_eth": str(MODELS/"age_gen_eth.pt"),
        },
        "classes": {
            "emotions":  ["angry","disgust","fear","happy","sad","surprise","neutral"],
            "genders":   ["male","female"],
            "ethnicity": FAIRFACE_CLASSES,
        }
    }
    with open(MODELS/"manifest.json","w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[PT] Манифест сохранён → {MODELS/'manifest.json'}")

#  MAIN

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Face AI — обучение моделей v3")
    p.add_argument("--all",     action="store_true", help="Все модели")
    p.add_argument("--embed",   action="store_true", help="Только эмбеддер")
    p.add_argument("--emotion", action="store_true", help="Только эмоции")
    p.add_argument("--age",     action="store_true", help="Возраст/пол/этничность")
    p.add_argument("--epochs",  type=int, default=80)
    p.add_argument("--resume",  action="store_true",
                   help="Продолжить с последнего чекпоинта")
    args = p.parse_args()

    print("  Face AI — Обучение v3 (FairFace + FP16)")
    print(f"  Эпох        : {args.epochs}")
    print(f"  Размер фото : {IMG_SIZE}px")
    print(f"  Эмбеддинг   : {EMBED_DIM}d")
    print(f"  Классов эт. : {N_ETH} (FairFace)")
    print(f"  Backbone    : EfficientNet-B2")
    print(f"  FP16 AMP    : {'ВКЛ' if USE_AMP else 'ОТКЛ'}")
    print(f"  Батч        : {BATCH}")
    print()

    if args.all or args.embed:
        train_embedder(epochs=args.epochs)
    if args.all or args.emotion:
        train_emotion(epochs=args.epochs)
    if args.all or args.age:
        train_age_gender_eth(epochs=args.epochs, resume=args.resume)

    save_manifest()
    print("\nГотово. Следующий шаг: python compare_models.py")


