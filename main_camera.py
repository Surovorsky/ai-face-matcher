

import os, sys, cv2, time, json, uuid, threading, warnings
import numpy as np
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# настройки

CAMERA_ID        = 0
FRAME_W          = 1280
FRAME_H          = 720
FACE_SIZE        = 96
FACE_SIZE_EMO    = 48
FRAMES_FOR_EMB   = 5       # кадров до первого анализа
DIST_THRESHOLD   = 0.25
EMOTION_INTERVAL = 0.35
MAX_WORKERS      = 3
DB_PATH          = "face_db_v2.json"
SCREENSHOT_DIR   = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


# максимальное смещение центра лица чтобы трек считался тем же
TRACK_RADIUS     = 120
# сколько кадров трек живёт без обнаружения (убирает мерцание)
TRACK_PATIENCE   = 12
# минимальный размер лица в пикселях
MIN_FACE_PX      = 60


YOLO_CONF        = 0.65    # повышено чтобы меньше ложных срабатываний от одежды
MIN_SKIN_RATIO   = 0.14    # повышено — шея и плечи содержат меньше кожных пикселей
MIN_FACE_PX      = 80      # повышено — игнорируем мелкие детекции (шея обычно узкая)
MAX_H_W_RATIO    = 1.6

EMOTIONS  = ["angry","disgust","fear","happy","sad","surprise","neutral"]
GENDERS   = ["male","female"]
ETHNICITY = ["White","Black","Asian","Indian","Other"]

COL_KNOWN   = (50,  220, 100)
COL_NEW     = (50,  180, 255)
COL_COLLECT = (160, 160, 160)
COL_PROC    = (120, 80,  255)

# хаб моделей

class ModelHub:
    def __init__(self):
        self._tf_ok = self._pt_ok = False
        self._device = None
        self.tf = {}
        self.pt = {}
        self._setup_tf()
        self._setup_pt()
        self.rec = self._load_rec()

    def _setup_tf(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
            self._keras = keras
            self._tf_ok = True
        except Exception:
            pass

    def _setup_pt(self):
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                print(f"  Видеокарта: CUDA — {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
                print("  Видеокарта: Apple MPS")
            else:
                self._device = torch.device("cpu")
                if "+cpu" in torch.__version__:
                    print("  ВНИМАНИЕ: CPU-версия PyTorch — видеокарта не используется")
                    print("  pip uninstall torch torchvision -y")
                    print("     pip install torch torchvision "
                          "--index-url https://download.pytorch.org/whl/cu121")
            self._pt_ok = True
        except Exception:
            pass

    def _load_rec(self):
        p = Path("models/comparison.json")
        if p.exists():
            with open(p) as f:
                return json.load(f).get("recommended", {})
        return {"embedder": "pytorch", "emotion": "pytorch", "age_gen_eth": "pytorch"}

    def _pref(self, key):
        return self.rec.get(key, "pytorch")

    def _keras_load(self, path):
        if not self._tf_ok or not Path(path).exists():
            return None
        try:
            return self._keras.models.load_model(path)
        except Exception as e:
            print(f"  TF: не удалось загрузить {Path(path).name}: {e}")
            return None

    def _torch_load(self, cls, path, **kw):
        if not self._pt_ok or not Path(path).exists():
            return None
        try:
            m = cls(**kw).to(self._device)
            for wo in [True, False]:
                try:
                    st = self._torch.load(path, map_location=self._device, weights_only=wo)
                    m.load_state_dict(st)
                    m.eval()
                    return m
                except Exception:
                    continue
        except Exception as e:
            print(f"  PT: не удалось загрузить {Path(path).name}: {e}")
        return None

    def load_embedder(self):
        if self._pref("embedder") != "tensorflow" and self._pt_ok:
            from train_pt import FaceEmbedder
            m = self._torch_load(FaceEmbedder, "models/pt/embedder.pt")
            if m:
                self.pt["embedder"] = m
                print("  Эмбеддер: PyTorch")
                return "pt"
        m = self._keras_load("models/tf/embedder.keras")
        if m:
            self.tf["embedder"] = m
            print("  Эмбеддер: TensorFlow")
            return "tf"
        print("  Эмбеддер: резервный HOG")
        return "hog"

    def load_emotion(self):
        if self._pref("emotion") != "tensorflow" and self._pt_ok:
            from train_pt import EmotionNet
            m = self._torch_load(EmotionNet, "models/pt/emotion.pt")
            if m:
                self.pt["emotion"] = m
                print("  Эмоции: PyTorch")
                return "pt"
        m = self._keras_load("models/tf/emotion.keras")
        if m:
            self.tf["emotion"] = m
            print("  Эмоции: TensorFlow")
            return "tf"
        print("  Эмоции: резервный Haar")
        return "haar"

    def load_age(self):
        if self._pref("age_gen_eth") != "tensorflow" and self._pt_ok:
            from train_pt import AgeGenderEthNet
            m = self._torch_load(AgeGenderEthNet, "models/pt/age_gen_eth.pt", n_eth=5)
            if m:
                self.pt["age_gen_eth"] = m
                print("  Возраст/пол/этничность: PyTorch")
                return "pt"
        m = self._keras_load("models/tf/age_gender_eth.keras")
        if m:
            self.tf["age_gen_eth"] = m
            print("  Возраст/пол/этничность: TensorFlow")
            return "tf"
        print("  Возраст/пол/этничность: эвристика")
        return "heuristic"

    def load_yolo(self):
        try:
            from ultralytics import YOLO
            face_path = Path("models/pt/yolo_face.pt")
            if face_path.exists():
                model = YOLO(str(face_path))
                print("  Детектор: YOLOv8-face (локальный)")
                return model
            import urllib.request, ssl
            url = ("https://github.com/akanametov/yolo-face/releases/download/"
                   "v0.0.0/yolov8n-face.pt")
            face_path.parent.mkdir(parents=True, exist_ok=True)
            print("  Детектор: скачиваем YOLOv8n-face (~6 МБ)...")
            try:
                urllib.request.urlretrieve(url, str(face_path))
                model = YOLO(str(face_path))
                print("  Детектор: YOLOv8n-face")
                return model
            except Exception:
                pass
            print("  Детектор: обычный YOLOv8n (дополнительные фильтры включены)")
            return YOLO("yolov8n.pt")
        except Exception as e:
            print(f"  Детектор: резервный Haar ({e})")
            return None

# вспомогательные функции детекции

CASCADE       = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def _skin_ratio(bgr_crop):
    # доля пикселей похожих на кожу в цветовом пространстве YCrCb
    if bgr_crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:,:,1].astype(int)
    cb = ycrcb[:,:,2].astype(int)
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return mask.sum() / max(mask.size, 1)

def _has_skin(bgr_crop):
    # True если в кропе достаточно пикселей цвета кожи
    return _skin_ratio(bgr_crop) >= MIN_SKIN_RATIO

def detect(frame, yolo_model):
    # обнаружение лиц
    # для обычных моделей дополнительно проверяет кожу и пропорции
    # если YOLO недоступен — переходит на каскад Хаара
    raw_boxes = []

    if yolo_model:
        try:
            res = yolo_model(frame, verbose=False, conf=YOLO_CONF)
            is_face_model = len(yolo_model.names) <= 4  # face models have 1-2 classes
            for r in res:
                for i, b in enumerate(r.boxes.xyxy.cpu().numpy()):
                    # обычная COCO модель — берём только класс 0 (человек)
                    if not is_face_model and r.boxes.cls is not None:
                        if int(r.boxes.cls[i].cpu().item()) != 0:
                            continue
                    x1,y1,x2,y2 = map(int, b[:4])
                    raw_boxes.append((x1, y1, x2-x1, y2-y1))
        except Exception:
            pass

    if not raw_boxes:
        # запасной вариант
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        raw_boxes = list(dets) if len(dets) > 0 else []

    clean    = []
    h_frame, w_frame = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, bw, bh) in raw_boxes:
        x  = max(0, x);  y  = max(0, y)
        bw = min(bw, w_frame - x);  bh = min(bh, h_frame - y)

        # слишком маленький — отбрасывает шеи и далёкие пятна
        if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
            continue

        # шея выше чем шире — отбрасываем если высота сильно больше ширины
        ratio = bw / max(bh, 1)
        if not (0.45 < ratio < MAX_H_W_RATIO):
            continue

        # минимальная площадь — отбрасывает тонкие полоски
        if bw * bh < MIN_FACE_PX * MIN_FACE_PX:
            continue

        # детекция ниже 85% высоты кадра — почти всегда шея
        if y > h_frame * 0.85:
            continue

        crop = frame[y:y+bh, x:x+bw]
        if crop.size == 0:
            continue

        # проверка цвета кожи — должна превышать порог
        if not _has_skin(crop):
            continue

        # проверка шеи: верхняя половина лица содержит БОЛЬШЕ кожи
        # (лоб + щёки против подбородка + шея). Если наоборот — это шея
        top_skin = _skin_ratio(crop[:bh//2, :])
        bot_skin = _skin_ratio(crop[bh//2:, :])
        if bh > 80 and top_skin < bot_skin * 0.55:
            continue


        crop_gray = gray_frame[y:y+bh, x:x+bw]
        haar_hits = CASCADE.detectMultiScale(
            crop_gray, scaleFactor=1.1, minNeighbors=2,
            minSize=(int(bw*0.3), int(bh*0.3))
        )
        if len(haar_hits) == 0:
            continue

        clean.append((x, y, bw, bh))

    return clean

# инференс — запуск нейросетей

def _pt_tensor(bgr, size, hub):
    import torch
    from torchvision.transforms.functional import normalize, to_tensor
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
    return normalize(to_tensor(rgb), [0.5]*3, [0.5]*3).unsqueeze(0).to(hub._device)

def run_embedding(bgr, hub, mode):
    if mode == "pt":
        import torch
        with torch.no_grad():
            return hub.pt["embedder"](_pt_tensor(bgr, FACE_SIZE, hub)).cpu().numpy()[0].astype(np.float32)
    if mode == "tf":
        rgb = cv2.cvtColor(cv2.resize(bgr, (FACE_SIZE, FACE_SIZE)), cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 127.5 - 1.0
        return hub.tf["embedder"].predict(inp[np.newaxis], verbose=0)[0].astype(np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g    = cv2.resize(gray, (64, 64))
    gx   = cv2.Sobel(g, cv2.CV_32F, 1, 0);  gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag  = np.sqrt(gx**2 + gy**2);  ang = np.arctan2(gy, gx)*(180/np.pi)%180
    feats = [np.histogram(ang[r:r+8,c:c+8],9,(0,180),weights=mag[r:r+8,c:c+8])[0]
             for r in range(0,57,8) for c in range(0,57,8)]
    v = np.concatenate([f/(np.linalg.norm(f)+1e-6) for f in feats])
    return (v/(np.linalg.norm(v)+1e-6)).astype(np.float32)

def run_emotion(bgr, hub, mode):
    if mode == "pt":
        import torch
        with torch.no_grad():
            out = hub.pt["emotion"](_pt_tensor(bgr, FACE_SIZE_EMO, hub)).cpu().numpy()[0]
        return EMOTIONS[out.argmax()]
    if mode == "tf":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g    = cv2.resize(gray, (FACE_SIZE_EMO, FACE_SIZE_EMO))
        inp  = np.stack([g]*3, -1).astype(np.float32)/127.5-1.0
        return EMOTIONS[hub.tf["emotion"].predict(inp[np.newaxis],verbose=0)[0].argmax()]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return "happy" if len(SMILE_CASCADE.detectMultiScale(gray,1.7,20))>0 else "neutral"

def run_age_gen_eth(bgr, hub, mode):
    if mode == "pt":
        import torch
        from torchvision.transforms.functional import normalize, to_tensor
        rgb = cv2.cvtColor(cv2.resize(bgr,(FACE_SIZE,FACE_SIZE)),cv2.COLOR_BGR2RGB)

        t   = normalize(to_tensor(rgb),
                        [0.485,0.456,0.406],[0.229,0.224,0.225]).unsqueeze(0).to(hub._device)
        with torch.no_grad():
            out = hub.pt["age_gen_eth"](t)
        # возраст напрямую 1-100, не делится на 100
        age = max(1, min(100, int(out["age"].cpu().item())))
        gen = GENDERS[int(out["gender"].cpu()[0].argmax())]
        eth = ETHNICITY[int(out["ethnicity"].cpu()[0].argmax()) % len(ETHNICITY)]
        return age, gen, eth
    if mode == "tf":
        rgb = cv2.cvtColor(cv2.resize(bgr,(FACE_SIZE,FACE_SIZE)),cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32)/127.5-1.0
        out = hub.tf["age_gen_eth"].predict(inp[np.newaxis],verbose=0)
        age = max(1, min(100, int(float(out["age"][0][0])*100)))
        gen = GENDERS[int(out["gender"][0].argmax())]
        eth = ETHNICITY[int(out["ethnicity"][0].argmax()) % len(ETHNICITY)]
        return age, gen, eth
    # если нет модели
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
    age   = 10 if lap<80 else 20 if lap<200 else 33 if lap<600 else 50 if lap<1200 else 68
    h     = gray.shape[0]
    gen   = "male" if np.std(gray[2*h//3:])>np.std(gray[:h//3])*1.05 else "female"
    cr    = cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)[:,:,1].mean()
    eth   = "White" if cr>155 else "Asian" if cr>145 else "Indian" if cr>135 else "Black"
    return age, gen, eth

# база данных

def load_db():
    if Path(DB_PATH).exists():
        with open(DB_PATH) as f:
            return json.load(f).get("faces", [])
    return []

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump({"faces": db}, f, indent=2)

def cosine_dist(a, b):
    return 1 - np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

def sharpness(bgr):
    return cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()

# глобальное состояние

HUB      = None
EMB_MODE = EMO_MODE = AGE_MODE = "heuristic"
DB       = []
db_lock  = threading.Lock()
faces    = {}
face_ctr = 0

# обработчики лиц

def emotion_worker(fid):
    last = 0
    while fid in faces:
        now = time.time()
        if now - last < EMOTION_INTERVAL:
            time.sleep(0.04)
            continue
        crop = faces[fid].get("last_crop")
        if crop is not None:
            try:
                faces[fid]["emotion"] = run_emotion(crop, HUB, EMO_MODE)
            except Exception:
                pass
        last = now

def process_face(fid):
    f = faces.get(fid)
    if not f:
        return

    frames = list(f["frames"])
    if not frames:
        f["status"] = "COLLECT"
        return

    best = max(frames, key=sharpness)

    try:
        emb = run_embedding(best, HUB, EMB_MODE)
    except Exception as e:
        print(f"Ошибка эмбеддера: {e}")
        f["status"] = "COLLECT"
        return

    with db_lock:
        best_dist  = float("inf")
        best_match = None
        for person in DB:
            stored = np.array(person["embedding"], dtype=np.float32)
            if len(stored) != len(emb):
                continue
            d = cosine_dist(emb, stored)
            if d < best_dist:
                best_dist  = d
                best_match = person

        if best_match and best_dist < DIST_THRESHOLD:
            f["info"]   = best_match["meta"]
            f["name"]   = best_match.get("name")
            f["status"] = "KNOWN"
            f["dist"]   = round(float(best_dist), 3)
            return

        # новое лицо — запускаем классификаторы
        try:
            age, gen, eth = run_age_gen_eth(best, HUB, AGE_MODE)
        except Exception:
            age, gen, eth = "?", "?", "?"




        f["gender_votes"].append(gen)
        f["age_votes"].append(age)

        gvotes = list(f["gender_votes"])
        gen    = max(set(gvotes), key=gvotes.count)

        age_nums = [v for v in f["age_votes"] if isinstance(v, int)]
        if age_nums:
            age_nums.sort()
            age = age_nums[len(age_nums)//2]

        meta  = {"age": age, "gender": gen, "ethnicity": eth}
        entry = {
            "id":         str(uuid.uuid4()),
            "embedding":  emb.tolist(),
            "meta":       meta,
            "name":       None,
            "seen":       1,
            "first_seen": time.strftime("%Y-%m-%d %H:%M"),
            "source":     "camera",
        }
        DB.append(entry)
        save_db(DB)

    f["info"]   = meta
    f["name"]   = None
    f["status"] = "NEW"

# отрисовка

def rounded_rect(img, x1, y1, x2, y2, color, t=2, r=10):
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,t); cv2.line(img,(x1+r,y2),(x2-r,y2),color,t)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,t); cv2.line(img,(x2,y1+r),(x2,y2-r),color,t)
    for ang,cx,cy in [(180,x1+r,y1+r),(270,x2-r,y1+r),(90,x1+r,y2-r),(0,x2-r,y2-r)]:
        cv2.ellipse(img,(cx,cy),(r,r),ang,0,90,color,t)

def label(img, text, x, y, color, scale=0.42):
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
    cv2.rectangle(img,(x-2,y-th-3),(x+tw+2,y+3),(0,0,0),-1)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1)

def draw_face(frame, fid, x, y, bw, bh):
    f      = faces[fid]
    status = f["status"]
    info   = f.get("info") or {}
    emo    = f.get("emotion","")
    name   = f.get("name")
    dist   = f.get("dist")

    color = (COL_KNOWN  if status=="KNOWN"      else
             COL_NEW    if status=="NEW"         else
             COL_PROC   if status=="PROCESSING"  else COL_COLLECT)

    rounded_rect(frame, x, y, x+bw, y+bh, color)

    if status == "KNOWN":
        row1 = name or f"Известен  {dist}"
    elif status == "NEW":
        row1 = "Новое лицо — сохранено"
    elif status == "PROCESSING":
        row1 = "Анализируем..."
    else:
        row1 = f"Собираем кадры {len(f['frames'])}/{FRAMES_FOR_EMB}"

    row2 = (f"{info.get('age','?')}yr  {info.get('gender','?')}  {info.get('ethnicity','?')}"
            if info else "")

    if row2:
        label(frame, row2, x, y-6,  (180,180,180))
        label(frame, row1, x, y-22, color)
    else:
        label(frame, row1, x, y-8,  color)

    if emo:
        label(frame, emo, x, y+bh+16, (180,180,255))

def draw_hud(frame, n_active, fps):
    w = frame.shape[1]
    cv2.rectangle(frame,(0,0),(w,26),(0,0,0),-1)
    txt = (f"FaceAI  |  DB:{len(DB)}  Tracking:{n_active}  {fps:.0f}fps  |  "
           f"Embed:{EMB_MODE}  Emo:{EMO_MODE}  Age:{AGE_MODE}  |  "
           f"ESC=выход  S=скриншот  D=база  R=сброс")
    cv2.putText(frame,txt,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.37,(200,200,200),1)

# отслеживание лиц

def match_track(cx, cy):
    best_fid  = None
    best_dist = float("inf")
    for fid, f in faces.items():
        fx, fy = f["center"]
        d = np.sqrt((fx-cx)**2 + (fy-cy)**2)
        if d < TRACK_RADIUS and d < best_dist:
            best_dist = d
            best_fid  = fid
    return best_fid

def new_track(cx, cy):
    global face_ctr
    fid = face_ctr
    face_ctr += 1
    faces[fid] = {
        "center":       (cx, cy),
        "frames":       deque(maxlen=FRAMES_FOR_EMB),
        "status":       "COLLECT",
        "info":         None,
        "name":         None,
        "emotion":      "",
        "dist":         None,
        "last_crop":    None,
        "missing":      0,           # кадров с последнего обнаружения
        "box":          (0,0,1,1),   # последняя известная рамка для рисования призрака
        "gender_votes": deque(maxlen=8),
        "age_votes":    deque(maxlen=8),
    }
    threading.Thread(target=emotion_worker, args=(fid,), daemon=True).start()
    return fid

# запуск

def main():
    global faces, face_ctr, DB, HUB, EMB_MODE, EMO_MODE, AGE_MODE

    print("=" * 56)
    print("  Face AI — Камера")
    print("=" * 56)
    print("Шаг 1/4: загружаем модели...")
    HUB      = ModelHub()
    EMB_MODE = HUB.load_embedder()
    EMO_MODE = HUB.load_emotion()
    AGE_MODE = HUB.load_age()
    yolo     = HUB.load_yolo()

    print("Шаг 2/4: загружаем базу лиц...")
    DB = load_db()
    print(f"  {len(DB)} лиц в базе")

    print("Шаг 3/4: открываем камеру...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW if sys.platform=="win32" else 0)
    if not cap.isOpened():
        print(f"ОШИБКА: не удаётся открыть камеру {CAMERA_ID}")
        print("  Закрой OBS и другие программы использующие камеру")
        print("  Попробуй CAMERA_ID = 1 или 2 в начале файла")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Шаг 4/4: запускаем — нажми ESC для выхода\n")

    executor   = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    fps_time   = time.time()
    fps_frames = 0
    fps        = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        fps_frames += 1
        if fps_frames >= 20:
            fps        = fps_frames / (time.time()-fps_time)
            fps_frames = 0
            fps_time   = time.time()

        detections = detect(frame, yolo)
        seen_fids  = set()

        for (x, y, bw, bh) in detections:
            cx, cy = x + bw//2, y + bh//2

            fid = match_track(cx, cy)
            if fid is None:
                fid = new_track(cx, cy)

            f = faces[fid]
            f["center"]  = (cx, cy)
            f["box"]     = (x, y, bw, bh)
            f["missing"] = 0
            seen_fids.add(fid)

            crop = frame[y:y+bh, x:x+bw]
            if crop.size == 0:
                continue
            crop_r = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
            f["frames"].append(crop_r)
            f["last_crop"] = crop_r

            if len(f["frames"]) >= FRAMES_FOR_EMB and f["status"] == "COLLECT":
                f["status"] = "PROCESSING"
                executor.submit(process_face, fid)

            draw_face(frame, fid, x, y, bw, bh)

        for fid in list(faces):
            if fid not in seen_fids:
                faces[fid]["missing"] += 1
                if faces[fid]["missing"] > TRACK_PATIENCE:
                    del faces[fid]
                else:
                    # рисуем рамку с затухающим цветом пока трек держится
                    x, y, bw, bh = faces[fid]["box"]
                    alpha = 1.0 - faces[fid]["missing"] / TRACK_PATIENCE
                    ghost_color = tuple(int(c * alpha) for c in COL_COLLECT)
                    rounded_rect(frame, x, y, x+bw, y+bh, ghost_color, t=1)

        draw_hud(frame, len(seen_fids), fps)
        cv2.imshow("Face AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            p = SCREENSHOT_DIR / f"cap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"Скриншот сохранён: {p}")
        elif key == ord("d"):
            print(f"\nБаза данных: {len(DB)} лиц:")
            for p in DB:
                n = p.get("name") or "(без имени)"
                print(f"  {p['id'][:8]}  {n}  {p['meta']}")
        elif key == ord("r"):
            faces.clear()
            face_ctr = 0
            print("Треки сброшены")

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    print("Программа завершена")

if __name__ == "__main__":
    main()


