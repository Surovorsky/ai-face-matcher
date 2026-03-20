# compare_models.py

import time, json, os, sys, warnings
from pathlib import Path
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

MODELS_TF = Path("models/tf")
MODELS_PT = Path("models/pt")

N_WARMUP = 3
N_BENCH  = 30

TASK_NAMES = ["embedder", "emotion", "age_gen_eth"]

def make_rgb(size=96):
    return np.random.randint(80, 200, (size, size, 3), dtype=np.uint8)

def make_gray_as_rgb(size=48):
    g = np.random.randint(80, 200, (size, size), dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)

# тест TensorFlow

def bench_tf():
    has_any = any((MODELS_TF / f"{n}.keras").exists() for n in TASK_NAMES)
    if not has_any:
        print("[TF] Обученные TF модели не найдены — пропускаем тест TF.")
        print("     (Запусти python train_tf.py --all если нужно сравнение с TF)")
        return None

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("[TF] TensorFlow не установлен — пропускаем.")
        return None

    import platform
    if platform.system() == "Windows":
        print("[TF] ВНИМАНИЕ: TensorFlow 2.11+ не использует GPU на Windows.")
        print("     Тест TF будет на CPU. PyTorch использует RTX 4050.")
        print("     PyTorch будет быстрее — это ожидаемо.\n")

    results = {"framework": "TensorFlow", "models": {}}

    for name in TASK_NAMES:
        path = MODELS_TF / f"{name}.keras"
        if not path.exists():
            print(f"  [TF] {name} не найдена — пропускаем")
            continue

        print(f"  [TF] Тестируем {name}...", end="", flush=True)
        try:
            model = keras.models.load_model(str(path))
        except Exception as e:
            print(f" Ошибка загрузки: {e}")
            continue

        try:
            inp = _make_tf_input(model)
        except Exception as e:
            print(f" Ошибка создания тестового входа: {e}")
            continue

        try:
            for _ in range(N_WARMUP):
                model.predict(inp, verbose=0)
        except Exception as e:
            print(f" Ошибка прогрева: {e}")
            continue

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            model.predict(inp, verbose=0)
        ms = (time.perf_counter() - t0) / N_BENCH * 1000

        results["models"][name] = {"ms_per_image": round(ms, 2)}
        print(f" {ms:.1f} ms")

    return results if results["models"] else None

def _make_tf_input(model):
    # определяем форму входа модели и создаём тестовый тензор нужного размера
    try:
        expected = model.input_shape
        h = expected[1] or 96
        w = expected[2] or 96
        c = expected[3] or 3
    except Exception:
        h, w, c = 96, 96, 3

    if c == 1:
        g = np.random.randint(80, 200, (h, w), dtype=np.uint8)
        return (g.astype(np.float32) / 127.5 - 1.0)[np.newaxis, :, :, np.newaxis]
    else:
        img = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)
        return (img.astype(np.float32) / 127.5 - 1.0)[np.newaxis]

# тест PyTorch

def bench_pt():
    has_any = any((MODELS_PT / f"{n}.pt").exists() for n in TASK_NAMES)
    if not has_any:
        print("[PT] Обученные PT модели не найдены.")
        print("     Запусти: python train_pt.py --all --epochs 30")
        return None

    try:
        import torch
        from torchvision import transforms
    except ImportError:
        print("[PT] PyTorch не установлен — пропускаем.")
        return None

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"  [PT] Устройство: {device}" + (f" ({gpu_name})" if gpu_name else ""))

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train_pt import FaceEmbedder, EmotionNet, AgeGenderEthNet
    except ImportError as e:
        print(f"  [PT] Не удалось импортировать классы моделей из train_pt.py: {e}")
        return None

    MODEL_CLASSES = {
        "embedder":    (FaceEmbedder,    {}),
        "emotion":     (EmotionNet,      {}),
        "age_gen_eth": (AgeGenderEthNet, {"n_eth": 7}),   # старые модели использовали UTKFace с 5 классами
    }

    # нормализация ImageNet для EfficientNet
    norm_imagenet = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    norm_emotion  = transforms.Normalize([0.5]*3, [0.5]*3)
    to_tensor     = transforms.ToTensor()

    results = {"framework": "PyTorch", "device": str(device), "models": {}}

    for name, (Cls, kwargs) in MODEL_CLASSES.items():
        path = MODELS_PT / f"{name}.pt"
        if not path.exists():
            print(f"  [PT] {name} не найдена — пропускаем")
            continue

        print(f"  [PT] Тестируем {name}...", end="", flush=True)

        model = None
        for wo in [True, False]:
            try:
                m = Cls(**kwargs).to(device)
                m.load_state_dict(torch.load(str(path), map_location=device, weights_only=wo))
                m.eval()
                model = m
                break
            except Exception as e:
                last_err = e
        if model is None:
            print(f" Ошибка: {last_err}")
            continue

        # размеры входа: 112px для эмбеддера и возраста, 48px для эмоций
        size = 48 if name == "emotion" else 112
        img  = make_gray_as_rgb(size) if name == "emotion" else make_rgb(size)
        norm = norm_emotion if name == "emotion" else norm_imagenet
        t    = norm(to_tensor(img)).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(t)
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(N_BENCH):
                _ = model(t)
            if device.type == "cuda":
                torch.cuda.synchronize()

        ms = (time.perf_counter() - t0) / N_BENCH * 1000
        results["models"][name] = {"ms_per_image": round(ms, 2)}
        print(f" {ms:.1f} ms")

    return results if results["models"] else None

# сравнение и сохранение результатов

def compare_and_save(tf_res, pt_res):
    print("  РЕЗУЛЬТАТЫ ТЕСТА")

    if tf_res:
        print("  TF  : CPU (TF 2.11+ не поддерживает GPU на Windows)")
    if pt_res:
        print(f"  PT  : {pt_res.get('device','?')}")
    print()

    all_tasks = sorted(set(
        list(tf_res["models"].keys() if tf_res else []) +
        list(pt_res["models"].keys() if pt_res else [])
    ))

    recommendations = {}

    print(f"  {'Task':<18} {'TF (ms)':>10} {'PT (ms)':>10}   Winner")

    for task in all_tasks:
        tf_ms = tf_res["models"].get(task, {}).get("ms_per_image") if tf_res else None
        pt_ms = pt_res["models"].get(task, {}).get("ms_per_image") if pt_res else None

        tf_str = f"{tf_ms:.1f}" if tf_ms is not None else "N/A"
        pt_str = f"{pt_ms:.1f}" if pt_ms is not None else "N/A"

        if tf_ms is not None and pt_ms is not None:
            winner = "TF" if tf_ms < pt_ms else "PT (GPU)"
            recommendations[task] = "tensorflow" if tf_ms < pt_ms else "pytorch"
        elif pt_ms is not None:
            winner = "PT (единственный вариант)"
            recommendations[task] = "pytorch"
        elif tf_ms is not None:
            winner = "TF (единственный вариант)"
            recommendations[task] = "tensorflow"
        else:
            winner = "нет"

        print(f"  {task:<18} {tf_str:>10} {pt_str:>10}   {winner}")

    print()
    print("  РЕКОМЕНДАЦИЯ:")
    for task, fw in recommendations.items():
        print(f"    {task:<18} → {fw}")

    Path("models").mkdir(exist_ok=True)
    with open("models/comparison.json", "w") as f:
        json.dump({"tensorflow": tf_res, "pytorch": pt_res,
                   "recommended": recommendations}, f, indent=2)

    print("  Сохранено → models/comparison.json")
    print("  Следующие шаги:")
    print("    python telegram_bot.py Телеграм бот")
    print("    python main_camera.py камера в реальном времени")


# запуск

if __name__ == "__main__":
    print("  Face AI — Сравнение моделей")
    print("[1/2] Тестируем TensorFlow...")
    tf_res = bench_tf()
    print()
    print("[2/2] Тестируем PyTorch...")
    pt_res = bench_pt()

    if tf_res is None and pt_res is None:
        print("\nОШИБКА: Обученные модели не найдены.")
        print("Запусти: python train_pt.py --all --epochs 30")
        sys.exit(1)

    compare_and_save(tf_res, pt_res)