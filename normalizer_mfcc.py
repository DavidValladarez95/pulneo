import json
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")

normalizer = np.load(MODELS_DIR / "normalizer_mfcc.npz")

with open(MODELS_DIR / "metrics_tflite.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

mean = normalizer["mean"].reshape(-1).astype(float).tolist()
std = normalizer["std"].reshape(-1).astype(float).tolist()

data = {
    "target_sr": 16000,
    "window_sec": 1.0,
    "window_samples": 16000,
    "n_mfcc": 40,
    "expected_frames": 97,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "fmin": 50,
    "fmax": 8000,
    "threshold": metrics.get("threshold", 0.5),
    "mean": mean,
    "std": std
}

with open(MODELS_DIR / "normalizer_mfcc.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Generado:", MODELS_DIR / "normalizer_mfcc.json")
print("mean:", len(mean))
print("std:", len(std))
print("threshold:", data["threshold"])