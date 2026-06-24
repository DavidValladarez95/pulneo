"""
Pulneo - Entrenamiento RAD acústico para fisioterapia respiratoria

Pipeline completo:
    Coswara healthy breathing-deep.wav
    + UrbanSound8K ruido ambiental real
    -> MFCC
    -> CNN 2D liviana
    -> evaluación Keras
    -> TensorFlow Lite INT8
    -> evaluación TFLite
    -> artefactos para artículo científico

Salida:
    models/respiro_rad_mfcc.keras
    models/respiro_rad_mfcc_int8.tflite
    models/normalizer_mfcc.npz
    models/metrics_keras.json
    models/metrics_tflite.json
    models/experiment_metadata.json
    server.log

Notas científicas:
    - El modelo es RAD binario:
        0 = ruido/silencio/ambiente
        1 = actividad respiratoria
    - La clasificación clínica "correcto / corto-interrumpido / débil"
      se deriva después por reglas temporales y RMS, no por la CNN.
    - No se usa ruido gaussiano sintético como clase 0 principal.
"""

import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import kagglehub
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers


# ============================================================
# Configuración científica del experimento
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    # Centraliza los parámetros para que el experimento sea reproducible y fácil de auditar.
    # Reproducibilidad
    random_seed: int = 42
    experiment_name: str = "pulneo_rad_mfcc_int8_v1"

    # Rutas
    base_dir: Path = Path(__file__).resolve().parent
    data_dir_name: str = "data"
    coswara_dir_name: str = "coswara_extracted"
    urbansound_dir_name: str = "urbansound8k"
    model_dir_name: str = "models"
    log_file_name: str = "server.log"

    # Datasets
    coswara_dataset_slug: str = "janashreeananthan/coswara"
    urbansound_dataset_slug: str = "chrisfilo/urbansound8k"

    # Coswara
    positive_status_column: str = "covid_status"
    subject_id_column: str = "id"
    positive_status_values: Tuple[str, ...] = ("healthy",)
    positive_audio_files: Tuple[str, ...] = ("breathing-deep.wav",)

    # Modo debug
    debug_mode: bool = False
    debug_positive_subject_limit: int = 150

    # Negativos
    allow_synthetic_negative_fallback: bool = False
    balance_negative_to_positive_windows: bool = True

    # Audio
    target_sr: int = 16000
    n_channels: int = 1
    window_sec: float = 1.0
    hop_sec: float = 0.5

    # MFCC
    feature_type: str = "mfcc"
    n_mfcc: int = 40
    n_fft: int = 512
    win_length: int = 400       # 25 ms a 16 kHz
    hop_length: int = 160       # 10 ms a 16 kHz
    fmin: int = 50
    fmax: int = 8000

    # Preprocesamiento
    normalize_audio_for_mfcc: bool = True
    min_window_rms: float = 1e-4

    # Clases RAD
    class_noise: int = 0
    class_breathing: int = 1

    # Split por sujeto/grupo
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    # Entrenamiento
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    dropout_rate: float = 0.25

    # Inferencia RAD
    default_breath_prob_threshold: float = 0.50
    threshold_grid_min: float = 0.10
    threshold_grid_max: float = 0.90
    threshold_grid_steps: int = 81

    # Reglas clínicas posteriores al RAD
    min_correct_exhalation_sec: float = 4.0
    max_interruption_gap_sec: float = 0.5
    weak_exhalation_rms_percentile: float = 25.0

    # TensorFlow Lite
    tflite_quantization: str = "full_int8"
    representative_dataset_size: int = 100

    # Evaluación TFLite
    max_tflite_eval_samples: Optional[int] = None


CFG = ExperimentConfig()


# ============================================================
# Rutas derivadas
# ============================================================

# Las rutas y constantes derivadas dependen solo de CFG; no deben modificarse manualmente.
DATA_DIR = CFG.base_dir / CFG.data_dir_name
COSWARA_DIR = DATA_DIR / CFG.coswara_dir_name
URBANSOUND_DIR = DATA_DIR / CFG.urbansound_dir_name
MODEL_DIR = CFG.base_dir / CFG.model_dir_name
LOG_FILE = CFG.base_dir / CFG.log_file_name

WINDOW_LENGTH = int(CFG.target_sr * CFG.window_sec)
HOP_LENGTH_AUDIO = int(CFG.target_sr * CFG.hop_sec)
EXPECTED_MFCC_FRAMES = 1 + max(0, (WINDOW_LENGTH - CFG.n_fft) // CFG.hop_length)

KERAS_MODEL_FILE = MODEL_DIR / "respiro_rad_mfcc.keras"
TFLITE_MODEL_FILE = MODEL_DIR / "respiro_rad_mfcc_int8.tflite"
NORMALIZER_FILE = MODEL_DIR / "normalizer_mfcc.npz"
METRICS_KERAS_FILE = MODEL_DIR / "metrics_keras.json"
METRICS_TFLITE_FILE = MODEL_DIR / "metrics_tflite.json"
HISTORY_FILE = MODEL_DIR / "training_history.json"
METADATA_FILE = MODEL_DIR / "experiment_metadata.json"


# ============================================================
# Logging
# ============================================================

class _TeeStream:
    """Escribe simultáneamente en consola y archivo de log."""

    def __init__(self, stream, log_fp):
        self._stream = stream
        self._log_fp = log_fp

    def write(self, data):
        self._stream.write(data)
        self._log_fp.write(data)
        self._log_fp.flush()
        try:
            self._stream.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_fp.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


def start_server_log():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_FILE, "a", encoding="utf-8")
    log_fp.write(f"\n{'=' * 80}\n")
    log_fp.write(f"Inicio: {datetime.now().isoformat()}\n")
    log_fp.write(f"Experimento: {CFG.experiment_name}\n")
    log_fp.write(f"{'=' * 80}\n")
    log_fp.flush()

    # Redirige stdout/stderr para dejar trazabilidad completa de cada ejecución.
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    return (old_out, old_err), log_fp


def stop_server_log(state, log_fp):
    sys.stdout, sys.stderr = state[0], state[1]
    log_fp.write(f"\nFin: {datetime.now().isoformat()}\n")
    log_fp.close()


# ============================================================
# Utilidades generales
# ============================================================

def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COSWARA_DIR.mkdir(parents=True, exist_ok=True)
    URBANSOUND_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def path_to_str(path: Path) -> str:
    return str(path.resolve())


def list_wavs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def is_positive_audio_file(path: Path) -> bool:
    return path.name in CFG.positive_audio_files


def count_positive_wavs(root: Path) -> int:
    return sum(1 for p in list_wavs(root) if is_positive_audio_file(p))


def find_file(root: Path, filename: str) -> Optional[Path]:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


# ============================================================
# Coswara: descarga, unión y extracción segura
# ============================================================

def read_coswara_healthy_ids(coswara_base_path: Path) -> set:
    csv_path = find_file(coswara_base_path, "combined_data.csv")
    if csv_path is None:
        raise FileNotFoundError(
            f"No se encontró combined_data.csv dentro de {coswara_base_path}"
        )

    print(f"Metadatos Coswara: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validar el esquema temprano evita extraer audios con etiquetas ambiguas o incompletas.
    required_columns = {CFG.subject_id_column, CFG.positive_status_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"combined_data.csv no contiene columnas requeridas: {sorted(missing)}"
        )

    healthy_df = df[
        df[CFG.positive_status_column].astype(str).isin(CFG.positive_status_values)
    ]

    healthy_ids = set(
        healthy_df[CFG.subject_id_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not healthy_ids:
        raise ValueError("No se encontraron sujetos healthy en combined_data.csv")

    print(f"Sujetos healthy en metadatos: {len(healthy_ids)}")
    return healthy_ids


def copy_existing_coswara_wavs_from_tree(
    source_root: Path,
    target_root: Path,
    healthy_ids: set,
    subject_limit: Optional[int],
) -> int:
    copied = 0
    seen_subjects = set()

    for wav_path in sorted(source_root.rglob("*.wav")):
        if wav_path.name not in CFG.positive_audio_files:
            continue

        subject_id = wav_path.parent.name.strip()

        if subject_id not in healthy_ids:
            continue

        if subject_limit is not None and subject_id not in seen_subjects:
            if len(seen_subjects) >= subject_limit:
                continue

        seen_subjects.add(subject_id)

        dst = target_root / subject_id / wav_path.name
        if not dst.exists():
            safe_copy(wav_path, dst)
            copied += 1

    if copied:
        print(f"Copiados desde árbol Coswara ya extraído: {copied}")

    return copied


def combine_split_archive_parts(folder_path: Path, aa_file: Path, temp_dir: Path) -> Path:
    base_tar_name = aa_file.name[:-3]
    parts = sorted(
        p for p in folder_path.iterdir()
        if p.is_file() and p.name.startswith(base_tar_name)
    )

    if not parts:
        raise FileNotFoundError(f"No hay partes para {aa_file}")

    combined_tar_path = temp_dir / base_tar_name

    # Coswara puede venir como tar dividido; se recompone en temporal para no ensuciar data/.
    print(f"Fusionando {len(parts)} partes: {base_tar_name}")
    with open(combined_tar_path, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    return combined_tar_path


def extract_coswara_members_from_tar(
    tar_path: Path,
    target_root: Path,
    healthy_ids: set,
    subject_limit: Optional[int],
    existing_subjects: set,
) -> int:
    extracted = 0

    try:
        with tarfile.open(tar_path, mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                member_path = Path(member.name)
                filename = member_path.name

                if filename not in CFG.positive_audio_files:
                    continue

                subject_id = member_path.parent.name.strip()

                # Solo se extraen sujetos sanos para mantener la clase positiva clínicamente consistente.
                if subject_id not in healthy_ids:
                    continue

                if subject_limit is not None and subject_id not in existing_subjects:
                    if len(existing_subjects) >= subject_limit:
                        continue

                existing_subjects.add(subject_id)

                out_path = target_root / subject_id / filename
                if out_path.exists():
                    continue

                extracted_file = tar.extractfile(member)
                if extracted_file is None:
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(extracted_file, f)

                extracted += 1

                if extracted % 25 == 0:
                    print(f"  Audios Coswara extraídos: {extracted}")

    except tarfile.TarError as e:
        print(f"Error leyendo tar {tar_path}: {e}")

    return extracted


def process_coswara_split_archives(
    coswara_base_path: Path,
    target_root: Path,
    healthy_ids: set,
    subject_limit: Optional[int],
) -> int:
    aa_files = sorted(coswara_base_path.rglob("*.aa"))

    if not aa_files:
        print("No se encontraron archivos .aa. Se intentará usar árbol ya extraído.")
        return 0

    existing_subjects = {
        p.parent.name for p in target_root.rglob("*.wav")
        if p.name in CFG.positive_audio_files
    }

    total_extracted = 0

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        for aa_file in aa_files:
            if subject_limit is not None and len(existing_subjects) >= subject_limit:
                break

            folder_path = aa_file.parent

            try:
                combined_tar = combine_split_archive_parts(folder_path, aa_file, temp_dir)
                extracted = extract_coswara_members_from_tar(
                    tar_path=combined_tar,
                    target_root=target_root,
                    healthy_ids=healthy_ids,
                    subject_limit=subject_limit,
                    existing_subjects=existing_subjects,
                )
                total_extracted += extracted

                try:
                    combined_tar.unlink(missing_ok=True)
                except TypeError:
                    if combined_tar.exists():
                        combined_tar.unlink()

            except Exception as e:
                print(f"Error procesando {aa_file}: {e}")

    print(f"Extracción Coswara desde archivos partidos: {total_extracted}")
    return total_extracted


def ensure_coswara_wavs() -> Path:
    print("\n[1] Verificando Coswara...")

    existing = count_positive_wavs(COSWARA_DIR)
    subject_limit = CFG.debug_positive_subject_limit if CFG.debug_mode else None

    if subject_limit is not None and existing >= subject_limit:
        print(f"Coswara ya tiene {existing} audios positivos en modo debug.")
        return COSWARA_DIR

    if subject_limit is None and existing > 0:
        print(f"Coswara ya tiene {existing} audios positivos extraídos.")
        return COSWARA_DIR

    print("Descargando/verificando Coswara con kagglehub...")
    coswara_base = Path(kagglehub.dataset_download(CFG.coswara_dataset_slug))
    print(f"Ruta Coswara kagglehub: {coswara_base}")

    healthy_ids = read_coswara_healthy_ids(coswara_base)

    copy_existing_coswara_wavs_from_tree(
        source_root=coswara_base,
        target_root=COSWARA_DIR,
        healthy_ids=healthy_ids,
        subject_limit=subject_limit,
    )

    process_coswara_split_archives(
        coswara_base_path=coswara_base,
        target_root=COSWARA_DIR,
        healthy_ids=healthy_ids,
        subject_limit=subject_limit,
    )

    final_count = count_positive_wavs(COSWARA_DIR)

    if final_count < 1:
        raise RuntimeError(
            "No se encontró ningún breathing-deep.wav healthy. "
            "Revisa descarga de Coswara, credenciales Kaggle o estructura del dataset."
        )

    print(f"Coswara listo: {final_count} archivos positivos disponibles.")
    return COSWARA_DIR


# ============================================================
# UrbanSound8K: descarga / verificación
# ============================================================

def copy_urbansound_if_downloaded(download_root: Path, target_root: Path) -> int:
    wavs = list_wavs(download_root)
    if not wavs:
        return 0

    copied = 0

    for wav in wavs:
        relative_parts = wav.parts[-3:] if len(wav.parts) >= 3 else (wav.name,)
        dst = target_root.joinpath(*relative_parts)

        if not dst.exists():
            safe_copy(wav, dst)
            copied += 1

    if copied:
        print(f"UrbanSound8K copiado a data local: {copied} wavs")

    return copied


def ensure_urbansound8k() -> Path:
    print("\n[2] Verificando UrbanSound8K / negativos reales...")

    existing = list_wavs(URBANSOUND_DIR)
    if existing:
        print(f"UrbanSound8K local disponible: {len(existing)} wavs.")
        return URBANSOUND_DIR

    print("Descargando/verificando UrbanSound8K con kagglehub...")

    try:
        download_root = Path(kagglehub.dataset_download(CFG.urbansound_dataset_slug))
        print(f"Ruta UrbanSound8K kagglehub: {download_root}")
        copied = copy_urbansound_if_downloaded(download_root, URBANSOUND_DIR)

        if copied < 1 and list_wavs(download_root):
            print("Se usará la ruta cacheada de kagglehub para UrbanSound8K.")
            return download_root

    except Exception as e:
        print(f"No se pudo descargar UrbanSound8K: {e}")

    final = list_wavs(URBANSOUND_DIR)
    if final:
        return URBANSOUND_DIR

    # Para resultados publicables se prefieren negativos reales sobre ruido gaussiano sintético.
    if CFG.allow_synthetic_negative_fallback:
        print(
            "ADVERTENCIA: se activó fallback sintético. "
            "No recomendado para resultados científicos."
        )
        return URBANSOUND_DIR

    raise RuntimeError(
        "No hay negativos reales. Coloca wavs de ruido ambiental en "
        f"{URBANSOUND_DIR} o habilita allow_synthetic_negative_fallback=True "
        "solo para depuración."
    )


# ============================================================
# Audio y características
# ============================================================

@dataclass
class WindowRecord:
    segment: np.ndarray
    label: int
    group_id: str
    source_path: str
    rms: float


def load_audio_mono(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=CFG.target_sr, mono=True)

    audio = audio.astype(np.float32)

    if audio.size == 0:
        return audio

    if CFG.normalize_audio_for_mfcc:
        # La normalización por archivo estabiliza MFCC frente a diferencias de ganancia de grabación.
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 0:
            audio = audio / max_abs

    return audio.astype(np.float32)


def iter_audio_windows(audio: np.ndarray) -> Iterable[np.ndarray]:
    if audio.size < 1:
        return

    # Las ventanas solapadas aumentan cobertura temporal sin mezclar muestras entre archivos.
    if audio.size < WINDOW_LENGTH:
        padded = np.pad(audio, (0, WINDOW_LENGTH - audio.size))
        yield padded.astype(np.float32)
        return

    last_start = audio.size - WINDOW_LENGTH

    for start in range(0, last_start + 1, HOP_LENGTH_AUDIO):
        window = audio[start:start + WINDOW_LENGTH]
        if window.size == WINDOW_LENGTH:
            yield window.astype(np.float32)


def compute_rms(segment: np.ndarray) -> float:
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(segment.astype(np.float32)))))


def compute_mfcc(segment: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=segment.astype(np.float32),
        sr=CFG.target_sr,
        n_mfcc=CFG.n_mfcc,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        win_length=CFG.win_length,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        center=False,
    )

    feature = mfcc.T.astype(np.float32)

    # Mantener una forma fija simplifica entrenamiento, normalización y exportación a TFLite.
    if feature.shape[0] < EXPECTED_MFCC_FRAMES:
        pad_rows = EXPECTED_MFCC_FRAMES - feature.shape[0]
        feature = np.pad(feature, ((0, pad_rows), (0, 0)), mode="constant")
    elif feature.shape[0] > EXPECTED_MFCC_FRAMES:
        feature = feature[:EXPECTED_MFCC_FRAMES, :]

    return feature.astype(np.float32)


def collect_positive_records(coswara_root: Path) -> List[WindowRecord]:
    print("\n[3] Construyendo ventanas positivas: respiración Coswara healthy...")

    records: List[WindowRecord] = []
    seen_subjects = set()
    subject_limit = CFG.debug_positive_subject_limit if CFG.debug_mode else None

    positive_wavs = [
        p for p in sorted(coswara_root.rglob("*.wav"))
        if p.name in CFG.positive_audio_files
    ]

    for wav_path in positive_wavs:
        subject_id = wav_path.parent.name

        # En debug se limita por sujeto, no por ventana, para conservar independencia del split.
        if subject_limit is not None and subject_id not in seen_subjects:
            if len(seen_subjects) >= subject_limit:
                continue

        seen_subjects.add(subject_id)

        try:
            audio = load_audio_mono(wav_path)
        except Exception as e:
            print(f"No se pudo cargar positivo {wav_path}: {e}")
            continue

        for segment in iter_audio_windows(audio):
            rms = compute_rms(segment)
            if rms < CFG.min_window_rms:
                continue

            records.append(
                WindowRecord(
                    segment=segment,
                    label=CFG.class_breathing,
                    group_id=f"pos_subject_{subject_id}",
                    source_path=str(wav_path),
                    rms=rms,
                )
            )

    if not records:
        raise RuntimeError("No se generaron ventanas positivas.")

    print(f"Ventanas positivas: {len(records)}")
    print(f"Sujetos positivos: {len(seen_subjects)}")
    return records


def collect_negative_records(
    noise_root: Path,
    target_count: Optional[int],
) -> List[WindowRecord]:
    print("\n[4] Construyendo ventanas negativas: ruido ambiental real...")

    records: List[WindowRecord] = []
    rng = np.random.default_rng(CFG.random_seed)

    wavs = list_wavs(noise_root)

    if not wavs and CFG.allow_synthetic_negative_fallback:
        print("Generando negativos sintéticos solo para depuración.")
        synthetic_count = target_count or 1000

        for i in range(synthetic_count):
            segment = rng.normal(0, 0.05, WINDOW_LENGTH).astype(np.float32)
            records.append(
                WindowRecord(
                    segment=segment,
                    label=CFG.class_noise,
                    group_id=f"synthetic_noise_{i}",
                    source_path="synthetic_gaussian_noise",
                    rms=compute_rms(segment),
                )
            )

        return records

    if not wavs:
        raise RuntimeError(f"No se encontraron wavs negativos en {noise_root}")

    wavs_array = np.array(wavs, dtype=object)
    rng.shuffle(wavs_array)
    wavs = [Path(p) for p in wavs_array.tolist()]

    # Se recorre ruido real en orden aleatorio reproducible para balancear sin sesgar por carpeta.
    for wav_path in wavs:
        if target_count is not None and len(records) >= target_count:
            break

        try:
            audio = load_audio_mono(wav_path)
        except Exception as e:
            print(f"No se pudo cargar negativo {wav_path}: {e}")
            continue

        group_id = f"neg_file_{wav_path.stem}"

        for segment in iter_audio_windows(audio):
            if target_count is not None and len(records) >= target_count:
                break

            rms = compute_rms(segment)

            if rms < CFG.min_window_rms:
                continue

            records.append(
                WindowRecord(
                    segment=segment,
                    label=CFG.class_noise,
                    group_id=group_id,
                    source_path=str(wav_path),
                    rms=rms,
                )
            )

    if not records:
        raise RuntimeError("No se generaron ventanas negativas.")

    print(f"Ventanas negativas: {len(records)}")
    return records


def records_to_feature_arrays(
    records: Sequence[WindowRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    print("\n[5] Extrayendo MFCC...")

    # Cada registro conserva etiqueta, grupo y RMS para evaluación y reglas clínicas posteriores.
    X_features = []
    y = []
    groups = []
    source_paths = []
    rms_values = []

    total = len(records)

    for i, rec in enumerate(records, start=1):
        feature = compute_mfcc(rec.segment)
        X_features.append(feature)
        y.append(rec.label)
        groups.append(rec.group_id)
        source_paths.append(rec.source_path)
        rms_values.append(rec.rms)

        if i % 500 == 0:
            print(f"  MFCC procesados: {i}/{total}")

    X = np.asarray(X_features, dtype=np.float32)
    X = X[..., np.newaxis]

    y_arr = np.asarray(y, dtype=np.int32)
    groups_arr = np.asarray(groups, dtype=object)
    rms_arr = np.asarray(rms_values, dtype=np.float32)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y_arr.shape}")
    print(f"Clases: ruido={int(np.sum(y_arr == 0))}, respiración={int(np.sum(y_arr == 1))}")

    return X, y_arr, groups_arr, source_paths, rms_arr


# ============================================================
# Split por grupo/sujeto
# ============================================================

def split_group_list(groups: List[str], rng: np.random.Generator) -> Tuple[set, set, set]:
    groups = list(groups)
    rng.shuffle(groups)

    n = len(groups)

    if n < 3:
        raise RuntimeError(
            "Hay muy pocos grupos para hacer train/val/test por sujeto/grupo."
        )

    n_train = max(1, int(round(n * CFG.train_split)))
    n_val = max(1, int(round(n * CFG.val_split)))

    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    if not test_groups:
        test_groups = {val_groups.pop()}

    return train_groups, val_groups, test_groups


def subject_wise_train_val_test_split(
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n[6] Split train/val/test por sujeto/grupo...")

    rng = np.random.default_rng(CFG.random_seed)

    # El split por grupo evita que ventanas del mismo sujeto/archivo aparezcan en conjuntos distintos.
    group_to_labels: Dict[str, set] = {}
    for label, group in zip(y.tolist(), groups.tolist()):
        group_to_labels.setdefault(group, set()).add(int(label))

    for group, labels in group_to_labels.items():
        if len(labels) != 1:
            raise RuntimeError(
                f"El grupo {group} contiene múltiples clases: {labels}. "
                "Esto rompe el split por grupo."
            )

    class_to_groups: Dict[int, List[str]] = {
        CFG.class_noise: [],
        CFG.class_breathing: [],
    }

    for group, labels in group_to_labels.items():
        label = next(iter(labels))
        class_to_groups[label].append(group)

    train_groups_all = set()
    val_groups_all = set()
    test_groups_all = set()

    for label in [CFG.class_noise, CFG.class_breathing]:
        cls_groups = class_to_groups[label]

        if len(cls_groups) < 3:
            raise RuntimeError(
                f"La clase {label} tiene muy pocos grupos: {len(cls_groups)}"
            )

        tr, va, te = split_group_list(cls_groups, rng)
        train_groups_all.update(tr)
        val_groups_all.update(va)
        test_groups_all.update(te)

        print(
            f"Clase {label}: grupos train={len(tr)}, "
            f"val={len(va)}, test={len(te)}"
        )

    train_idx = np.asarray(
        [i for i, g in enumerate(groups.tolist()) if g in train_groups_all],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [i for i, g in enumerate(groups.tolist()) if g in val_groups_all],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [i for i, g in enumerate(groups.tolist()) if g in test_groups_all],
        dtype=np.int64,
    )

    print(f"Ventanas train: {len(train_idx)}")
    print(f"Ventanas val:   {len(val_idx)}")
    print(f"Ventanas test:  {len(test_idx)}")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        yy = y[idx]
        print(
            f"{name}: ruido={int(np.sum(yy == 0))}, "
            f"respiración={int(np.sum(yy == 1))}"
        )

    return train_idx, val_idx, test_idx


# ============================================================
# Normalización
# ============================================================

def fit_feature_normalizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # El normalizador se ajusta solo con train para evitar fuga estadística hacia validación/test.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_normalizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def save_normalizer(mean: np.ndarray, std: np.ndarray) -> None:
    # Se guardan también parámetros de extracción para reproducir inferencia fuera de Python.
    np.savez(
        NORMALIZER_FILE,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        target_sr=np.asarray([CFG.target_sr], dtype=np.int32),
        window_sec=np.asarray([CFG.window_sec], dtype=np.float32),
        hop_sec=np.asarray([CFG.hop_sec], dtype=np.float32),
        n_mfcc=np.asarray([CFG.n_mfcc], dtype=np.int32),
        n_fft=np.asarray([CFG.n_fft], dtype=np.int32),
        win_length=np.asarray([CFG.win_length], dtype=np.int32),
        hop_length=np.asarray([CFG.hop_length], dtype=np.int32),
        fmin=np.asarray([CFG.fmin], dtype=np.int32),
        fmax=np.asarray([CFG.fmax], dtype=np.int32),
    )
    print(f"Normalizador guardado: {NORMALIZER_FILE}")


# ============================================================
# Modelo
# ============================================================

def build_rad_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    # CNN compacta sobre MFCC: suficiente para RAD binario y compatible con despliegue móvil.
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(48, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(32, activation="relu"),
            layers.Dropout(CFG.dropout_rate),

            layers.Dense(1, activation="sigmoid"),
        ],
        name="pulneo_rad_mfcc_cnn",
    )

    optimizer = optimizers.Adam(learning_rate=CFG.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.asarray([CFG.class_noise, CFG.class_breathing], dtype=np.int32)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(np.int32),
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.Model:
    print("\n[7] Entrenando modelo RAD MFCC...")

    model = build_rad_model(input_shape=X_train.shape[1:])
    model.summary()

    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    # Los callbacks priorizan generalización: detienen sobreajuste y reducen LR si val_loss se estanca.
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=CFG.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        class_weight=class_weights,
        callbacks=cb,
        verbose=1,
    )

    history_dict = {
        k: [float(x) for x in v]
        for k, v in history.history.items()
    }
    json_dump(history_dict, HISTORY_FILE)
    print(f"Historial guardado: {HISTORY_FILE}")

    model.save(KERAS_MODEL_FILE)
    print(f"Modelo Keras guardado: {KERAS_MODEL_FILE}")

    return model


# ============================================================
# Evaluación y umbral
# ============================================================

def predict_keras_probabilities(
    model: tf.keras.Model,
    X: np.ndarray,
) -> np.ndarray:
    probs = model.predict(X, batch_size=CFG.batch_size, verbose=0)
    return probs.reshape(-1).astype(np.float32)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    # El umbral se calibra en validación; test queda reservado para estimar rendimiento final.
    thresholds = np.linspace(
        CFG.threshold_grid_min,
        CFG.threshold_grid_max,
        CFG.threshold_grid_steps,
    )

    best_threshold = CFG.default_breath_prob_threshold
    best_f1 = -1.0

    for th in thresholds:
        pred = (probabilities >= th).astype(np.int32)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(th)

    print(f"Umbral óptimo por F1 en validación: {best_threshold:.3f}")
    print(f"F1 validación en ese umbral: {best_f1:.4f}")

    return best_threshold, best_f1


def build_metrics_dict(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    elapsed_sec: Optional[float] = None,
) -> Dict:
    y_pred = (probabilities >= threshold).astype(np.int32)

    # Las métricas se serializan en formato estable para auditoría y comparación entre ejecuciones.
    report = classification_report(
        y_true,
        y_pred,
        target_names=["noise_or_silence", "breathing"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels": ["noise_or_silence", "breathing"],
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
        "n_samples": int(len(y_true)),
    }

    if elapsed_sec is not None:
        metrics["total_inference_time_sec"] = float(elapsed_sec)
        metrics["mean_inference_time_ms"] = float((elapsed_sec / max(1, len(y_true))) * 1000.0)

    return metrics


def print_metrics(metrics: Dict, title: str) -> None:
    print(f"\n{title}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print("Matriz de confusión [[TN, FP], [FN, TP]]:")
    print(np.asarray(metrics["confusion_matrix"]))

    if "mean_inference_time_ms" in metrics:
        print(f"Latencia media: {metrics['mean_inference_time_ms']:.4f} ms/muestra")


def evaluate_keras_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict, float]:
    print("\n[8] Evaluando modelo Keras...")

    val_probs = predict_keras_probabilities(model, X_val)
    best_threshold, _ = find_best_threshold(y_val, val_probs)

    start = time.perf_counter()
    test_probs = predict_keras_probabilities(model, X_test)
    elapsed = time.perf_counter() - start

    metrics = build_metrics_dict(
        y_true=y_test,
        probabilities=test_probs,
        threshold=best_threshold,
        elapsed_sec=elapsed,
    )

    print_metrics(metrics, "Resultados Keras en test")
    json_dump(metrics, METRICS_KERAS_FILE)
    print(f"Métricas Keras guardadas: {METRICS_KERAS_FILE}")

    return metrics, best_threshold


# ============================================================
# Conversión TensorFlow Lite INT8
# ============================================================

def representative_dataset_generator(X_train: np.ndarray):
    n = min(CFG.representative_dataset_size, len(X_train))

    # TFLite usa estas muestras para estimar rangos de activación durante cuantización INT8.
    for i in range(n):
        sample = X_train[i:i + 1].astype(np.float32)
        yield [sample]


def convert_to_tflite_int8(
    model: tf.keras.Model,
    X_train: np.ndarray,
) -> bytes:
    print("\n[9] Convirtiendo a TensorFlow Lite INT8...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_generator(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Forzar entrada/salida INT8 reduce tamaño y prepara el modelo para aceleradores embebidos.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_FILE, "wb") as f:
        f.write(tflite_model)

    size_mb = TFLITE_MODEL_FILE.stat().st_size / (1024 * 1024)

    print(f"Modelo TFLite INT8 guardado: {TFLITE_MODEL_FILE}")
    print(f"Tamaño TFLite: {size_mb:.4f} MB")

    return tflite_model


def quantize_input_for_tflite(
    x: np.ndarray,
    input_details: Dict,
) -> np.ndarray:
    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]

    if dtype in (np.int8, np.uint8):
        if scale == 0:
            raise RuntimeError("Escala de cuantización de entrada es 0.")
        x_q = np.round(x / scale + zero_point)
        info = np.iinfo(dtype)
        x_q = np.clip(x_q, info.min, info.max)
        return x_q.astype(dtype)

    return x.astype(dtype)


def dequantize_output_from_tflite(
    y: np.ndarray,
    output_details: Dict,
) -> np.ndarray:
    dtype = output_details["dtype"]
    scale, zero_point = output_details["quantization"]

    if dtype in (np.int8, np.uint8):
        if scale == 0:
            raise RuntimeError("Escala de cuantización de salida es 0.")
        return (y.astype(np.float32) - zero_point) * scale

    return y.astype(np.float32)


def predict_tflite_probabilities(
    tflite_model_path: Path,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite input:", input_details["shape"], input_details["dtype"], input_details["quantization"])
    print("TFLite output:", output_details["shape"], output_details["dtype"], output_details["quantization"])

    probs = []

    start = time.perf_counter()

    # La evaluación se ejecuta muestra a muestra para medir la latencia real del intérprete TFLite.
    for i in range(len(X)):
        sample = X[i:i + 1].astype(np.float32)
        sample_q = quantize_input_for_tflite(sample, input_details)

        interpreter.set_tensor(input_details["index"], sample_q)
        interpreter.invoke()

        output_q = interpreter.get_tensor(output_details["index"])
        output = dequantize_output_from_tflite(output_q, output_details)
        probs.append(float(output.reshape(-1)[0]))

    elapsed = time.perf_counter() - start

    return np.asarray(probs, dtype=np.float32), elapsed


def evaluate_tflite_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict:
    print("\n[10] Evaluando modelo TFLite INT8...")

    if CFG.max_tflite_eval_samples is not None:
        # Útil cuando el set de prueba es grande y solo se necesita una estimación rápida de latencia.
        X_eval = X_test[:CFG.max_tflite_eval_samples]
        y_eval = y_test[:CFG.max_tflite_eval_samples]
    else:
        X_eval = X_test
        y_eval = y_test

    probs, elapsed = predict_tflite_probabilities(TFLITE_MODEL_FILE, X_eval)

    metrics = build_metrics_dict(
        y_true=y_eval,
        probabilities=probs,
        threshold=threshold,
        elapsed_sec=elapsed,
    )

    print_metrics(metrics, "Resultados TFLite INT8 en test")
    json_dump(metrics, METRICS_TFLITE_FILE)
    print(f"Métricas TFLite guardadas: {METRICS_TFLITE_FILE}")

    return metrics


# ============================================================
# Clasificación clínica posterior al RAD
# ============================================================

def classify_exhalation_from_rad_sequence(
    breath_probabilities: Sequence[float],
    rms_values: Sequence[float],
    threshold: float,
    weak_rms_threshold: Optional[float] = None,
) -> Dict:
    """
    Convierte una secuencia de probabilidades RAD en clasificación clínica.

    Entrada:
        breath_probabilities:
            Probabilidades de respiración por ventana deslizante.
        rms_values:
            RMS por ventana correspondiente.
        threshold:
            Umbral RAD.
        weak_rms_threshold:
            Umbral absoluto de debilidad. Si None, se usa percentil configurado.

    Salida:
        {
            "clinical_label": "correct" | "short_or_interrupted" | "weak",
            "exhalation_duration_sec": float,
            "max_interruption_gap_sec": float,
            "mean_rms_during_breath": float,
            "n_breath_windows": int,
        }
    """

    probs = np.asarray(breath_probabilities, dtype=np.float32)
    rms = np.asarray(rms_values, dtype=np.float32)

    if probs.size != rms.size:
        raise ValueError("breath_probabilities y rms_values deben tener igual longitud.")

    is_breath = probs >= threshold
    n_breath_windows = int(np.sum(is_breath))

    # Sin ventanas positivas no hay evidencia suficiente de exhalación sostenida.
    if n_breath_windows == 0:
        return {
            "clinical_label": "short_or_interrupted",
            "exhalation_duration_sec": 0.0,
            "max_interruption_gap_sec": 0.0,
            "mean_rms_during_breath": 0.0,
            "n_breath_windows": 0,
        }

    breath_indices = np.where(is_breath)[0]
    exhalation_duration_sec = float(n_breath_windows * CFG.hop_sec)

    # Las brechas entre ventanas respiratorias capturan interrupciones durante la exhalación.
    gaps = np.diff(breath_indices) - 1
    max_gap_windows = int(np.max(gaps)) if gaps.size else 0
    max_gap_sec = float(max_gap_windows * CFG.hop_sec)

    breath_rms = rms[is_breath]
    mean_rms = float(np.mean(breath_rms))

    if weak_rms_threshold is None:
        # El umbral adaptativo compara la fuerza de la exhalación contra la propia secuencia.
        weak_rms_threshold = float(
            np.percentile(rms[rms > 0], CFG.weak_exhalation_rms_percentile)
        ) if np.any(rms > 0) else 0.0

    if (
        exhalation_duration_sec >= CFG.min_correct_exhalation_sec
        and max_gap_sec <= CFG.max_interruption_gap_sec
        and mean_rms >= weak_rms_threshold
    ):
        label = "correct"
    elif (
        exhalation_duration_sec < CFG.min_correct_exhalation_sec
        or max_gap_sec > CFG.max_interruption_gap_sec
    ):
        label = "short_or_interrupted"
    else:
        label = "weak"

    return {
        "clinical_label": label,
        "exhalation_duration_sec": exhalation_duration_sec,
        "max_interruption_gap_sec": max_gap_sec,
        "mean_rms_during_breath": mean_rms,
        "n_breath_windows": n_breath_windows,
        "weak_rms_threshold": float(weak_rms_threshold),
    }


# ============================================================
# Metadatos
# ============================================================

def save_experiment_metadata(
    n_positive_windows: int,
    n_negative_windows: int,
    X_shape: Tuple[int, ...],
    keras_metrics: Dict,
    tflite_metrics: Dict,
) -> None:
    cfg_dict = asdict(CFG)

    # El metadato une configuración, artefactos y métricas para reconstruir el experimento.
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)

    metadata = {
        "experiment_name": CFG.experiment_name,
        "created_at": datetime.now().isoformat(),
        "contributors": {
            "code_author": "David Fernando Valladarez Muñoz",
            "clinical_contributor": "María Gabriela Viñansaca Cabrera",
            "clinical_contribution": (
                "Conceptualización clínica, validación fisioterapéutica "
                "y revisión del protocolo."
            ),
        },
        "config": cfg_dict,
        "derived_constants": {
            "window_length_samples": WINDOW_LENGTH,
            "hop_length_audio_samples": HOP_LENGTH_AUDIO,
            "expected_mfcc_frames": EXPECTED_MFCC_FRAMES,
        },
        "dataset_summary": {
            "positive_source": "Coswara healthy breathing-deep.wav",
            "negative_source": "UrbanSound8K real environmental audio",
            "n_positive_windows": int(n_positive_windows),
            "n_negative_windows": int(n_negative_windows),
            "X_shape": list(X_shape),
        },
        "artifacts": {
            "keras_model": path_to_str(KERAS_MODEL_FILE),
            "tflite_model": path_to_str(TFLITE_MODEL_FILE),
            "normalizer": path_to_str(NORMALIZER_FILE),
            "keras_metrics": path_to_str(METRICS_KERAS_FILE),
            "tflite_metrics": path_to_str(METRICS_TFLITE_FILE),
            "history": path_to_str(HISTORY_FILE),
        },
        "keras_test_summary": {
            "accuracy": keras_metrics["accuracy"],
            "precision": keras_metrics["precision"],
            "recall": keras_metrics["recall"],
            "f1_score": keras_metrics["f1_score"],
            "threshold": keras_metrics["threshold"],
            "mean_inference_time_ms": keras_metrics.get("mean_inference_time_ms"),
        },
        "tflite_test_summary": {
            "accuracy": tflite_metrics["accuracy"],
            "precision": tflite_metrics["precision"],
            "recall": tflite_metrics["recall"],
            "f1_score": tflite_metrics["f1_score"],
            "threshold": tflite_metrics["threshold"],
            "mean_inference_time_ms": tflite_metrics.get("mean_inference_time_ms"),
        },
    }

    json_dump(metadata, METADATA_FILE)
    print(f"Metadatos guardados: {METADATA_FILE}")


# ============================================================
# Pipeline principal
# ============================================================

def run_pipeline() -> None:
    set_reproducibility(CFG.random_seed)
    ensure_directories()

    print("Configuración principal:")
    print(f"  TARGET_SR: {CFG.target_sr}")
    print(f"  WINDOW_SEC: {CFG.window_sec}")
    print(f"  HOP_SEC: {CFG.hop_sec}")
    print(f"  N_MFCC: {CFG.n_mfcc}")
    print(f"  DEBUG_MODE: {CFG.debug_mode}")
    print(f"  FEATURE_SHAPE esperado: ({EXPECTED_MFCC_FRAMES}, {CFG.n_mfcc}, 1)")

    coswara_root = ensure_coswara_wavs()
    urbansound_root = ensure_urbansound8k()

    positive_records = collect_positive_records(coswara_root)

    # El balance evita que una clase domine el entrenamiento cuando hay mucho ruido disponible.
    negative_target = (
        len(positive_records)
        if CFG.balance_negative_to_positive_windows
        else None
    )

    negative_records = collect_negative_records(
        noise_root=urbansound_root,
        target_count=negative_target,
    )

    all_records = positive_records + negative_records

    rng = np.random.default_rng(CFG.random_seed)
    rng.shuffle(all_records)

    X, y, groups, source_paths, rms_values = records_to_feature_arrays(all_records)

    train_idx, val_idx, test_idx = subject_wise_train_val_test_split(y, groups)

    # Separar los arreglos antes de normalizar mantiene clara la frontera entre conjuntos.
    X_train_raw = X[train_idx]
    y_train = y[train_idx]

    X_val_raw = X[val_idx]
    y_val = y[val_idx]

    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    print("\n[Normalización] Ajustando normalizador solo con train...")
    mean, std = fit_feature_normalizer(X_train_raw)
    save_normalizer(mean, std)

    # Se reutilizan exactamente mean/std de train en validación, test y posterior inferencia.
    X_train = apply_feature_normalizer(X_train_raw, mean, std)
    X_val = apply_feature_normalizer(X_val_raw, mean, std)
    X_test = apply_feature_normalizer(X_test_raw, mean, std)

    model = train_model(X_train, y_train, X_val, y_val)

    keras_metrics, best_threshold = evaluate_keras_model(
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    convert_to_tflite_int8(model, X_train)

    tflite_metrics = evaluate_tflite_model(
        X_test=X_test,
        y_test=y_test,
        threshold=best_threshold,
    )

    n_positive_windows = int(np.sum(y == CFG.class_breathing))
    n_negative_windows = int(np.sum(y == CFG.class_noise))

    save_experiment_metadata(
        n_positive_windows=n_positive_windows,
        n_negative_windows=n_negative_windows,
        X_shape=tuple(X.shape),
        keras_metrics=keras_metrics,
        tflite_metrics=tflite_metrics,
    )

    print("\nPipeline completado correctamente.")
    print(f"Keras:  {KERAS_MODEL_FILE}")
    print(f"TFLite: {TFLITE_MODEL_FILE}")
    print(f"Log:    {LOG_FILE}")


def main() -> None:
    state, log_fp = start_server_log()
    try:
        run_pipeline()
    finally:
        stop_server_log(state, log_fp)


if __name__ == "__main__":
    main()