import json
import os
import re
import hashlib
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, register_keras_serializable


TRAINING_CONFIG_ENV_VAR = "SIREN_TRAINING_CONFIG_PATH"


def load_runtime_config_overrides():
    """
    Carga overrides de configuracion desde un JSON opcional.

    Esto permite lanzar el mismo script varias veces con ajustes distintos sin
    editar manualmente el fichero principal entre experimentos.
    """
    config_path = os.environ.get(TRAINING_CONFIG_ENV_VAR)
    if not config_path:
        return {}, None

    with open(config_path, "r", encoding="utf-8") as file_handle:
        overrides = json.load(file_handle)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"El fichero indicado en {TRAINING_CONFIG_ENV_VAR} debe contener un objeto JSON."
        )

    return overrides, config_path


RUNTIME_CONFIG_OVERRIDES, RUNTIME_CONFIG_PATH = load_runtime_config_overrides()


def get_config_value(name, default):
    """Devuelve el valor sobrescrito desde JSON o el valor por defecto."""
    return RUNTIME_CONFIG_OVERRIDES.get(name, default)


def build_default_split_manifest_basename():
    """
    Construye un basename reproducible y especifico de la configuracion de split.

    Esto evita que configuraciones distintas del pipeline compartan el mismo
    CSV/JSON de manifiesto dentro de `dataset/metadata`.
    """
    fingerprint_payload = {
        "sample_rate": int(get_config_value("SAMPLE_RATE", 16000)),
        "chunk_length_s": float(get_config_value("CHUNK_LENGTH_S", 0.5)),
        "overlap_s": float(get_config_value("OVERLAP_S", 0.0)),
        "random_seed": int(get_config_value("RANDOM_SEED", 42)),
        "split_train_fraction": float(get_config_value("SPLIT_TRAIN_FRACTION", 0.70)),
        "split_validation_fraction": float(
            get_config_value("SPLIT_VALIDATION_FRACTION", 0.15)
        ),
        "split_test_fraction": float(get_config_value("SPLIT_TEST_FRACTION", 0.15)),
        "split_stratify_columns": list(
            get_config_value("SPLIT_STRATIFY_COLUMNS", ["label", "domain"])
        ),
        "split_weight_column": str(get_config_value("SPLIT_WEIGHT_COLUMN", "num_chunks")),
        "split_row_cost_weight": float(get_config_value("SPLIT_ROW_COST_WEIGHT", 0.15)),
        "apply_train_background_subsampling": bool(
            get_config_value("APPLY_TRAIN_BACKGROUND_SUBSAMPLING", True)
        ),
        "train_background_to_siren_chunk_ratio": float(
            get_config_value("TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO", 1.0)
        ),
        "train_background_min_groups_per_bucket": int(
            get_config_value("TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET", 1)
        ),
        "train_background_default_bucket_weight": float(
            get_config_value("TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT", 1.0)
        ),
        "train_background_hard_negative_weight": float(
            get_config_value("TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT", 2.0)
        ),
        "train_background_reduced_bucket_weight": float(
            get_config_value("TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT", 0.35)
        ),
        "train_background_hard_negative_buckets": list(
            get_config_value(
                "TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS",
                [
                    "UrbanSound8K_Clasificado/car_horn",
                    "UrbanSound8K_Clasificado/children_playing",
                    "UrbanSound8K_Clasificado/drilling",
                    "UrbanSound8K_Clasificado/jackhammer",
                    "UrbanSound8K_Clasificado/street_music",
                ],
            )
        ),
        "train_background_reduced_buckets": list(
            get_config_value(
                "TRAIN_BACKGROUND_REDUCED_BUCKETS",
                [
                    "UrbanSound8K_Clasificado/air_conditioner",
                    "UrbanSound8K_Clasificado/dog_bark",
                    "UrbanSound8K_Clasificado/gun_shot",
                ],
            )
        ),
        "manifest_version": int(get_config_value("SPLIT_MANIFEST_VERSION", 3)),
    }
    payload_bytes = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload_bytes).hexdigest()[:12]
    return f"split_manifest_margin_3_v3_{digest}"


# ---------------------------------------------------------------------------
# Script de entrenamiento de una CNN para detectar sirenas en audio.
#
# Idea general del pipeline:
# 1. Leer cada audio y dividirlo en chunks temporales configurables.
# 2. Convertir cada chunk en una representacion espectral de tamano fijo.
# 3. Entrenar una CNN binaria para estimar la probabilidad de "sirena".
# 4. Evaluar el modelo chunk a chunk con metricas utiles para deteccion.
#
# El objetivo del proyecto no es dar una decision binaria final dentro de este
# script, sino obtener una probabilidad por chunk que luego pueda usarse en un
# sistema de streaming o en una interfaz externa.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Configuracion global del experimento
# ---------------------------------------------------------------------------
RANDOM_SEED = int(get_config_value("RANDOM_SEED", 42))
SAMPLE_RATE = int(get_config_value("SAMPLE_RATE", 16000))
CHUNK_LENGTH_S = float(get_config_value("CHUNK_LENGTH_S", 0.5))
OVERLAP_S = float(get_config_value("OVERLAP_S", 0.0))
CHUNK_STEP_S = CHUNK_LENGTH_S - OVERLAP_S
BATCH_SIZE = int(get_config_value("BATCH_SIZE", 4))
EPOCHS = int(get_config_value("EPOCHS", 50))
THRESHOLD_GRID = np.array(
    get_config_value("THRESHOLD_GRID", np.linspace(0.10, 0.95, 18).tolist()),
    dtype=np.float32,
)
TARGET_FALSE_ALARMS_PER_MIN = float(get_config_value("TARGET_FALSE_ALARMS_PER_MIN", 1.0))

# ---------------------------------------------------------------------------
# Opciones activables del entrenamiento y evaluacion
# ---------------------------------------------------------------------------
# Si es True, aplica pesos por clase para compensar desbalance en el entrenamiento.
# Si el numero de audios por clase ya esta equilibrado, puede dejarse en False.
USE_CLASS_WEIGHTS = bool(get_config_value("USE_CLASS_WEIGHTS", True))

# Si es True, aplica augmentacion a los chunks del conjunto de entrenamiento.
# Sirve para que la red vea ejemplos mas variados y generalice mejor.
USE_DATA_AUGMENTATION = bool(get_config_value("USE_DATA_AUGMENTATION", True))

# Probabilidad de aplicar augmentacion a cada chunk cuando la opcion anterior
# esta activada. El resto de chunks se usan originales para mezclar ejemplos
# limpios y augmentados dentro del mismo entrenamiento.
AUGMENTATION_APPLY_PROB = float(get_config_value("AUGMENTATION_APPLY_PROB", 0.5))

# Si es True, permite aplicar una coloracion espectral suave para simular
# cambios de microfono, colocacion o respuesta del habitaculo del coche.
USE_SPECTRAL_EQ_AUGMENTATION = bool(
    get_config_value("USE_SPECTRAL_EQ_AUGMENTATION", True)
)

# Probabilidad objetivo de aplicar la ecualizacion suave sobre el total de
# chunks de entrenamiento cuando la augmentacion esta activada.
EQ_AUGMENTATION_PROB = float(get_config_value("EQ_AUGMENTATION_PROB", 0.20))

# Probabilidad de usar una sola curva de EQ. El resto de casos usa 2 curvas
# anchas en zonas distintas del espectro.
EQ_ONE_FILTER_PROB = float(get_config_value("EQ_ONE_FILTER_PROB", 0.70))

# Ganancia maxima absoluta por shelf ancho.
EQ_SHELF_GAIN_DB_MAX = float(get_config_value("EQ_SHELF_GAIN_DB_MAX", 3.0))

# Ganancia maxima absoluta del bell cuando cae en la banda principal esperada
# de la sirena. Se limita mas que los shelves por seguridad.
EQ_BELL_GAIN_DB_MAX_SIREN_BAND = float(
    get_config_value("EQ_BELL_GAIN_DB_MAX_SIREN_BAND", 2.0)
)

# Limite absoluto de la coloracion total acumulada tras sumar 1-2 curvas.
EQ_TOTAL_GAIN_DB_LIMIT = float(get_config_value("EQ_TOTAL_GAIN_DB_LIMIT", 4.0))

# Rangos de frecuencias de las curvas suaves usadas por la EQ.
EQ_LOW_SHELF_CUTOFF_HZ_RANGE = tuple(
    get_config_value("EQ_LOW_SHELF_CUTOFF_HZ_RANGE", [150.0, 500.0])
)
EQ_BELL_CENTER_HZ_RANGE = tuple(
    get_config_value("EQ_BELL_CENTER_HZ_RANGE", [700.0, 2200.0])
)
EQ_HIGH_SHELF_CUTOFF_HZ_RANGE = tuple(
    get_config_value("EQ_HIGH_SHELF_CUTOFF_HZ_RANGE", [2500.0, 6000.0])
)

# Anchura de las curvas: campanas amplias en octavas y transiciones suaves
# para los shelves, evitando picos estrechos poco realistas.
EQ_BELL_BANDWIDTH_OCTAVES_RANGE = tuple(
    get_config_value("EQ_BELL_BANDWIDTH_OCTAVES_RANGE", [1.0, 1.8])
)
EQ_SHELF_SHARPNESS_RANGE = tuple(
    get_config_value("EQ_SHELF_SHARPNESS_RANGE", [2.0, 3.0])
)

# Controla el frontend espectral usado por la CNN:
# - "linear_stft": mantiene la representacion actual basada en STFT lineal.
# - "log_mel": usa un log-mel espectrograma de un solo canal.
SPECTRAL_FRONTEND = str(get_config_value("SPECTRAL_FRONTEND", "linear_stft"))

# Controla que representacion espectral entra en la CNN:
# - "harmonic": solo la componente armonica tras HPSS.
# - "full": espectrograma completo sin separar armonica/percusiva.
# - "harmonic_full": dos canales, uno armonico y otro completo.
# Esta opcion solo afecta a `SPECTRAL_FRONTEND = "linear_stft"`.
FEATURE_REPRESENTATION = str(get_config_value("FEATURE_REPRESENTATION", "harmonic_full"))

# Controla como se normaliza el espectrograma antes de entrar en la CNN:
# - "frequency": normalizacion por banda de frecuencia (modo actual del proyecto).
# - "minmax": replica la idea de Camilo, reescalando cada chunk a [0, 1].
# - "none": no aplica normalizacion adicional.
SPECTROGRAM_NORMALIZATION = str(get_config_value("SPECTROGRAM_NORMALIZATION", "minmax"))

# Si es True, muestra las graficas de loss, precision, recall, AUC-PR y F1.
# Puede desactivarse si se quiere ejecutar el script de forma mas automatizada.
SHOW_TRAINING_PLOTS = bool(get_config_value("SHOW_TRAINING_PLOTS", True))

# Si es True, busca un umbral de referencia para convertir probabilidades
# en decisiones binarias solo con fines de analisis.
USE_THRESHOLD_ANALYSIS = bool(get_config_value("USE_THRESHOLD_ANALYSIS", True))

# Si es True, guarda un fichero JSON con la configuracion de salida del modelo.
# Esto ayuda a documentar como debe integrarse la CNN en produccion.
SAVE_POSTPROCESSING_CONFIG = bool(get_config_value("SAVE_POSTPROCESSING_CONFIG", True))

# Si es True, guarda en un .txt las metricas y matrices de confusion de
# validacion y test para poder reutilizarlas en la memoria del proyecto.
SAVE_CONFUSION_REPORT = bool(get_config_value("SAVE_CONFUSION_REPORT", True))

# Si es True, el generador de entrenamiento intenta construir lotes con el
# mismo numero de chunks por clase. Solo se aplica a train, nunca a val/test.
USE_BALANCED_CHUNK_BATCHES = bool(get_config_value("USE_BALANCED_CHUNK_BATCHES", True))

# Numero objetivo de chunks por lote cuando se activa el balanceo anterior.
TRAIN_CHUNK_BATCH_SIZE = int(get_config_value("TRAIN_CHUNK_BATCH_SIZE", 64))

# Version del manifiesto persistido para invalidar reutilizaciones antiguas
# cuando cambie la logica de grouping/sampling del script.
SPLIT_MANIFEST_VERSION = int(get_config_value("SPLIT_MANIFEST_VERSION", 3))

# Reparto estable entre splits. Se usa para generar o validar el manifiesto.
SPLIT_TRAIN_FRACTION = float(get_config_value("SPLIT_TRAIN_FRACTION", 0.70))
SPLIT_VALIDATION_FRACTION = float(get_config_value("SPLIT_VALIDATION_FRACTION", 0.15))
SPLIT_TEST_FRACTION = float(get_config_value("SPLIT_TEST_FRACTION", 0.15))

# Columnas usadas para que train/validation/test mantengan proporciones mas
# comparables sin romper la agrupacion por `safe_group_id`.
SPLIT_STRATIFY_COLUMNS = tuple(
    get_config_value("SPLIT_STRATIFY_COLUMNS", ["label", "domain"])
)

# El split se optimiza por chunks, pero se penaliza tambien la desviacion en
# numero de audios para evitar subconjuntos muy raros.
SPLIT_WEIGHT_COLUMN = str(get_config_value("SPLIT_WEIGHT_COLUMN", "num_chunks"))
SPLIT_ROW_COST_WEIGHT = float(get_config_value("SPLIT_ROW_COST_WEIGHT", 0.15))
REUSE_SPLIT_MANIFEST = bool(get_config_value("REUSE_SPLIT_MANIFEST", True))
SAVE_SPLIT_MANIFEST = bool(get_config_value("SAVE_SPLIT_MANIFEST", True))
SPLIT_MANIFEST_BASENAME = str(
    get_config_value(
        "SPLIT_MANIFEST_BASENAME",
        build_default_split_manifest_basename(),
    )
)

# Submuestreo opcional de backgrounds solo en train. Validation y test se
# mantienen completos para medir falsas alarmas en una distribucion realista.
APPLY_TRAIN_BACKGROUND_SUBSAMPLING = bool(
    get_config_value("APPLY_TRAIN_BACKGROUND_SUBSAMPLING", True)
)
TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO = float(
    get_config_value("TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO", 1.0)
)
TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET = int(
    get_config_value("TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET", 1)
)
TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT", 1.0)
)
TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT", 2.0)
)
TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT", 0.35)
)
TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS = tuple(
    get_config_value(
        "TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS",
        [
            "UrbanSound8K_Clasificado/car_horn",
            "UrbanSound8K_Clasificado/children_playing",
            "UrbanSound8K_Clasificado/drilling",
            "UrbanSound8K_Clasificado/jackhammer",
            "UrbanSound8K_Clasificado/street_music",
        ],
    )
)
TRAIN_BACKGROUND_REDUCED_BUCKETS = tuple(
    get_config_value(
        "TRAIN_BACKGROUND_REDUCED_BUCKETS",
        [
            "UrbanSound8K_Clasificado/air_conditioner",
            "UrbanSound8K_Clasificado/dog_bark",
            "UrbanSound8K_Clasificado/gun_shot",
        ],
    )
)

# La CNN se hace algo mas ancha que la version inicial para reducir
# infraajuste, manteniendo un coste razonable para CPU.
CONV_FILTERS = tuple(get_config_value("CONV_FILTERS", [16, 32, 64]))
DENSE_UNITS = int(get_config_value("DENSE_UNITS", 32))

# Parametros del frontend espectral.
N_FFT = int(get_config_value("N_FFT", 1024))
HOP_LENGTH = int(get_config_value("HOP_LENGTH", 512))
LINEAR_FREQ_BINS = int(get_config_value("LINEAR_FREQ_BINS", 359))
STFT_WINDOW = str(get_config_value("STFT_WINDOW", "hamming"))
HPSS_MARGIN = float(get_config_value("HPSS_MARGIN", 3.0))


def get_default_time_frames():
    """
    Devuelve una anchura temporal coherente con la duracion real del chunk.

    Para 0.5 s conserva el comportamiento historico del proyecto (17 frames),
    pero al subir a 1.0 s amplía la entrada para que la CNN vea todo el
    contexto temporal y no solo el primer medio segundo.
    """
    chunk_samples = int(round(CHUNK_LENGTH_S * SAMPLE_RATE))
    return max(1, int(np.ceil(chunk_samples / HOP_LENGTH)) + 1)


TIME_FRAMES = int(get_config_value("TIME_FRAMES", get_default_time_frames()))
MEL_BINS = int(get_config_value("MEL_BINS", 128))
PADDED_CHUNK_SAMPLES = int(
    max(
        int(round(CHUNK_LENGTH_S * SAMPLE_RATE)),
        N_FFT,
        HOP_LENGTH * max(0, TIME_FRAMES - 1),
    )
)


def validate_split_configuration():
    """Valida la configuracion global del manifiesto y del split."""
    split_total = SPLIT_TRAIN_FRACTION + SPLIT_VALIDATION_FRACTION + SPLIT_TEST_FRACTION
    if not np.isclose(split_total, 1.0, atol=1e-6):
        raise ValueError(
            "Las fracciones de train/validation/test deben sumar 1.0. "
            f"Valor actual: {split_total:.6f}."
        )

    for split_name, split_value in (
        ("train", SPLIT_TRAIN_FRACTION),
        ("validation", SPLIT_VALIDATION_FRACTION),
        ("test", SPLIT_TEST_FRACTION),
    ):
        if split_value <= 0.0 or split_value >= 1.0:
            raise ValueError(
                f"La fraccion del split {split_name} debe estar en (0, 1). "
                f"Valor actual: {split_value}."
            )

    if TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO <= 0.0:
        raise ValueError(
            "TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO debe ser positivo para "
            "conservar negativos en entrenamiento."
        )

    if TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET < 0:
        raise ValueError(
            "TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET no puede ser negativo."
        )


validate_split_configuration()

# ---------------------------------------------------------------------------
# Paralelismo para exprimir mejor la CPU durante el entrenamiento.
#
# En este proyecto hay dos focos de carga:
# - TensorFlow: convoluciones, pooling y capas densas.
# - Python/librosa: carga de audios, STFT, HPSS y construccion de batches.
#
# Como aqui se entrena en CPU, repartir los hilos suele rendir mejor que dar
# todos los cores a TensorFlow y dejar el generador en serie.
# ---------------------------------------------------------------------------
LOGICAL_CPU_COUNT = max(1, os.cpu_count() or 1)
PYDATASET_WORKERS = int(
    get_config_value("PYDATASET_WORKERS", max(1, min(8, LOGICAL_CPU_COUNT // 4)))
)
TF_INTER_OP_THREADS = int(
    get_config_value("TF_INTER_OP_THREADS", 2 if LOGICAL_CPU_COUNT >= 8 else 1)
)
TF_INTRA_OP_THREADS = int(
    get_config_value("TF_INTRA_OP_THREADS", max(1, LOGICAL_CPU_COUNT - PYDATASET_WORKERS))
)
PYDATASET_USE_MULTIPROCESSING = bool(
    get_config_value("PYDATASET_USE_MULTIPROCESSING", False)
)
PYDATASET_MAX_QUEUE_SIZE = int(
    get_config_value("PYDATASET_MAX_QUEUE_SIZE", max(10, PYDATASET_WORKERS * 4))
)


def get_linear_num_input_channels():
    """Numero de canales de entrada para el frontend STFT lineal."""
    if FEATURE_REPRESENTATION in {"harmonic", "full"}:
        return 1
    if FEATURE_REPRESENTATION == "harmonic_full":
        return 2
    raise ValueError(
        "FEATURE_REPRESENTATION debe ser 'harmonic', 'full' o 'harmonic_full'."
    )


def get_input_shape():
    """Forma de entrada de la CNN segun el frontend espectral configurado."""
    if SPECTRAL_FRONTEND == "linear_stft":
        return (LINEAR_FREQ_BINS, TIME_FRAMES, get_linear_num_input_channels())
    if SPECTRAL_FRONTEND == "log_mel":
        return (MEL_BINS, TIME_FRAMES, 1)
    raise ValueError(
        "SPECTRAL_FRONTEND debe ser 'linear_stft' o 'log_mel'."
    )


INPUT_SHAPE = get_input_shape()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata", "master_index.csv")
RUN_OUTPUT_DIR = os.path.normpath(str(get_config_value("RUN_OUTPUT_DIR", SCRIPT_DIR)))
RUN_NAME_PREFIX = str(get_config_value("RUN_NAME_PREFIX", "modelo_sirenas_margin_3"))
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_BASENAME = f"{RUN_NAME_PREFIX}_{RUN_TIMESTAMP}"
MODEL_PATH = os.path.join(RUN_OUTPUT_DIR, f"{RUN_BASENAME}.keras")
POSTPROCESSING_PATH = os.path.join(RUN_OUTPUT_DIR, f"{RUN_BASENAME}_postprocesado.json")
CONFUSION_REPORT_PATH = os.path.join(RUN_OUTPUT_DIR, f"{RUN_BASENAME}_matrices_confusion.txt")
SPLIT_MANIFEST_PATH = os.path.join(
    DATASET_DIR,
    "metadata",
    f"{SPLIT_MANIFEST_BASENAME}.csv",
)
SPLIT_MANIFEST_INFO_PATH = os.path.join(
    DATASET_DIR,
    "metadata",
    f"{SPLIT_MANIFEST_BASENAME}_info.json",
)


def configure_tensorflow_cpu_runtime():
    """
    Configura el reparto de hilos de TensorFlow antes de inicializar el runtime.

    - intra_op: paralelismo dentro de operaciones grandes.
    - inter_op: operaciones independientes ejecutandose en paralelo.
    """
    try:
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
    except RuntimeError as exc:
        print(
            "Aviso: no se ha podido fijar el paralelismo de TensorFlow porque "
            f"el runtime ya estaba inicializado ({exc})."
        )


def print_parallelism_configuration():
    """Muestra por pantalla como se reparte la CPU en este experimento."""
    print(
        "Configuracion de CPU -> hilos logicos: {logical} | "
        "TF intra_op: {intra} | TF inter_op: {inter} | "
        "PyDataset workers: {workers} | multiprocessing: {multiprocessing} | "
        "cola maxima: {queue}".format(
            logical=LOGICAL_CPU_COUNT,
            intra=TF_INTRA_OP_THREADS,
            inter=TF_INTER_OP_THREADS,
            workers=PYDATASET_WORKERS,
            multiprocessing=PYDATASET_USE_MULTIPROCESSING,
            queue=PYDATASET_MAX_QUEUE_SIZE,
        )
    )


configure_tensorflow_cpu_runtime()
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@register_keras_serializable()
class BinaryF1Score(tf.keras.metrics.Metric):
    """
    Implementacion propia de F1 para poder verla durante el entrenamiento.

    Keras no siempre ofrece F1 binario listo para usar en todos los entornos,
    asi que se calcula a partir de TP, FP y FN acumulados en cada epoca.
    """

    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, tf.float32)

        weights = tf.ones_like(y_true, dtype=tf.float32)
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        tp = tf.reduce_sum(weights * y_true * y_pred)
        fp = tf.reduce_sum(weights * (1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(weights * y_true * (1.0 - y_pred))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def pad_or_trim(audio, target_length):
    """Ajusta un vector de audio a una longitud exacta."""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    if len(audio) > target_length:
        return audio[:target_length]
    return audio


def add_shaped_noise(audio, rng):
    """
    Genera un ruido sintetico suavizado para simular fondo continuo tipo trafico.
    """
    noise = rng.normal(0.0, 1.0, len(audio) + 64).astype(np.float32)
    kernel = np.exp(-np.linspace(0.0, 3.5, 65)).astype(np.float32)
    kernel /= np.sum(kernel)
    traffic_like_noise = np.convolve(noise, kernel, mode="valid")[: len(audio)]
    traffic_like_noise /= np.max(np.abs(traffic_like_noise)) + 1e-6

    signal_rms = np.sqrt(np.mean(np.square(audio)) + 1e-8)
    noise_rms = signal_rms / (10.0 ** (rng.uniform(12.0, 24.0) / 20.0))
    return audio + traffic_like_noise * noise_rms


def add_reverb(audio, rng):
    """Aplica una reverberacion sintetica simple para simular entornos reales."""
    ir_length = int(rng.integers(32, 128))
    impulse = rng.normal(0.0, 1.0, ir_length).astype(np.float32)
    impulse *= np.exp(-np.linspace(0.0, rng.uniform(2.0, 4.5), ir_length)).astype(np.float32)
    impulse[0] += 1.0

    reverberated = np.convolve(audio, impulse, mode="full")[: len(audio)]
    reverberated = reverberated.astype(np.float32)
    peak = np.max(np.abs(reverberated)) + 1e-6
    reference_peak = max(np.max(np.abs(audio)), 1e-3)
    return (reverberated / peak) * reference_peak


def apply_compression(audio, rng):
    """Simula una compresion o saturacion ligera del audio."""
    drive = rng.uniform(1.0, 1.6)
    return np.tanh(drive * audio).astype(np.float32)


def build_low_shelf_curve_db(freqs_hz, cutoff_hz, gain_db, sharpness):
    """Curva suave de low shelf en dB, sin transiciones estrechas."""
    safe_freqs = np.maximum(freqs_hz, 1.0)
    return gain_db / (1.0 + np.power(safe_freqs / max(cutoff_hz, 1.0), sharpness))


def build_high_shelf_curve_db(freqs_hz, cutoff_hz, gain_db, sharpness):
    """Curva suave de high shelf en dB, simetrica respecto al low shelf."""
    safe_freqs = np.maximum(freqs_hz, 1.0)
    low_component = 1.0 / (1.0 + np.power(safe_freqs / max(cutoff_hz, 1.0), sharpness))
    return gain_db * (1.0 - low_component)


def build_bell_curve_db(freqs_hz, center_hz, gain_db, bandwidth_octaves):
    """Campana ancha en dominio log-frecuencia para evitar resonancias estrechas."""
    safe_freqs = np.maximum(freqs_hz, 1.0)
    log_distance = (
        np.log2(safe_freqs) - np.log2(max(center_hz, 1.0))
    ) / max(bandwidth_octaves, 1e-3)
    return gain_db * np.exp(-0.5 * np.square(log_distance))


def apply_random_spectral_eq(audio, sr, rng):
    """
    Aplica una coloracion espectral suave y controlada.

    Reglas de seguridad aplicadas:
    - probabilidad moderada
    - 1 o 2 curvas maximo
    - filtros anchos, nunca estrechos
    - shelves limitados a +/-3 dB
    - bell en banda principal de sirena limitado a +/-2 dB
    - coloracion total final recortada a +/-4 dB acumulados
    """
    freqs_hz = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total_curve_db = np.zeros_like(freqs_hz, dtype=np.float32)

    num_filters = 1 if rng.random() < EQ_ONE_FILTER_PROB else 2
    filter_types = rng.choice(
        np.array(["low_shelf", "bell", "high_shelf"], dtype=object),
        size=num_filters,
        replace=False,
    )

    for filter_type in np.atleast_1d(filter_types):
        if filter_type == "low_shelf":
            cutoff_hz = rng.uniform(*EQ_LOW_SHELF_CUTOFF_HZ_RANGE)
            gain_db = rng.uniform(-EQ_SHELF_GAIN_DB_MAX, EQ_SHELF_GAIN_DB_MAX)
            sharpness = rng.uniform(*EQ_SHELF_SHARPNESS_RANGE)
            curve_db = build_low_shelf_curve_db(freqs_hz, cutoff_hz, gain_db, sharpness)
        elif filter_type == "high_shelf":
            cutoff_hz = rng.uniform(*EQ_HIGH_SHELF_CUTOFF_HZ_RANGE)
            gain_db = rng.uniform(-EQ_SHELF_GAIN_DB_MAX, EQ_SHELF_GAIN_DB_MAX)
            sharpness = rng.uniform(*EQ_SHELF_SHARPNESS_RANGE)
            curve_db = build_high_shelf_curve_db(freqs_hz, cutoff_hz, gain_db, sharpness)
        else:
            center_hz = rng.uniform(*EQ_BELL_CENTER_HZ_RANGE)
            gain_db = rng.uniform(
                -EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
                EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
            )
            bandwidth_octaves = rng.uniform(*EQ_BELL_BANDWIDTH_OCTAVES_RANGE)
            curve_db = build_bell_curve_db(
                freqs_hz,
                center_hz,
                gain_db,
                bandwidth_octaves,
            )

        total_curve_db += curve_db.astype(np.float32)

    total_curve_db = np.clip(
        total_curve_db,
        -EQ_TOTAL_GAIN_DB_LIMIT,
        EQ_TOTAL_GAIN_DB_LIMIT,
    )

    spectrum = np.fft.rfft(audio.astype(np.float32))
    eq_gain = np.power(10.0, total_curve_db / 20.0).astype(np.float32)
    equalized = np.fft.irfft(spectrum * eq_gain, n=len(audio)).astype(np.float32)

    # La EQ debe cambiar el color espectral, no el nivel medio del chunk.
    original_rms = np.sqrt(np.mean(np.square(audio)) + 1e-8)
    equalized_rms = np.sqrt(np.mean(np.square(equalized)) + 1e-8)
    equalized *= original_rms / max(equalized_rms, 1e-8)
    return equalized.astype(np.float32)


def get_effective_eq_apply_probability():
    """
    Ajusta la probabilidad condicionada de la EQ para respetar la probabilidad
    objetivo sobre el total de chunks de entrenamiento.
    """
    if AUGMENTATION_APPLY_PROB <= 0.0:
        return 0.0
    return min(1.0, EQ_AUGMENTATION_PROB / AUGMENTATION_APPLY_PROB)


def augment_audio_chunk(audio_chunk, sr, rng):
    """
    Crea variaciones realistas de un chunk solo para entrenamiento.

    La idea es exponer la red a pequenas variaciones de volumen, ruido,
    reverberacion y afinacion para que generalice mejor fuera del dataset.

    Orden aplicado:
    1. Ganancia
    2. Time stretch
    3. Pitch shift (solo con frontend STFT lineal)
    4. Compresion/saturacion ligera
    5. Reverb
    6. Ruido
    7. EQ suave

    Este orden intenta separar:
    - cambios de la fuente o de la senal base
    - propagacion/reflexiones
    - mezcla con el entorno
    - coloracion final de canal/captura

    Nota:
    - El `pitch_shift` solo se aplica con `linear_stft`.
    - Con `log_mel` se desactiva para no mezclar la invariancia propia del
      frontend logaritmico con una perturbacion extra del tono.
    - La EQ suave se usa como simulacion de coloracion de canal, no como efecto
      creativo, y se limita para no superar +/-4 dB acumulados.
    """
    augmented = np.copy(audio_chunk).astype(np.float32)

    if rng.random() < 0.8:
        augmented *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.20:
        augmented = librosa.effects.time_stretch(augmented, rate=rng.uniform(0.97, 1.03))
    if SPECTRAL_FRONTEND == "linear_stft" and rng.random() < 0.20:
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=sr,
            n_steps=rng.uniform(-0.35, 0.35),
        )
    if rng.random() < 0.20:
        augmented = apply_compression(augmented, rng)
    if rng.random() < 0.20:
        augmented = add_reverb(augmented, rng)
    if rng.random() < 0.35:
        augmented = add_shaped_noise(augmented, rng)
    if (
        USE_SPECTRAL_EQ_AUGMENTATION
        and rng.random() < get_effective_eq_apply_probability()
    ):
        augmented = apply_random_spectral_eq(augmented, sr, rng)

    augmented = pad_or_trim(augmented, len(audio_chunk)).astype(np.float32)
    peak = np.max(np.abs(augmented))
    if peak > 1.0:
        augmented /= peak
    return augmented


def normalize_spectrogram(db_spectrogram, mode=SPECTROGRAM_NORMALIZATION):
    """
    Normaliza un espectrograma en dB segun la estrategia configurada.

    - `frequency`: reduce diferencias de nivel entre grabaciones banda a banda.
    - `minmax`: replica el reescalado [0, 1] del modelo compartido.
    - `none`: conserva la escala en dB sin cambios extra.
    """
    if mode == "frequency":
        freq_mean = np.mean(db_spectrogram, axis=1, keepdims=True)
        freq_std = np.std(db_spectrogram, axis=1, keepdims=True)
        normalized = (db_spectrogram - freq_mean) / (freq_std + 1e-6)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    if mode == "minmax":
        min_db = np.min(db_spectrogram)
        max_db = np.max(db_spectrogram)
        if max_db - min_db > 0:
            normalized = (db_spectrogram - min_db) / (max_db - min_db)
        else:
            normalized = db_spectrogram - min_db
        return normalized.astype(np.float32)

    if mode == "none":
        return db_spectrogram.astype(np.float32)

    raise ValueError(
        "SPECTROGRAM_NORMALIZATION debe ser 'frequency', 'minmax' o 'none'."
    )


def pad_or_trim_time_frames(feature_map, target_frames=TIME_FRAMES):
    """Ajusta el numero de frames temporales de una representacion 2D."""
    current_frames = feature_map.shape[1]
    if current_frames < target_frames:
        return np.pad(feature_map, ((0, 0), (0, target_frames - current_frames)))
    if current_frames > target_frames:
        return feature_map[:, :target_frames]
    return feature_map


def build_feature_tensor_from_linear_stft(stft_matrix):
    """
    Construye la entrada final de la CNN a partir de una STFT lineal.

    Segun `FEATURE_REPRESENTATION`, devuelve:
    - un canal armonico
    - un canal completo
    - o ambos canales apilados
    """
    full_sliced = stft_matrix[:LINEAR_FREQ_BINS, :TIME_FRAMES]
    harmonic, _ = librosa.decompose.hpss(stft_matrix, margin=HPSS_MARGIN)
    harmonic_sliced = harmonic[:LINEAR_FREQ_BINS, :TIME_FRAMES]

    full_db = librosa.amplitude_to_db(np.abs(full_sliced), ref=np.max)
    harmonic_db = librosa.amplitude_to_db(np.abs(harmonic_sliced), ref=np.max)

    full_db = normalize_spectrogram(full_db, mode=SPECTROGRAM_NORMALIZATION)
    harmonic_db = normalize_spectrogram(harmonic_db, mode=SPECTROGRAM_NORMALIZATION)

    if FEATURE_REPRESENTATION == "harmonic":
        return np.expand_dims(harmonic_db, axis=-1)
    if FEATURE_REPRESENTATION == "full":
        return np.expand_dims(full_db, axis=-1)
    if FEATURE_REPRESENTATION == "harmonic_full":
        return np.stack([harmonic_db, full_db], axis=-1).astype(np.float32)

    raise ValueError(
        "FEATURE_REPRESENTATION debe ser 'harmonic', 'full' o 'harmonic_full'."
    )


def build_feature_tensor_from_audio_chunk(audio_chunk_padded, sr=SAMPLE_RATE):
    """
    Construye la entrada final de la CNN a partir de un chunk de audio.

    Segun `SPECTRAL_FRONTEND`, puede devolver:
    - una representacion STFT lineal (actual del proyecto)
    - un log-mel espectrograma de un solo canal
    """
    if SPECTRAL_FRONTEND == "linear_stft":
        stft = librosa.stft(
            audio_chunk_padded,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=STFT_WINDOW,
        )
        return build_feature_tensor_from_linear_stft(stft)

    if SPECTRAL_FRONTEND == "log_mel":
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_chunk_padded,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=MEL_BINS,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_db = pad_or_trim_time_frames(mel_db, target_frames=TIME_FRAMES)
        mel_db = normalize_spectrogram(mel_db, mode=SPECTROGRAM_NORMALIZATION)
        return np.expand_dims(mel_db, axis=-1).astype(np.float32)

    raise ValueError(
        "SPECTRAL_FRONTEND debe ser 'linear_stft' o 'log_mel'."
    )


def extract_features_chunks(
    audio_path,
    sr=SAMPLE_RATE,
    chunk_length_s=CHUNK_LENGTH_S,
    overlap_s=OVERLAP_S,
    augment=False,
    rng=None,
):
    """
    Extrae chunks de audio y los convierte en entradas para la CNN.

    Flujo por chunk:
    1. Cortar una ventana temporal fija.
    2. Aplicar augmentacion opcional si estamos en entrenamiento y el chunk
       cae dentro de la probabilidad configurada.
    3. Calcular el frontend espectral elegido.
    4. Construir la representacion final para la CNN.
    5. Pasar a dB, normalizar y dar formato final.
    """
    rng = rng if rng is not None else np.random.default_rng()

    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)

        chunk_samples = int(chunk_length_s * sr)
        step_samples = int((chunk_length_s - overlap_s) * sr)
        chunks = []

        for start in range(0, max(1, len(audio) - chunk_samples + 1), step_samples):
            audio_chunk = audio[start : start + chunk_samples]

            if len(audio_chunk) < chunk_samples:
                continue

            # La augmentacion solo se usa durante entrenamiento y se aplica
            # con una probabilidad menor que 1 para mezclar chunks originales
            # y chunks modificados on the fly.
            if augment and rng.random() < AUGMENTATION_APPLY_PROB:
                audio_chunk = augment_audio_chunk(audio_chunk, sr, rng)

            # Se ajusta el chunk a la longitud temporal esperada por la CNN.
            # Asi 0.5 s mantiene 17 frames, mientras que 1.0 s puede usar una
            # anchura mayor sin truncar la mitad del contexto temporal.
            audio_chunk_padded = pad_or_trim(
                audio_chunk,
                PADDED_CHUNK_SAMPLES,
            ).astype(np.float32)

            # A partir del chunk temporal se genera la entrada final para la CNN.
            features = build_feature_tensor_from_audio_chunk(audio_chunk_padded, sr=sr)
            chunks.append(features)

        return np.asarray(chunks, dtype=np.float32)

    except Exception as exc:
        print(f"Error procesando {audio_path}: {exc}")
        return None


def count_valid_chunks_in_audio(
    audio_path,
    sr=SAMPLE_RATE,
    chunk_length_s=CHUNK_LENGTH_S,
    overlap_s=OVERLAP_S,
):
    """
    Cuenta cuantos chunks validos generaria un audio sin calcular espectrogramas.

    Se usa para estimar el peso real de cada clase a nivel de chunk, que es la
    unidad con la que entrena la CNN.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        chunk_samples = int(chunk_length_s * sr)
        step_samples = int((chunk_length_s - overlap_s) * sr)

        if len(audio) < chunk_samples or step_samples <= 0:
            return 0

        return len(range(0, max(1, len(audio) - chunk_samples + 1), step_samples))

    except Exception as exc:
        print(f"Error contando chunks en {audio_path}: {exc}")
        return 0


def compute_class_sample_weights_from_chunks(
    df,
    base_path=DATASET_DIR,
    sr=SAMPLE_RATE,
    chunk_length_s=CHUNK_LENGTH_S,
    overlap_s=OVERLAP_S,
):
    """
    Calcula pesos inversamente proporcionales al numero real de chunks por clase.

    Procedimiento:
    1. Recorrer los audios del conjunto de entrenamiento.
    2. Contar cuantos chunks validos genera cada audio.
    3. Sumar esos chunks por clase.
    4. Calcular los pesos a partir de esos totales.
    """
    if "num_chunks" in df.columns:
        class_chunk_counts = (
            df.groupby("target")["num_chunks"].sum().astype(int).to_dict()
        )
    else:
        class_chunk_counts = {}

        for _, row in df.iterrows():
            audio_path = os.path.normpath(os.path.join(base_path, row["path"]))
            num_chunks = count_valid_chunks_in_audio(
                audio_path,
                sr=sr,
                chunk_length_s=chunk_length_s,
                overlap_s=overlap_s,
            )

            class_id = int(row["target"])
            class_chunk_counts[class_id] = class_chunk_counts.get(class_id, 0) + num_chunks

    total_chunks = sum(class_chunk_counts.values())
    num_classes = len(class_chunk_counts)

    if total_chunks == 0 or num_classes == 0:
        raise RuntimeError(
            "No se han podido contar chunks validos para calcular class weights. "
            "Revisa las rutas de audio, la duracion de los archivos y la configuracion temporal."
        )

    weights = {}

    for class_id, count in sorted(class_chunk_counts.items()):
        if count > 0:
            weights[class_id] = float(total_chunks / (num_classes * count))

    return weights, class_chunk_counts


def get_path_source(path_value):
    """
    Extrae la fuente/dataset de un path tipo `raw/siren/audioset/audio.wav`.

    En este proyecto suele corresponder al tercer componente de la ruta.
    """
    if pd.isna(path_value):
        return "missing"

    normalized_path = str(path_value).replace("\\", "/")
    path_parts = [part for part in normalized_path.split("/") if part]

    if len(path_parts) >= 3:
        return path_parts[2]
    if len(path_parts) >= 1:
        return path_parts[-1]
    return "missing"


def normalize_multichannel_group_id(group_id_value):
    """Elimina sufijos de canal/microfono para agrupar una misma escena."""
    if pd.isna(group_id_value):
        return ""
    normalized_value = str(group_id_value).strip()
    if not normalized_value:
        return ""
    return re.sub(r"(?i)([-_](?:ch|mic)\d+)$", "", normalized_value)


def infer_background_group_id_from_row(row):
    """
    Reconstruye un identificador de escena para backgrounds.

    Casos especiales:
    - A3S multicanal: agrupa `n-...-ch1/ch2/...` en una misma escena base.
    - UrbanSound8K_Clasificado: agrupa subsegmentos `...-0`, `...-1`, etc.
    """
    candidate_value = row.get("group_id", pd.NA)
    if pd.isna(candidate_value) or not str(candidate_value).strip():
        path_value = row.get("path", pd.NA)
        if pd.isna(path_value):
            return pd.NA
        normalized_path = str(path_value).replace("\\", "/")
        candidate_value = os.path.splitext(os.path.basename(normalized_path))[0]

    normalized_value = normalize_multichannel_group_id(candidate_value)
    if not normalized_value:
        return pd.NA

    source_value = str(row.get("source", "")).strip().lower()
    if source_value == "urbansound8k_clasificado":
        normalized_value = re.sub(r"-(\d+)$", "", normalized_value)

    return normalized_value or pd.NA


def infer_siren_id_from_row(row):
    """
    Reconstruye un identificador de evento de sirena cuando el CSV no trae
    `siren_id`.

    Para grabaciones multicanal como `s-20210506-1652-ch1`, agrupa los canales
    bajo el mismo evento base (`s-20210506-1652`) y evita leakage entre splits.
    """
    label_value = str(row.get("label", "")).strip().lower()
    if label_value not in {"siren", "sirena"}:
        return pd.NA

    candidate_value = row.get("group_id", pd.NA)
    if pd.isna(candidate_value) or not str(candidate_value).strip():
        path_value = row.get("path", pd.NA)
        if pd.isna(path_value):
            return pd.NA
        normalized_path = str(path_value).replace("\\", "/")
        candidate_value = os.path.splitext(os.path.basename(normalized_path))[0]

    normalized_value = str(candidate_value).strip()
    if not normalized_value:
        return pd.NA

    return re.sub(r"(?i)([-_](?:ch|mic)\d+)$", "", normalized_value)


def get_background_subclass_from_row(row):
    """Devuelve la subcarpeta relevante del background cuando aporta contexto."""
    label_value = str(row.get("label", "")).strip().lower()
    if label_value != "background":
        return pd.NA

    source_value = str(row.get("source", "")).strip()
    if source_value != "UrbanSound8K_Clasificado":
        return pd.NA

    path_value = row.get("path", pd.NA)
    if pd.isna(path_value):
        return pd.NA

    normalized_path = str(path_value).replace("\\", "/")
    path_parts = [part for part in normalized_path.split("/") if part]
    if len(path_parts) >= 4:
        return path_parts[3]
    return pd.NA


def get_background_sampling_bucket_from_row(row):
    """
    Define el bucket de muestreo de negatives usado solo en train.

    - En UrbanSound8K_Clasificado se baja al nivel de subcarpeta para poder
      priorizar hard negatives concretos.
    - En el resto de fuentes se trabaja a nivel `source`.
    """
    label_value = str(row.get("label", "")).strip().lower()
    if label_value != "background":
        return pd.NA

    source_value = str(row.get("source", "missing")).strip() or "missing"
    subclass_value = row.get("background_subclass", pd.NA)
    if pd.notna(subclass_value) and str(subclass_value).strip():
        return f"{source_value}/{str(subclass_value).strip()}"
    return source_value


def normalize_scene_base_id(group_id_value):
    """
    Normaliza el identificador de escena para compartir split entre etiquetas.

    Ejemplo:
    - `s-20210506-1652` y `n-20210506-1652` se consideran la misma escena.
    """
    normalized_value = str(group_id_value).strip()
    if not normalized_value:
        return "missing"
    return re.sub(r"^(?:s|n)-(?=\d)", "", normalized_value, flags=re.IGNORECASE)


def build_safe_group_id(row):
    """
    Crea un identificador de grupo globalmente unico y anti-leakage.

    La fuente se conserva en la clave, pero se intenta compartir escena entre
    siren/background cuando el identificador base apunta al mismo evento.
    """
    source_value = str(row.get("source", "missing")).strip() or "missing"
    label_value = str(row.get("label", "")).strip().lower() or "missing"

    if label_value in {"siren", "sirena"}:
        base_group_id = row.get("siren_id", pd.NA)
        if pd.isna(base_group_id) or not str(base_group_id).strip():
            base_group_id = infer_siren_id_from_row(row)
    else:
        base_group_id = infer_background_group_id_from_row(row)

    if pd.isna(base_group_id) or not str(base_group_id).strip():
        raw_group_id = row.get("group_id", pd.NA)
        if pd.notna(raw_group_id) and str(raw_group_id).strip():
            base_group_id = str(raw_group_id).strip()
        else:
            path_value = row.get("path", pd.NA)
            if pd.notna(path_value):
                normalized_path = str(path_value).replace("\\", "/")
                base_group_id = os.path.splitext(os.path.basename(normalized_path))[0]
            else:
                base_group_id = "missing"

    normalized_scene_id = normalize_scene_base_id(base_group_id)
    return f"{source_value}|{normalized_scene_id}"


def enrich_metadata_columns(df):
    """
    Anade columnas auxiliares derivadas de `path` cuando el CSV no las incluye.

    - `source`: dataset o procedencia del audio.
    - `domain`: proxy del dominio acustico. Si no existe en metadata, se toma
      igual que `source` para poder estratificar sin romper el script.
    - `siren_id`: identificador de evento de sirena; si falta en el CSV se
      deriva desde `group_id`/`path`.
    - `background_subclass`: subcarpeta relevante para negatives de UrbanSound.
    - `background_sampling_bucket`: bucket usado para priorizar train.
    - `safe_group_id`: grupo robusto frente a canales y subsegmentos.
    """
    df_local = df.copy()

    if "source" not in df_local.columns:
        df_local["source"] = df_local["path"].apply(get_path_source)

    if "domain" not in df_local.columns:
        df_local["domain"] = df_local["source"]

    if "siren_id" not in df_local.columns:
        df_local["siren_id"] = pd.NA

    missing_siren_id_mask = df_local["siren_id"].isna() | (
        df_local["siren_id"].astype(str).str.strip() == ""
    )
    if missing_siren_id_mask.any():
        df_local.loc[missing_siren_id_mask, "siren_id"] = df_local.loc[
            missing_siren_id_mask
        ].apply(infer_siren_id_from_row, axis=1)

    df_local["background_subclass"] = df_local.apply(get_background_subclass_from_row, axis=1)
    df_local["background_sampling_bucket"] = df_local.apply(
        get_background_sampling_bucket_from_row,
        axis=1,
    )
    df_local["safe_group_id"] = df_local.apply(build_safe_group_id, axis=1)

    return df_local


def resolve_stratify_columns(df, requested_columns, fallback_columns=("label",)):
    """
    Devuelve las columnas de estratificacion realmente disponibles en el DataFrame.

    Si alguna columna pedida no existe, se informa y se intenta usar una
    alternativa razonable antes de abortar.
    """
    available_columns = [column for column in requested_columns if column in df.columns]
    missing_columns = [column for column in requested_columns if column not in df.columns]

    if missing_columns:
        print(
            "Aviso: faltan columnas de estratificacion en metadata: "
            f"{', '.join(missing_columns)}"
        )

    if available_columns:
        return tuple(available_columns)

    fallback_available_columns = [
        column for column in fallback_columns if column in df.columns
    ]
    if fallback_available_columns:
        print(
            "Aviso: se usaran columnas alternativas para estratificar: "
            f"{', '.join(fallback_available_columns)}"
        )
        return tuple(fallback_available_columns)

    raise RuntimeError(
        "No hay columnas validas para estratificar los splits. "
        "Revisa la metadata o la configuracion de SPLIT_STRATIFY_COLUMNS."
    )


def make_stratum_keys(df, columns):
    """Combina varias columnas categoricas en una clave de estratificacion."""
    if not columns:
        return pd.Series(["global"] * len(df), index=df.index)
    return df.loc[:, columns].fillna("missing").astype(str).agg("|".join, axis=1)


def split_assignment_cost(
    current_counts,
    current_weight,
    current_rows,
    target_counts,
    target_weight,
    target_rows,
    row_cost_weight=SPLIT_ROW_COST_WEIGHT,
):
    """Mide lo lejos que esta un split de sus proporciones objetivo."""
    weight_error = ((current_weight - target_weight) / max(1.0, target_weight)) ** 2
    row_error = ((current_rows - target_rows) / max(1.0, target_rows)) ** 2
    stratum_error = 0.0

    for key, target_value in target_counts.items():
        current_value = current_counts.get(key, 0)
        stratum_error += ((current_value - target_value) / max(1.0, target_value)) ** 2

    return weight_error + (row_cost_weight * row_error) + stratum_error


def grouped_stratified_split(
    df,
    group_col,
    test_size,
    stratify_columns=SPLIT_STRATIFY_COLUMNS,
    random_state=RANDOM_SEED,
    weight_col=SPLIT_WEIGHT_COLUMN,
    row_cost_weight=SPLIT_ROW_COST_WEIGHT,
):
    """
    Divide un DataFrame en dos subconjuntos manteniendo grupos completos y
    aproximando las proporciones de las columnas indicadas en `stratify_columns`.

    El objetivo principal se mide por peso en chunks (`weight_col`) porque la
    unidad real de entrenamiento/evaluacion de la CNN es el chunk, no el audio.
    """
    df_local = df.copy()
    df_local["_stratum_key"] = make_stratum_keys(df_local, list(stratify_columns))
    df_local["_split_weight"] = pd.to_numeric(
        df_local.get(weight_col, 0.0),
        errors="coerce",
    ).fillna(0.0)

    if float(df_local["_split_weight"].sum()) <= 0.0:
        df_local["_split_weight"] = 1.0

    total_weight = float(df_local["_split_weight"].sum())
    target_test_weight = total_weight * float(test_size)
    target_train_weight = total_weight - target_test_weight

    target_test_rows = len(df_local) * float(test_size)
    target_train_rows = len(df_local) - target_test_rows

    total_stratum_counts = df_local.groupby("_stratum_key")["_split_weight"].sum().to_dict()
    target_test_counts = {
        key: value * float(test_size) for key, value in total_stratum_counts.items()
    }
    target_train_counts = {
        key: value - target_test_counts[key] for key, value in total_stratum_counts.items()
    }

    rng = np.random.default_rng(random_state)
    group_summaries = []

    for group_id, group_df in df_local.groupby(group_col):
        group_summaries.append(
            {
                "group_id": group_id,
                "indices": group_df.index.to_numpy(),
                "size": len(group_df),
                "weight": float(group_df["_split_weight"].sum()),
                "counts": group_df.groupby("_stratum_key")["_split_weight"].sum().to_dict(),
                "tie_breaker": float(rng.random()),
            }
        )

    group_summaries.sort(
        key=lambda item: (-item["weight"], -item["size"], item["tie_breaker"])
    )

    train_indices = []
    test_indices = []
    train_weight = 0.0
    test_weight = 0.0
    train_rows = 0
    test_rows = 0
    train_counts = {}
    test_counts = {}

    for summary in group_summaries:
        candidate_train_weight = train_weight + summary["weight"]
        candidate_test_weight = test_weight + summary["weight"]
        candidate_train_rows = train_rows + summary["size"]
        candidate_test_rows = test_rows + summary["size"]

        candidate_train_counts = train_counts.copy()
        candidate_test_counts = test_counts.copy()
        for key, value in summary["counts"].items():
            candidate_train_counts[key] = candidate_train_counts.get(key, 0) + value
            candidate_test_counts[key] = candidate_test_counts.get(key, 0) + value

        train_cost = split_assignment_cost(
            candidate_train_counts,
            candidate_train_weight,
            candidate_train_rows,
            target_train_counts,
            target_train_weight,
            target_train_rows,
            row_cost_weight=row_cost_weight,
        ) + split_assignment_cost(
            test_counts,
            test_weight,
            test_rows,
            target_test_counts,
            target_test_weight,
            target_test_rows,
            row_cost_weight=row_cost_weight,
        )
        test_cost = split_assignment_cost(
            train_counts,
            train_weight,
            train_rows,
            target_train_counts,
            target_train_weight,
            target_train_rows,
            row_cost_weight=row_cost_weight,
        ) + split_assignment_cost(
            candidate_test_counts,
            candidate_test_weight,
            candidate_test_rows,
            target_test_counts,
            target_test_weight,
            target_test_rows,
            row_cost_weight=row_cost_weight,
        )

        if test_cost < train_cost:
            test_indices.extend(summary["indices"].tolist())
            test_weight = candidate_test_weight
            test_rows = candidate_test_rows
            test_counts = candidate_test_counts
        else:
            train_indices.extend(summary["indices"].tolist())
            train_weight = candidate_train_weight
            train_rows = candidate_train_rows
            train_counts = candidate_train_counts

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError(
            "El split agrupado/estratificado ha dejado un subconjunto vacio. "
            "Revisa las columnas de estratificacion o el tamano de test."
        )

    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def add_chunk_count_column(
    df,
    base_path=DATASET_DIR,
    sr=SAMPLE_RATE,
    chunk_length_s=CHUNK_LENGTH_S,
    overlap_s=OVERLAP_S,
):
    """Anade a un DataFrame una columna con el numero de chunks validos por audio."""
    df_with_counts = df.copy()
    chunk_counts = []

    for _, row in df_with_counts.iterrows():
        audio_path = os.path.normpath(os.path.join(base_path, row["path"]))
        chunk_counts.append(
            count_valid_chunks_in_audio(
                audio_path,
                sr=sr,
                chunk_length_s=chunk_length_s,
                overlap_s=overlap_s,
            )
        )

    df_with_counts["num_chunks"] = chunk_counts
    return df_with_counts


def get_background_bucket_weight(bucket_value):
    """Devuelve el peso de muestreo de un bucket de backgrounds en train."""
    normalized_bucket = str(bucket_value).strip()
    if normalized_bucket in TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS:
        return TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT
    if normalized_bucket in TRAIN_BACKGROUND_REDUCED_BUCKETS:
        return TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT
    return TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT


def select_training_background_subset(
    train_df,
    random_state=RANDOM_SEED,
    target_bg_to_siren_ratio=TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
):
    """
    Selecciona un subconjunto reproducible de backgrounds solo para train.

    Reglas:
    - Siren se conserva completo.
    - Validation/test no se tocan nunca.
    - Background se submuestrea por grupos seguros y no por archivos sueltos.
    - La prioridad de muestreo se define por buckets reproducibles y no por
      orden fijo de fuentes.
    - Se garantiza una representacion minima por bucket cuando el presupuesto
      global de chunks lo permite.
    """
    curated_df = train_df.copy()
    label_series = curated_df["label"].astype(str).str.strip().str.lower()
    bg_mask = label_series == "background"
    siren_mask = label_series.isin({"siren", "sirena"})

    curated_df["train_keep"] = siren_mask
    curated_df["background_sampling_weight"] = np.nan
    if bg_mask.any():
        curated_df.loc[bg_mask, "background_sampling_weight"] = curated_df.loc[
            bg_mask, "background_sampling_bucket"
        ].apply(get_background_bucket_weight)

    if not APPLY_TRAIN_BACKGROUND_SUBSAMPLING:
        curated_df.loc[bg_mask, "train_keep"] = True
        return curated_df

    siren_chunks = int(curated_df.loc[siren_mask, "num_chunks"].sum())
    total_background_chunks = int(curated_df.loc[bg_mask, "num_chunks"].sum())
    target_background_chunks = int(np.ceil(max(1.0, siren_chunks * target_bg_to_siren_ratio)))

    if (
        total_background_chunks <= target_background_chunks
        or not bg_mask.any()
        or siren_chunks <= 0
    ):
        curated_df.loc[bg_mask, "train_keep"] = True
        return curated_df

    valid_background_df = curated_df.loc[bg_mask & (curated_df["num_chunks"] > 0)].copy()
    if valid_background_df.empty:
        curated_df.loc[bg_mask, "train_keep"] = False
        return curated_df

    rng = np.random.default_rng(random_state)
    group_records = []

    for safe_group_id, group_df in valid_background_df.groupby("safe_group_id"):
        bucket_series = group_df["background_sampling_bucket"].dropna().astype(str)
        bucket_value = (
            bucket_series.mode().iloc[0]
            if not bucket_series.empty
            else str(group_df["source"].iloc[0])
        )
        group_records.append(
            {
                "safe_group_id": safe_group_id,
                "bucket": bucket_value,
                "num_chunks": int(group_df["num_chunks"].sum()),
                "bucket_weight": float(get_background_bucket_weight(bucket_value)),
                "random_key": float(rng.random()),
            }
        )

    groups_df = pd.DataFrame(group_records)
    if groups_df.empty:
        curated_df.loc[bg_mask, "train_keep"] = False
        return curated_df

    bucket_available_chunks = groups_df.groupby("bucket")["num_chunks"].sum().astype(float)
    bucket_weights = groups_df.groupby("bucket")["bucket_weight"].first().astype(float)
    weighted_available_chunks = bucket_available_chunks * bucket_weights
    bucket_target_chunks = (
        target_background_chunks * weighted_available_chunks / weighted_available_chunks.sum()
    )

    selected_group_ids = set()
    selected_chunks = 0
    selected_chunks_by_bucket = {bucket_name: 0 for bucket_name in bucket_target_chunks.index}

    ordered_buckets = sorted(
        bucket_target_chunks.index.tolist(),
        key=lambda bucket_name: (
            -float(bucket_target_chunks[bucket_name]),
            -float(bucket_weights[bucket_name]),
            bucket_name,
        ),
    )

    def register_group_selection(group_row):
        nonlocal selected_chunks
        selected_group_ids.add(group_row.safe_group_id)
        selected_chunks += int(group_row.num_chunks)
        selected_chunks_by_bucket[group_row.bucket] = (
            selected_chunks_by_bucket.get(group_row.bucket, 0) + int(group_row.num_chunks)
        )

    minimum_bucket_plan = []
    if TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET > 0:
        for bucket_name in ordered_buckets:
            bucket_groups = groups_df.loc[groups_df["bucket"] == bucket_name].sort_values(
                by=["num_chunks", "random_key"],
                ascending=[True, True],
            )
            minimum_group_count = min(TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET, len(bucket_groups))
            if minimum_group_count <= 0 or float(bucket_target_chunks[bucket_name]) <= 0.0:
                continue

            mandatory_groups = bucket_groups.head(minimum_group_count).copy()
            minimum_bucket_plan.append(
                {
                    "bucket": bucket_name,
                    "mandatory_chunks": int(mandatory_groups["num_chunks"].sum()),
                    "mandatory_groups": mandatory_groups,
                    "bucket_target": float(bucket_target_chunks[bucket_name]),
                    "bucket_weight": float(bucket_weights[bucket_name]),
                }
            )

        minimum_bucket_plan.sort(
            key=lambda item: (
                -item["bucket_target"],
                -item["bucket_weight"],
                item["mandatory_chunks"],
                item["bucket"],
            )
        )

        for plan_item in minimum_bucket_plan:
            if (
                selected_chunks + plan_item["mandatory_chunks"] > target_background_chunks
                and selected_group_ids
            ):
                continue

            for group_row in plan_item["mandatory_groups"].itertuples(index=False):
                if group_row.safe_group_id in selected_group_ids:
                    continue
                register_group_selection(group_row)

    for bucket_name in ordered_buckets:
        if selected_chunks >= target_background_chunks and selected_group_ids:
            break

        bucket_target = float(bucket_target_chunks[bucket_name])
        bucket_selected_chunks = selected_chunks_by_bucket.get(bucket_name, 0)
        bucket_groups = groups_df.loc[groups_df["bucket"] == bucket_name].sort_values(
            by=["random_key", "num_chunks"],
            ascending=[True, True],
        )

        for group_row in bucket_groups.itertuples(index=False):
            if group_row.safe_group_id in selected_group_ids:
                continue
            if (
                bucket_selected_chunks >= bucket_target and bucket_selected_chunks > 0
            ) or (
                selected_chunks >= target_background_chunks and selected_group_ids
            ):
                break

            register_group_selection(group_row)
            bucket_selected_chunks = selected_chunks_by_bucket.get(bucket_name, 0)

    if selected_chunks < target_background_chunks:
        remaining_groups = groups_df.loc[
            ~groups_df["safe_group_id"].isin(selected_group_ids)
        ].sort_values(
            by=["bucket_weight", "random_key", "num_chunks"],
            ascending=[False, True, True],
        )

        for group_row in remaining_groups.itertuples(index=False):
            selected_group_ids.add(group_row.safe_group_id)
            selected_chunks += int(group_row.num_chunks)
            if selected_chunks >= target_background_chunks:
                break

    curated_df.loc[bg_mask, "train_keep"] = curated_df.loc[bg_mask, "safe_group_id"].isin(
        selected_group_ids
    )

    return curated_df


def build_split_manifest_settings(stratify_columns):
    """Resume la configuracion que hace valido un manifiesto persistido."""
    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "sample_rate": SAMPLE_RATE,
        "chunk_length_s": CHUNK_LENGTH_S,
        "overlap_s": OVERLAP_S,
        "random_seed": RANDOM_SEED,
        "split_train_fraction": SPLIT_TRAIN_FRACTION,
        "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
        "split_test_fraction": SPLIT_TEST_FRACTION,
        "split_weight_column": SPLIT_WEIGHT_COLUMN,
        "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
        "split_stratify_columns": list(stratify_columns),
        "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
        "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
        "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
        "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
        "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
        "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
        "train_background_hard_negative_buckets": list(
            TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
        ),
        "train_background_reduced_buckets": list(TRAIN_BACKGROUND_REDUCED_BUCKETS),
    }


def try_load_split_manifest(df_master, manifest_settings):
    """Carga el manifiesto si existe y sigue siendo compatible con el dataset."""
    if not REUSE_SPLIT_MANIFEST:
        return None

    if not os.path.exists(SPLIT_MANIFEST_PATH) or not os.path.exists(SPLIT_MANIFEST_INFO_PATH):
        return None

    try:
        with open(SPLIT_MANIFEST_INFO_PATH, "r", encoding="utf-8") as file_handle:
            stored_settings = json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Aviso: no se ha podido leer la informacion del manifiesto ({exc}).")
        return None

    if stored_settings != manifest_settings:
        print("Aviso: el manifiesto persistido no coincide con la configuracion actual. Se regenerara.")
        return None

    try:
        manifest_df = pd.read_csv(SPLIT_MANIFEST_PATH)
    except Exception as exc:
        print(f"Aviso: no se ha podido leer el manifiesto de splits ({exc}). Se regenerara.")
        return None

    required_columns = {"path", "split", "train_keep"}
    if not required_columns.issubset(set(manifest_df.columns)):
        print("Aviso: faltan columnas obligatorias en el manifiesto. Se regenerara.")
        return None

    current_paths = set(df_master["path"].astype(str))
    manifest_paths = set(manifest_df["path"].astype(str))
    if current_paths != manifest_paths:
        print("Aviso: el manifiesto no corresponde al dataset actual. Se regenerara.")
        return None

    manifest_subset = manifest_df.loc[:, ["path", "split", "train_keep"]].copy()
    manifest_subset["split"] = manifest_subset["split"].astype(str)
    manifest_subset["train_keep"] = (
        manifest_subset["train_keep"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if manifest_subset["train_keep"].isna().any():
        print("Aviso: el manifiesto contiene valores no validos en `train_keep`. Se regenerara.")
        return None
    manifest_subset["train_keep"] = manifest_subset["train_keep"].astype(bool)
    return manifest_subset


def save_split_manifest(manifest_df, manifest_settings):
    """Persistencia estable del split y del submuestreo de train."""
    if not SAVE_SPLIT_MANIFEST:
        return

    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)
    manifest_df.to_csv(SPLIT_MANIFEST_PATH, index=False)

    with open(SPLIT_MANIFEST_INFO_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(manifest_settings, file_handle, indent=2)


def build_split_manifest(df_master, stratify_columns):
    """Genera el manifiesto completo de split y curacion de train."""
    train_idx, temp_idx = grouped_stratified_split(
        df_master,
        group_col="safe_group_id",
        test_size=(1.0 - SPLIT_TRAIN_FRACTION),
        stratify_columns=stratify_columns,
        random_state=RANDOM_SEED,
        weight_col=SPLIT_WEIGHT_COLUMN,
        row_cost_weight=SPLIT_ROW_COST_WEIGHT,
    )

    temp_df = df_master.iloc[temp_idx].copy()
    relative_test_fraction = SPLIT_TEST_FRACTION / (
        SPLIT_VALIDATION_FRACTION + SPLIT_TEST_FRACTION
    )
    validation_idx, test_idx = grouped_stratified_split(
        temp_df,
        group_col="safe_group_id",
        test_size=relative_test_fraction,
        stratify_columns=stratify_columns,
        random_state=RANDOM_SEED,
        weight_col=SPLIT_WEIGHT_COLUMN,
        row_cost_weight=SPLIT_ROW_COST_WEIGHT,
    )

    split_labels = pd.Series("unassigned", index=df_master.index, dtype="object")
    split_labels.iloc[train_idx] = "train"
    split_labels.iloc[temp_df.iloc[validation_idx].index] = "validation"
    split_labels.iloc[temp_df.iloc[test_idx].index] = "test"

    manifest_df = df_master.loc[
        :,
        [
            "path",
            "label",
            "source",
            "domain",
            "safe_group_id",
            "background_sampling_bucket",
            "num_chunks",
        ],
    ].copy()
    manifest_df["split"] = split_labels.values
    manifest_df["train_keep"] = False

    train_curated_df = select_training_background_subset(
        df_master.loc[split_labels == "train"].copy(),
        random_state=RANDOM_SEED,
        target_bg_to_siren_ratio=TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
    )
    manifest_df.loc[train_curated_df.index, "train_keep"] = train_curated_df["train_keep"].astype(bool)
    return manifest_df


def summarize_training_selection(full_train_df, selected_train_df):
    """Resume en formato JSON-friendly que se ha quedado dentro de train."""
    full_bg_df = full_train_df.loc[
        full_train_df["label"].astype(str).str.strip().str.lower() == "background"
    ]
    selected_bg_df = selected_train_df.loc[
        selected_train_df["label"].astype(str).str.strip().str.lower() == "background"
    ]

    summary = {
        "train_split_audio_count": int(len(full_train_df)),
        "train_selected_audio_count": int(len(selected_train_df)),
        "train_split_chunk_count": int(full_train_df["num_chunks"].sum()),
        "train_selected_chunk_count": int(selected_train_df["num_chunks"].sum()),
        "siren_audio_count": int(
            (selected_train_df["label"].astype(str).str.strip().str.lower().isin({"siren", "sirena"})).sum()
        ),
        "siren_chunk_count": int(
            selected_train_df.loc[
                selected_train_df["label"].astype(str).str.strip().str.lower().isin({"siren", "sirena"}),
                "num_chunks",
            ].sum()
        ),
        "background_audio_count_before": int(len(full_bg_df)),
        "background_audio_count_after": int(len(selected_bg_df)),
        "background_chunk_count_before": int(full_bg_df["num_chunks"].sum()),
        "background_chunk_count_after": int(selected_bg_df["num_chunks"].sum()),
        "background_bucket_count_before": int(full_bg_df["background_sampling_bucket"].nunique(dropna=True)),
        "background_bucket_count_after": int(selected_bg_df["background_sampling_bucket"].nunique(dropna=True)),
        "background_chunk_ratio_after": float(
            selected_bg_df["num_chunks"].sum()
            / max(
                1.0,
                selected_train_df.loc[
                    selected_train_df["label"].astype(str).str.strip().str.lower().isin({"siren", "sirena"}),
                    "num_chunks",
                ].sum(),
            )
        ),
        "background_chunks_by_bucket_before": full_bg_df.groupby("background_sampling_bucket")[
            "num_chunks"
        ].sum().sort_values(ascending=False).astype(int).to_dict(),
        "background_chunks_by_bucket_after": selected_bg_df.groupby("background_sampling_bucket")[
            "num_chunks"
        ].sum().sort_values(ascending=False).astype(int).to_dict(),
    }
    return summary


def print_split_diagnostics(split_name, df, stratify_columns=SPLIT_STRATIFY_COLUMNS):
    """
    Imprime diagnosticos del split tanto por audio como por chunk para detectar
    desajustes entre train, validation y test.
    """
    print(f"\nDiagnostico del split: {split_name}")
    print(f"Audios: {len(df)} | Chunks validos: {int(df['num_chunks'].sum())}")
    print("Distribucion por audio (label):")
    print(df["label"].value_counts())
    print("Distribucion por chunk (label):")
    label_chunk_counts = df.groupby("label")["num_chunks"].sum()
    print(label_chunk_counts)
    if {"background", "siren"}.issubset(set(label_chunk_counts.index)):
        ratio = float(label_chunk_counts["background"] / max(1.0, label_chunk_counts["siren"]))
        print(f"Ratio background/siren por chunk: {ratio:.2f}")
    available_stratify_columns = [
        column for column in stratify_columns if column in df.columns
    ]

    if available_stratify_columns:
        print(f"Distribucion por audio ({', '.join(available_stratify_columns)}):")
        print(df.groupby(available_stratify_columns).size())
        print(f"Distribucion por chunk ({', '.join(available_stratify_columns)}):")
        print(df.groupby(available_stratify_columns)["num_chunks"].sum())
    else:
        print("No hay columnas adicionales disponibles para diagnostico estratificado.")

    if "source" in df.columns:
        print("Top fuentes por audio:")
        print(df["source"].value_counts().head(10))
        print("Top fuentes por chunk:")
        print(df.groupby("source")["num_chunks"].sum().sort_values(ascending=False).head(10))

    if "background_sampling_bucket" in df.columns:
        background_df = df.loc[
            df["label"].astype(str).str.strip().str.lower() == "background"
        ]
        if not background_df.empty:
            print("Top buckets background por chunk:")
            print(
                background_df.groupby("background_sampling_bucket")["num_chunks"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )


class AudioDataGenerator(Sequence):
    """
    Generador de datos para Keras.

    Cada elemento del DataFrame representa un archivo de audio completo, pero
    la red se entrena con los chunks extraidos de ese audio. Por eso un lote de
    4 audios puede convertirse internamente en decenas de muestras.

    Si `balance_chunks=True`, el generador construye los lotes a partir de los
    audios del split actual, manteniendo un numero similar de chunks por clase.
    Esto evita data leakage porque nunca mezcla archivos de train, validation y
    test entre si.
    """

    def __init__(
        self,
        df,
        base_path=DATASET_DIR,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=True,
        sample_weights=None,
        balance_chunks=False,
        chunk_batch_size=TRAIN_CHUNK_BATCH_SIZE,
        seed=RANDOM_SEED,
        dataset_workers=PYDATASET_WORKERS,
        use_multiprocessing=PYDATASET_USE_MULTIPROCESSING,
        max_queue_size=PYDATASET_MAX_QUEUE_SIZE,
    ):
        super().__init__(
            workers=dataset_workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )
        self.df = df.reset_index(drop=True).copy()
        self.base_path = base_path
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.balance_chunks = balance_chunks
        self.chunk_batch_size = chunk_batch_size
        self.indices = np.arange(len(self.df))
        self.rng = np.random.default_rng(seed)
        self.dataset_workers = dataset_workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

        valid_mask = self.df["num_chunks"] > 0 if "num_chunks" in self.df.columns else np.ones(len(self.df), dtype=bool)
        self.class_row_indices = {}
        for class_id in sorted(self.df["target"].unique()):
            class_indices = self.df.index[valid_mask & (self.df["target"] == class_id)].to_numpy()
            if len(class_indices) > 0:
                self.class_row_indices[int(class_id)] = class_indices

        if self.balance_chunks and len(self.class_row_indices) < 2:
            raise RuntimeError(
                "El balanceo por chunks requiere al menos dos clases con chunks validos "
                "dentro del split de entrenamiento."
            )

        self.on_epoch_end()

    def __len__(self):
        """Numero de lotes por epoca."""
        if self.balance_chunks:
            total_chunks = int(self.df["num_chunks"].sum()) if "num_chunks" in self.df.columns else len(self.df)
            return int(np.ceil(max(1, total_chunks) / self.chunk_batch_size))
        return int(np.ceil(len(self.df) / self.batch_size))

    def _sample_chunk_subset(self, features, max_chunks):
        """Selecciona de forma aleatoria un subconjunto de chunks de un audio."""
        if len(features) <= max_chunks:
            return features

        selected_indices = self.rng.choice(len(features), size=max_chunks, replace=False)
        return features[selected_indices]

    def _build_batch_from_indices(self, batch_indices):
        """
        Construye un lote a partir de una lista de indices del DataFrame.

        Salida:
        - X: espectrogramas de varios chunks.
        - y: etiquetas binarias por chunk.
        - sample_weight: opcional, para compensar desbalance de clases.
        """
        batch_data = self.df.iloc[batch_indices]

        x_batch_list = []
        y_batch_list = []
        weight_batch_list = []

        for _, row in batch_data.iterrows():
            audio_path = os.path.normpath(os.path.join(self.base_path, row["path"]))
            features = extract_features_chunks(audio_path, augment=self.augment, rng=self.rng)

            if features is None or len(features) == 0:
                continue

            # Todos los chunks de un mismo archivo heredan la etiqueta del audio.
            label = float(row["target"])
            labels = np.full(shape=(len(features),), fill_value=label, dtype=np.float32)

            x_batch_list.append(features)
            y_batch_list.append(labels)

            if self.sample_weights is not None:
                # El mismo peso de clase se replica para todos los chunks del audio.
                class_weight = self.sample_weights[int(label)]
                replicated_weights = np.full(shape=(len(features),), fill_value=class_weight, dtype=np.float32)
                weight_batch_list.append(replicated_weights)

        if not x_batch_list:
            empty_x = np.empty((0, *INPUT_SHAPE), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.float32)
            if self.sample_weights is not None:
                empty_weights = np.empty((0,), dtype=np.float32)
                return empty_x, empty_y, empty_weights
            return empty_x, empty_y

        x_data = np.vstack(x_batch_list).astype(np.float32)
        y_data = np.hstack(y_batch_list).astype(np.float32)

        if self.sample_weights is not None:
            weight_data = np.hstack(weight_batch_list).astype(np.float32)
            return x_data, y_data, weight_data

        return x_data, y_data

    def _build_balanced_chunk_batch(self):
        """
        Construye un lote balanceado por numero de chunks y no por numero de audios.

        Solo usa filas del split actual, asi que no introduce data leakage entre
        train, validation y test.
        """
        class_ids = sorted(self.class_row_indices.keys())
        num_classes = len(class_ids)
        base_chunks_per_class = self.chunk_batch_size // num_classes
        remainder = self.chunk_batch_size % num_classes

        x_batch_list = []
        y_batch_list = []
        weight_batch_list = []

        for class_position, class_id in enumerate(class_ids):
            target_chunks = base_chunks_per_class + (1 if class_position < remainder else 0)
            remaining_chunks = target_chunks
            attempts = 0
            max_attempts = max(20, len(self.class_row_indices[class_id]) * 5)

            while remaining_chunks > 0 and attempts < max_attempts:
                if len(self.class_row_indices[class_id]) == 0:
                    break

                row_index = int(self.rng.choice(self.class_row_indices[class_id]))
                row = self.df.iloc[row_index]
                audio_path = os.path.normpath(os.path.join(self.base_path, row["path"]))
                features = extract_features_chunks(audio_path, augment=self.augment, rng=self.rng)
                attempts += 1

                if features is None or len(features) == 0:
                    continue

                selected_features = self._sample_chunk_subset(features, remaining_chunks)
                label = float(class_id)
                labels = np.full(shape=(len(selected_features),), fill_value=label, dtype=np.float32)

                x_batch_list.append(selected_features)
                y_batch_list.append(labels)

                if self.sample_weights is not None:
                    class_weight = self.sample_weights[int(label)]
                    replicated_weights = np.full(
                        shape=(len(selected_features),),
                        fill_value=class_weight,
                        dtype=np.float32,
                    )
                    weight_batch_list.append(replicated_weights)

                remaining_chunks -= len(selected_features)

            if remaining_chunks > 0:
                raise RuntimeError(
                    "No se ha podido construir un lote balanceado por chunks para la clase "
                    f"{class_id}. Revisa la distribucion del split y los audios con 0 chunks."
                )

        x_data = np.vstack(x_batch_list).astype(np.float32)
        y_data = np.hstack(y_batch_list).astype(np.float32)
        permutation = self.rng.permutation(len(y_data))
        x_data = x_data[permutation]
        y_data = y_data[permutation]

        if self.sample_weights is not None:
            weight_data = np.hstack(weight_batch_list).astype(np.float32)[permutation]
            return x_data, y_data, weight_data

        return x_data, y_data

    def __getitem__(self, index):
        """
        Devuelve un lote valido para Keras.

        Si el lote correspondiente a `index` no produce ningun chunk valido,
        se buscan lotes posteriores dentro de la misma epoca hasta encontrar
        uno util. Esto evita que `model.fit` reciba batches vacios.
        """
        if self.balance_chunks:
            return self._build_balanced_chunk_batch()

        num_batches = len(self)

        for offset in range(num_batches):
            candidate_index = (index + offset) % num_batches
            batch_indices = self.indices[
                candidate_index * self.batch_size : (candidate_index + 1) * self.batch_size
            ]
            batch = self._build_batch_from_indices(batch_indices)

            if self.sample_weights is not None:
                x_data, _, weight_data = batch
                if x_data.shape[0] > 0 and weight_data.shape[0] > 0:
                    if offset > 0:
                        print(
                            "Aviso: se ha omitido un lote vacio y se ha reutilizado "
                            f"el lote valido siguiente (indice {candidate_index})."
                        )
                    return batch
            else:
                x_data, _ = batch
                if x_data.shape[0] > 0:
                    if offset > 0:
                        print(
                            "Aviso: se ha omitido un lote vacio y se ha reutilizado "
                            f"el lote valido siguiente (indice {candidate_index})."
                        )
                    return batch

        raise RuntimeError(
            "No se ha podido construir ningun lote con chunks validos. "
            "Revisa las rutas de audio, su duracion minima y el preprocesado."
        )

    def on_epoch_end(self):
        """Baraja el orden de los audios al final de cada epoca."""
        if self.shuffle:
            self.rng.shuffle(self.indices)
            for class_indices in self.class_row_indices.values():
                self.rng.shuffle(class_indices)


def build_cantarini_cnn_model(input_shape=INPUT_SHAPE):
    """
    Construye la CNN principal del proyecto.

    Decisiones de diseno:
    - SeparableConv2D para reducir coste computacional.
    - BatchNormalization para estabilizar el entrenamiento.
    - GlobalAveragePooling2D para reducir parametros frente a Flatten.
    - Salida sigmoid para obtener probabilidad de sirena por chunk.
    """
    model = Sequential(
        [
            Input(shape=input_shape),

            # Bloque 1: patrones locales de bajo nivel.
            SeparableConv2D(CONV_FILTERS[0], (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation("elu"),
            SeparableConv2D(CONV_FILTERS[0], (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation("elu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Bloque 2: patrones algo mas abstractos.
            SeparableConv2D(CONV_FILTERS[1], (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation("elu"),
            SeparableConv2D(CONV_FILTERS[1], (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation("elu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Bloque 3: representacion compacta antes de la clasificacion final.
            SeparableConv2D(CONV_FILTERS[2], (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation("elu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Cabezal de clasificacion.
            GlobalAveragePooling2D(),
            Dropout(0.25),
            Dense(DENSE_UNITS, activation="elu", kernel_initializer="he_uniform"),
            Dropout(0.15),
            Dense(1, activation="sigmoid"),
        ]
    )

    metrics = [
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        BinaryF1Score(name="f1", threshold=0.5),
    ]

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    return model


def build_training_callbacks():
    """
    Define callbacks para parar el entrenamiento a tiempo y guardar el mejor modelo.
    """
    return [
        EarlyStopping(
            monitor="val_auc_pr",
            patience=6,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc_pr",
            factor=0.5,
            patience=3,
            mode="max",
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def collect_chunk_predictions(model, df, base_path):
    """
    Recorre un conjunto de audios y devuelve:
    - etiquetas reales por chunk
    - probabilidades predichas por chunk

    Esta funcion se usa para evaluar fuera de `model.fit`, de forma mas flexible.
    """
    y_true_all = []
    y_score_all = []

    for _, row in df.iterrows():
        audio_path = os.path.normpath(os.path.join(base_path, row["path"]))
        features = extract_features_chunks(audio_path, augment=False)

        if features is None or len(features) == 0:
            continue

        scores = model.predict(features, verbose=0).reshape(-1).astype(np.float32)
        labels = np.full(shape=(len(scores),), fill_value=int(row["target"]), dtype=np.int32)

        y_true_all.append(labels)
        y_score_all.append(scores)

    if not y_true_all:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    return np.concatenate(y_true_all), np.concatenate(y_score_all)


def compute_metrics(y_true, y_pred, y_score, chunk_step_s=CHUNK_STEP_S):
    """
    Calcula metricas de clasificacion y una estimacion de falsas alarmas por minuto.

    `y_score` son probabilidades y `y_pred` son etiquetas binarias derivadas de
    un umbral concreto.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    auc_pr = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    negative_chunks = max(1, int(np.sum(y_true == 0)))
    false_alarms_per_min = (fp / negative_chunks) * (60.0 / chunk_step_s)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "auc_pr": float(auc_pr),
        "false_alarms_per_min": float(false_alarms_per_min),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


def select_best_threshold(y_true, y_scores, target_false_alarms_per_min=TARGET_FALSE_ALARMS_PER_MIN):
    """
    Busca un umbral de referencia para convertir probabilidades en binario.

    Importante:
    - El modelo produce probabilidades por chunk.
    - Este umbral se usa solo como analisis auxiliar.
    - La decision final en produccion puede usar otra logica externa.
    - La seleccion prioriza F2 para dar mas peso al recall que a la precision.
    """
    threshold_rows = []

    for threshold in THRESHOLD_GRID:
        y_pred = (y_scores >= threshold).astype(np.int32)
        metrics = compute_metrics(y_true, y_pred, y_scores)
        threshold_rows.append({"threshold": float(threshold), **metrics})

    threshold_df = pd.DataFrame(threshold_rows)
    allowed = threshold_df[threshold_df["false_alarms_per_min"] <= target_false_alarms_per_min]

    if not allowed.empty:
        best_row = allowed.sort_values(
            by=["f2", "recall", "precision", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    else:
        best_row = threshold_df.sort_values(
            by=["false_alarms_per_min", "f2", "recall"],
            ascending=[True, False, False],
        ).iloc[0]

    return float(best_row["threshold"]), threshold_df


def plot_training_history(history):
    """Dibuja la evolucion de las metricas principales durante el entrenamiento."""
    history_dict = history.history
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(history_dict["loss"], label="Entrenamiento")
    axes[0, 0].plot(history_dict["val_loss"], label="Validacion")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epocas")
    axes[0, 0].set_ylabel("Binary Crossentropy")
    axes[0, 0].legend()

    axes[0, 1].plot(history_dict["precision"], label="Precision train")
    axes[0, 1].plot(history_dict["val_precision"], label="Precision val")
    axes[0, 1].plot(history_dict["recall"], label="Recall train")
    axes[0, 1].plot(history_dict["val_recall"], label="Recall val")
    axes[0, 1].set_title("Precision y Recall")
    axes[0, 1].set_xlabel("Epocas")
    axes[0, 1].legend()

    axes[1, 0].plot(history_dict["auc_pr"], label="AUC-PR train")
    axes[1, 0].plot(history_dict["val_auc_pr"], label="AUC-PR val")
    axes[1, 0].set_title("AUC-PR")
    axes[1, 0].set_xlabel("Epocas")
    axes[1, 0].legend()

    axes[1, 1].plot(history_dict["f1"], label="F1 train")
    axes[1, 1].plot(history_dict["val_f1"], label="F1 val")
    axes[1, 1].set_title("F1")
    axes[1, 1].set_xlabel("Epocas")
    axes[1, 1].legend()

    fig.tight_layout()
    plt.show()


def print_metrics_block(title, metrics):
    """Imprime un resumen compacto de metricas para validacion o test."""
    print(f"\n{title}")
    print(
        "Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | "
        "AUC-PR: {auc_pr:.4f} | Falsas alarmas/min: {false_alarms_per_min:.2f}".format(**metrics)
    )
    print(
        "Matriz de confusion [[TN, FP], [FN, TP]] = "
        f"{metrics['confusion_matrix']}"
    )


def build_metrics_report_block(title, metrics):
    """
    Crea un bloque de texto con metricas y matriz de confusion para guardarlo
    en un fichero.
    """
    lines = [
        title,
        (
            "Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | "
            "AUC-PR: {auc_pr:.4f} | Falsas alarmas/min: {false_alarms_per_min:.2f}"
        ).format(**metrics),
        "Matriz de confusion [[TN, FP], [FN, TP]]:",
        str(np.array(metrics["confusion_matrix"])),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1. Cargar metadata y construir los splits sin leakage.
    # -----------------------------------------------------------------------
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

    print("Cargando metadata...")
    if RUNTIME_CONFIG_PATH is not None:
        print(f"Overrides de configuracion cargados desde: {RUNTIME_CONFIG_PATH}")
        print(f"Claves sobrescritas: {', '.join(sorted(RUNTIME_CONFIG_OVERRIDES.keys()))}")

    df_master = pd.read_csv(METADATA_PATH)
    if df_master["path"].duplicated().any():
        duplicated_paths = df_master.loc[df_master["path"].duplicated(), "path"].tolist()[:10]
        raise RuntimeError(
            "El metadata contiene paths duplicados. Esto rompe el manifiesto de splits. "
            f"Ejemplos: {duplicated_paths}"
        )

    df_master = enrich_metadata_columns(df_master)
    effective_split_stratify_columns = resolve_stratify_columns(
        df_master,
        SPLIT_STRATIFY_COLUMNS,
        fallback_columns=("label", "source"),
    )

    label_encoder = LabelEncoder()
    df_master["target"] = label_encoder.fit_transform(df_master["label"])
    print("Contando chunks validos de todo el dataset. Esto puede tardar unos minutos...")
    df_master = add_chunk_count_column(df_master, base_path=DATASET_DIR)

    print(
        f"Dataset completo -> audios: {len(df_master)} | "
        f"chunks validos totales: {int(df_master['num_chunks'].sum())}"
    )
    print("Distribucion por chunk en el dataset completo:")
    print(df_master.groupby("label")["num_chunks"].sum())

    manifest_settings = build_split_manifest_settings(effective_split_stratify_columns)
    split_manifest = try_load_split_manifest(df_master, manifest_settings)

    if split_manifest is None:
        print(
            "Generando un nuevo manifiesto de splits agrupados y estratificados por: "
            f"{', '.join(effective_split_stratify_columns)}"
        )
        split_manifest = build_split_manifest(df_master, effective_split_stratify_columns)
        save_split_manifest(split_manifest, manifest_settings)
        if SAVE_SPLIT_MANIFEST:
            print(f"Manifiesto de splits guardado en: {SPLIT_MANIFEST_PATH}")
            print(f"Informacion del manifiesto guardada en: {SPLIT_MANIFEST_INFO_PATH}")
    else:
        print(f"Usando manifiesto de splits existente: {SPLIT_MANIFEST_PATH}")

    df_master = df_master.merge(
        split_manifest.loc[:, ["path", "split", "train_keep"]],
        on="path",
        how="left",
        validate="one_to_one",
    )

    if df_master["split"].isna().any():
        raise RuntimeError(
            "Hay audios sin split asignado tras cargar el manifiesto. "
            "Revisa el metadata y el fichero de manifiesto."
        )

    train_split_df = df_master.loc[df_master["split"] == "train"].copy()
    val_df = df_master.loc[df_master["split"] == "validation"].copy()
    test_df = df_master.loc[df_master["split"] == "test"].copy()

    train_df = train_split_df.loc[
        train_split_df["label"].astype(str).str.strip().str.lower().isin({"siren", "sirena"})
        | train_split_df["train_keep"].astype(bool)
    ].copy()
    train_selection_summary = summarize_training_selection(train_split_df, train_df)

    print(
        f"Archivos originales distribuidos en -> Train: {len(train_split_df)} | "
        f"Validation: {len(val_df)} | Test: {len(test_df)}"
    )
    print(
        f"Chunks validos distribuidos en -> Train: {int(train_split_df['num_chunks'].sum())} | "
        f"Validation: {int(val_df['num_chunks'].sum())} | Test: {int(test_df['num_chunks'].sum())}"
    )
    print(
        f"Configuracion temporal -> chunk: {CHUNK_LENGTH_S:.2f} s | "
        f"solapamiento: {OVERLAP_S:.2f} s | paso entre decisiones: {CHUNK_STEP_S:.2f} s"
    )
    print(
        "Curacion de train -> audios: {train_audio_count} -> {selected_audio_count} | "
        "chunks: {train_chunk_count} -> {selected_chunk_count}".format(
            train_audio_count=train_selection_summary["train_split_audio_count"],
            selected_audio_count=train_selection_summary["train_selected_audio_count"],
            train_chunk_count=train_selection_summary["train_split_chunk_count"],
            selected_chunk_count=train_selection_summary["train_selected_chunk_count"],
        )
    )
    print(
        "Background train -> audios: {before_audio} -> {after_audio} | "
        "chunks: {before_chunk} -> {after_chunk} | buckets: {before_bucket} -> {after_bucket} | "
        "ratio bg/sirena final: {ratio:.2f}".format(
            before_audio=train_selection_summary["background_audio_count_before"],
            after_audio=train_selection_summary["background_audio_count_after"],
            before_chunk=train_selection_summary["background_chunk_count_before"],
            after_chunk=train_selection_summary["background_chunk_count_after"],
            before_bucket=train_selection_summary["background_bucket_count_before"],
            after_bucket=train_selection_summary["background_bucket_count_after"],
            ratio=train_selection_summary["background_chunk_ratio_after"],
        )
    )
    print_parallelism_configuration()
    print("Distribucion de clases en train (split completo antes de curacion):")
    print(train_split_df["label"].value_counts())
    print_split_diagnostics("Train split completo", train_split_df, effective_split_stratify_columns)
    print("Distribucion de clases en train (subset usado por el modelo):")
    print(train_df["label"].value_counts())
    print_split_diagnostics("Train usado por el modelo", train_df, effective_split_stratify_columns)
    print_split_diagnostics("Validation", val_df, effective_split_stratify_columns)
    print_split_diagnostics("Test", test_df, effective_split_stratify_columns)

    # -----------------------------------------------------------------------
    # 2. Compensar opcionalmente el desbalance entre clases.
    # -----------------------------------------------------------------------
    class_sample_weights = None
    if USE_CLASS_WEIGHTS:
        class_sample_weights, class_chunk_counts = compute_class_sample_weights_from_chunks(
            train_df,
            base_path=DATASET_DIR,
            sr=SAMPLE_RATE,
            chunk_length_s=CHUNK_LENGTH_S,
            overlap_s=OVERLAP_S,
        )
        print(f"Chunks validos por clase en entrenamiento: {class_chunk_counts}")
        print(f"Pesos por clase usados en entrenamiento: {class_sample_weights}")
    else:
        print("Pesos por clase desactivados. El entrenamiento usara el peso natural de cada chunk.")

    # -----------------------------------------------------------------------
    # 3. Crear generadores:
    #    - train con augmentacion
    #    - validation sin augmentacion
    # -----------------------------------------------------------------------
    train_sample_weights = class_sample_weights
    if USE_BALANCED_CHUNK_BATCHES:
        print(
            "Balanceo de lotes por chunk activado en train "
            f"({TRAIN_CHUNK_BATCH_SIZE} chunks objetivo por lote)."
        )
        if train_sample_weights is not None:
            print(
                "Aviso: se desactivan los class weights efectivos porque el generador "
                "ya equilibra el numero de chunks por clase dentro de cada lote."
            )
            train_sample_weights = None
    else:
        print("Balanceo de lotes por chunk desactivado. El batch se construira por audios.")

    effective_use_class_weights = train_sample_weights is not None

    train_gen = AudioDataGenerator(
        train_df,
        base_path=DATASET_DIR,
        batch_size=BATCH_SIZE,
        augment=USE_DATA_AUGMENTATION,
        shuffle=True,
        sample_weights=train_sample_weights,
        balance_chunks=USE_BALANCED_CHUNK_BATCHES,
        chunk_batch_size=TRAIN_CHUNK_BATCH_SIZE,
        seed=RANDOM_SEED,
        dataset_workers=PYDATASET_WORKERS,
        use_multiprocessing=PYDATASET_USE_MULTIPROCESSING,
        max_queue_size=PYDATASET_MAX_QUEUE_SIZE,
    )
    val_gen = AudioDataGenerator(
        val_df,
        base_path=DATASET_DIR,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False,
        balance_chunks=False,
        seed=RANDOM_SEED,
        dataset_workers=PYDATASET_WORKERS,
        use_multiprocessing=PYDATASET_USE_MULTIPROCESSING,
        max_queue_size=PYDATASET_MAX_QUEUE_SIZE,
    )

    # -----------------------------------------------------------------------
    # 4. Construir y entrenar la CNN.
    # -----------------------------------------------------------------------
    model = build_cantarini_cnn_model(input_shape=INPUT_SHAPE)
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=build_training_callbacks(),
    )

    model.save(MODEL_PATH)
    print(f"\nEntrenamiento finalizado. Modelo guardado en: {MODEL_PATH}")

    if SHOW_TRAINING_PLOTS:
        plot_training_history(history)

    # -----------------------------------------------------------------------
    # 5. Evaluar la salida probabilistica por chunk en validacion y test.
    # -----------------------------------------------------------------------
    val_metrics = None
    test_metrics = None

    print("\nEvaluando la salida probabilistica por chunk sobre validacion...")
    val_y_true, val_y_scores = collect_chunk_predictions(model, val_df, DATASET_DIR)
    if USE_THRESHOLD_ANALYSIS:
        best_threshold, threshold_table = select_best_threshold(
            val_y_true,
            val_y_scores,
            target_false_alarms_per_min=TARGET_FALSE_ALARMS_PER_MIN,
        )

        val_predictions = (val_y_scores >= best_threshold).astype(np.int32)
        val_metrics = compute_metrics(val_y_true, val_predictions, val_y_scores)

        print(
            f"Umbral de referencia por chunk seleccionado para analisis auxiliar: {best_threshold:.2f}"
        )
        print_metrics_block("Validacion por chunk", val_metrics)
        print(
            "La salida principal del sistema debe interpretarse como probabilidad por chunk. "
            "Cualquier logica binaria o de interfaz debe implementarse de forma causal fuera de este script."
        )
    else:
        best_threshold = 0.5
        threshold_table = None
        print("Analisis de umbral desactivado. La salida del modelo se tratara solo como probabilidad por chunk.")

    print("\nEvaluando el modelo con datos desconocidos (Test Set) por chunk...")
    test_y_true, test_y_scores = collect_chunk_predictions(model, test_df, DATASET_DIR)
    if USE_THRESHOLD_ANALYSIS:
        test_predictions = (test_y_scores >= best_threshold).astype(np.int32)
        test_metrics = compute_metrics(test_y_true, test_predictions, test_y_scores)
        print_metrics_block("Test por chunk", test_metrics)

        threshold_summary = threshold_table[
            ["threshold", "precision", "recall", "f1", "f2", "auc_pr", "false_alarms_per_min"]
        ]
        print("\nResumen de umbrales evaluados en validacion:")
        print(threshold_summary.to_string(index=False))
    else:
        print(
            "Analisis binario desactivado en test. "
            "Las probabilidades por chunk pueden evaluarse despues con la logica externa que use el dispositivo."
        )

    if SAVE_CONFUSION_REPORT and USE_THRESHOLD_ANALYSIS:
        report_sections = [
            "Reporte de metricas y matrices de confusion",
            f"Modelo: {MODEL_PATH}",
            f"Chunk length (s): {CHUNK_LENGTH_S}",
            f"Overlap (s): {OVERLAP_S}",
            f"Decision step (s): {CHUNK_STEP_S}",
            f"Spectral frontend: {SPECTRAL_FRONTEND}",
            f"Feature representation: {FEATURE_REPRESENTATION}",
            f"Spectrogram normalization: {SPECTROGRAM_NORMALIZATION}",
            f"STFT window: {STFT_WINDOW}",
            f"HPSS margin: {HPSS_MARGIN}",
            f"N_FFT: {N_FFT}",
            f"HOP_LENGTH: {HOP_LENGTH}",
            f"Padded chunk samples: {PADDED_CHUNK_SAMPLES}",
            f"Linear freq bins: {LINEAR_FREQ_BINS}",
            f"Time frames: {TIME_FRAMES}",
            f"Mel bins: {MEL_BINS}",
            f"Split manifest path: {SPLIT_MANIFEST_PATH}",
            f"Split manifest info path: {SPLIT_MANIFEST_INFO_PATH}",
            f"Split train fraction: {SPLIT_TRAIN_FRACTION}",
            f"Split validation fraction: {SPLIT_VALIDATION_FRACTION}",
            f"Split test fraction: {SPLIT_TEST_FRACTION}",
            f"Split stratify columns: {', '.join(effective_split_stratify_columns)}",
            f"Split weight column: {SPLIT_WEIGHT_COLUMN}",
            f"Split row cost weight: {SPLIT_ROW_COST_WEIGHT}",
            f"Train background subsampling: {APPLY_TRAIN_BACKGROUND_SUBSAMPLING}",
            "Train background target ratio (bg/sirena chunks): "
            f"{TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO}",
            "Train background minimum groups per bucket: "
            f"{TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET}",
            "Train background hard-negative buckets: "
            f"{', '.join(TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS)}",
            "Train background reduced buckets: "
            f"{', '.join(TRAIN_BACKGROUND_REDUCED_BUCKETS)}",
            f"Use data augmentation: {USE_DATA_AUGMENTATION}",
            f"Augmentation apply probability: {AUGMENTATION_APPLY_PROB}",
            f"Use spectral EQ augmentation: {USE_SPECTRAL_EQ_AUGMENTATION}",
            f"EQ augmentation target probability: {EQ_AUGMENTATION_PROB}",
            "EQ augmentation effective conditional probability: "
            f"{get_effective_eq_apply_probability():.4f}",
            f"EQ one-filter probability: {EQ_ONE_FILTER_PROB}",
            f"EQ shelf gain max (dB): +/-{EQ_SHELF_GAIN_DB_MAX}",
            f"EQ bell gain max in siren band (dB): +/-{EQ_BELL_GAIN_DB_MAX_SIREN_BAND}",
            f"EQ total accumulated gain limit (dB): +/-{EQ_TOTAL_GAIN_DB_LIMIT}",
            "EQ low-shelf cutoff range (Hz): "
            f"{EQ_LOW_SHELF_CUTOFF_HZ_RANGE[0]} - {EQ_LOW_SHELF_CUTOFF_HZ_RANGE[1]}",
            "EQ bell center range (Hz): "
            f"{EQ_BELL_CENTER_HZ_RANGE[0]} - {EQ_BELL_CENTER_HZ_RANGE[1]}",
            "EQ high-shelf cutoff range (Hz): "
            f"{EQ_HIGH_SHELF_CUTOFF_HZ_RANGE[0]} - {EQ_HIGH_SHELF_CUTOFF_HZ_RANGE[1]}",
            "EQ bell bandwidth range (octaves): "
            f"{EQ_BELL_BANDWIDTH_OCTAVES_RANGE[0]} - {EQ_BELL_BANDWIDTH_OCTAVES_RANGE[1]}",
            "EQ shelf sharpness range: "
            f"{EQ_SHELF_SHARPNESS_RANGE[0]} - {EQ_SHELF_SHARPNESS_RANGE[1]}",
            f"Balanced chunk batches: {USE_BALANCED_CHUNK_BATCHES}",
            f"Train chunk batch size: {TRAIN_CHUNK_BATCH_SIZE}",
            f"Effective class weights: {effective_use_class_weights}",
            f"Logical CPU count: {LOGICAL_CPU_COUNT}",
            f"TF intra_op threads: {TF_INTRA_OP_THREADS}",
            f"TF inter_op threads: {TF_INTER_OP_THREADS}",
            f"PyDataset workers: {PYDATASET_WORKERS}",
            f"PyDataset multiprocessing: {PYDATASET_USE_MULTIPROCESSING}",
            f"PyDataset max queue size: {PYDATASET_MAX_QUEUE_SIZE}",
            "Resumen de curacion de train: "
            f"{json.dumps(train_selection_summary, ensure_ascii=True)}",
            f"Umbral de referencia: {best_threshold:.2f}",
            "",
            build_metrics_report_block("Validacion por chunk", val_metrics),
            "",
            build_metrics_report_block("Test por chunk", test_metrics),
        ]

        with open(CONFUSION_REPORT_PATH, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(report_sections) + "\n")

        print(f"\nReporte de matrices de confusion guardado en: {CONFUSION_REPORT_PATH}")

    # -----------------------------------------------------------------------
    # 6. Guardar una pequena configuracion asociada al modelo.
    # -----------------------------------------------------------------------
    if SAVE_POSTPROCESSING_CONFIG:
        with open(POSTPROCESSING_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "model_path": MODEL_PATH,
                    "recommended_chunk_threshold": best_threshold,
                    "target_false_alarms_per_min": TARGET_FALSE_ALARMS_PER_MIN,
                    "chunk_length_s": CHUNK_LENGTH_S,
                    "overlap_s": OVERLAP_S,
                    "decision_step_s": CHUNK_STEP_S,
                    "spectral_frontend": SPECTRAL_FRONTEND,
                    "feature_representation": FEATURE_REPRESENTATION,
                    "spectrogram_normalization": SPECTROGRAM_NORMALIZATION,
                    "window": STFT_WINDOW,
                    "hpss_margin": HPSS_MARGIN,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "padded_chunk_samples": PADDED_CHUNK_SAMPLES,
                    "linear_freq_bins": LINEAR_FREQ_BINS,
                    "time_frames": TIME_FRAMES,
                    "mel_bins": MEL_BINS,
                    "split_manifest_path": SPLIT_MANIFEST_PATH,
                    "split_manifest_info_path": SPLIT_MANIFEST_INFO_PATH,
                    "split_manifest_version": SPLIT_MANIFEST_VERSION,
                    "split_train_fraction": SPLIT_TRAIN_FRACTION,
                    "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
                    "split_test_fraction": SPLIT_TEST_FRACTION,
                    "split_stratify_columns": list(effective_split_stratify_columns),
                    "split_weight_column": SPLIT_WEIGHT_COLUMN,
                    "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
                    "conv_filters": list(CONV_FILTERS),
                    "dense_units": DENSE_UNITS,
                    "runtime_config_path": RUNTIME_CONFIG_PATH,
                    "runtime_config_overrides": RUNTIME_CONFIG_OVERRIDES,
                    "labels": label_encoder.classes_.tolist(),
                    "output_mode": "chunk_probability",
                    "use_class_weights": USE_CLASS_WEIGHTS,
                    "effective_use_class_weights": effective_use_class_weights,
                    "use_data_augmentation": USE_DATA_AUGMENTATION,
                    "augmentation_apply_probability": AUGMENTATION_APPLY_PROB,
                    "use_spectral_eq_augmentation": USE_SPECTRAL_EQ_AUGMENTATION,
                    "eq_augmentation_probability": EQ_AUGMENTATION_PROB,
                    "eq_effective_conditional_probability": get_effective_eq_apply_probability(),
                    "eq_one_filter_probability": EQ_ONE_FILTER_PROB,
                    "eq_shelf_gain_db_max": EQ_SHELF_GAIN_DB_MAX,
                    "eq_bell_gain_db_max_siren_band": EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
                    "eq_total_gain_db_limit": EQ_TOTAL_GAIN_DB_LIMIT,
                    "eq_low_shelf_cutoff_hz_range": list(EQ_LOW_SHELF_CUTOFF_HZ_RANGE),
                    "eq_bell_center_hz_range": list(EQ_BELL_CENTER_HZ_RANGE),
                    "eq_high_shelf_cutoff_hz_range": list(EQ_HIGH_SHELF_CUTOFF_HZ_RANGE),
                    "eq_bell_bandwidth_octaves_range": list(EQ_BELL_BANDWIDTH_OCTAVES_RANGE),
                    "eq_shelf_sharpness_range": list(EQ_SHELF_SHARPNESS_RANGE),
                    "use_frequency_normalization": SPECTROGRAM_NORMALIZATION == "frequency",
                    "use_threshold_analysis": USE_THRESHOLD_ANALYSIS,
                    "use_balanced_chunk_batches": USE_BALANCED_CHUNK_BATCHES,
                    "train_chunk_batch_size": TRAIN_CHUNK_BATCH_SIZE,
                    "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
                    "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
                    "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
                    "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
                    "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
                    "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
                    "train_background_hard_negative_buckets": list(
                        TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
                    ),
                    "train_background_reduced_buckets": list(
                        TRAIN_BACKGROUND_REDUCED_BUCKETS
                    ),
                    "train_selection_summary": train_selection_summary,
                    "logical_cpu_count": LOGICAL_CPU_COUNT,
                    "tf_intra_op_threads": TF_INTRA_OP_THREADS,
                    "tf_inter_op_threads": TF_INTER_OP_THREADS,
                    "pydataset_workers": PYDATASET_WORKERS,
                    "pydataset_use_multiprocessing": PYDATASET_USE_MULTIPROCESSING,
                    "pydataset_max_queue_size": PYDATASET_MAX_QUEUE_SIZE,
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "threshold_selection_metric": "f2",
                    "notes": (
                        "La CNN produce una probabilidad por chunk. "
                        "Las decisiones binarias o el suavizado temporal deben implementarse "
                        "de forma causal en el sistema de produccion."
                    ),
                },
                file_handle,
                indent=2,
            )

        print(f"\nConfiguracion de postprocesado guardada en: {POSTPROCESSING_PATH}")
