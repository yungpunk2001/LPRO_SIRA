import json
import os
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
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


# ---------------------------------------------------------------------------
# Script de entrenamiento de una CNN para detectar sirenas en audio.
#
# Idea general del pipeline:
# 1. Leer cada audio y dividirlo en chunks de 0.5 s.
# 2. Convertir cada chunk en un espectrograma armonico de tamano fijo.
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
RANDOM_SEED = 42
SAMPLE_RATE = 16000
CHUNK_LENGTH_S = 0.5
OVERLAP_S = 0.0
CHUNK_STEP_S = CHUNK_LENGTH_S - OVERLAP_S
BATCH_SIZE = 4
EPOCHS = 50
THRESHOLD_GRID = np.linspace(0.10, 0.95, 18)
TARGET_FALSE_ALARMS_PER_MIN = 1.0

# ---------------------------------------------------------------------------
# Opciones activables del entrenamiento y evaluacion
# ---------------------------------------------------------------------------
# Si es True, aplica pesos por clase para compensar desbalance en el entrenamiento.
# Si el numero de audios por clase ya esta equilibrado, puede dejarse en False.
USE_CLASS_WEIGHTS = False

# Si es True, aplica augmentacion a los chunks del conjunto de entrenamiento.
# Sirve para que la red vea ejemplos mas variados y generalice mejor.
USE_DATA_AUGMENTATION = False

# Controla que representacion espectral entra en la CNN:
# - "harmonic": solo la componente armonica tras HPSS.
# - "full": espectrograma completo sin separar armonica/percusiva.
# - "harmonic_full": dos canales, uno armonico y otro completo.
FEATURE_REPRESENTATION = "harmonic_full"

# Si es True, normaliza cada banda de frecuencia del espectrograma antes de
# darselo a la CNN. Ayuda a reducir diferencias de nivel entre grabaciones.
USE_FREQUENCY_NORMALIZATION = True

# Si es True, muestra las graficas de loss, precision, recall, AUC-PR y F1.
# Puede desactivarse si se quiere ejecutar el script de forma mas automatizada.
SHOW_TRAINING_PLOTS = True

# Si es True, busca un umbral de referencia para convertir probabilidades
# en decisiones binarias solo con fines de analisis.
USE_THRESHOLD_ANALYSIS = True

# Si es True, guarda un fichero JSON con la configuracion de salida del modelo.
# Esto ayuda a documentar como debe integrarse la CNN en produccion.
SAVE_POSTPROCESSING_CONFIG = True

# Si es True, guarda en un .txt las metricas y matrices de confusion de
# validacion y test para poder reutilizarlas en la memoria del proyecto.
SAVE_CONFUSION_REPORT = True

# Si es True, el generador de entrenamiento intenta construir lotes con el
# mismo numero de chunks por clase. Solo se aplica a train, nunca a val/test.
USE_BALANCED_CHUNK_BATCHES = True

# Numero objetivo de chunks por lote cuando se activa el balanceo anterior.
TRAIN_CHUNK_BATCH_SIZE = 64

# Columnas usadas para que train/validation/test mantengan proporciones mas
# comparables sin romper la agrupacion por `grupo_seguro`.
SPLIT_STRATIFY_COLUMNS = ("label", "domain")

# La CNN se hace algo mas ancha que la version inicial para reducir
# infraajuste, manteniendo un coste razonable para CPU.
CONV_FILTERS = (16, 32, 64)
DENSE_UNITS = 32


def get_num_input_channels():
    """Numero de canales de entrada segun la representacion espectral elegida."""
    if FEATURE_REPRESENTATION in {"harmonic", "full"}:
        return 1
    if FEATURE_REPRESENTATION == "harmonic_full":
        return 2
    raise ValueError(
        "FEATURE_REPRESENTATION debe ser 'harmonic', 'full' o 'harmonic_full'."
    )


INPUT_SHAPE = (359, 17, get_num_input_channels())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata", "master_index.csv")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_BASENAME = f"modelo_sirenas_margin_3_{RUN_TIMESTAMP}"
MODEL_PATH = os.path.join(SCRIPT_DIR, f"{RUN_BASENAME}.keras")
POSTPROCESSING_PATH = os.path.join(SCRIPT_DIR, f"{RUN_BASENAME}_postprocesado.json")
CONFUSION_REPORT_PATH = os.path.join(SCRIPT_DIR, f"{RUN_BASENAME}_matrices_confusion.txt")

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


def augment_audio_chunk(audio_chunk, sr, rng):
    """
    Crea variaciones realistas de un chunk solo para entrenamiento.

    La idea es exponer la red a pequenas variaciones de volumen, ruido,
    reverberacion y afinacion para que generalice mejor fuera del dataset.
    """
    augmented = np.copy(audio_chunk).astype(np.float32)

    if rng.random() < 0.8:
        augmented *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.35:
        augmented = add_shaped_noise(augmented, rng)
    if rng.random() < 0.20:
        augmented = librosa.effects.time_stretch(augmented, rate=rng.uniform(0.97, 1.03))
    if rng.random() < 0.20:
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=sr,
            n_steps=rng.uniform(-0.35, 0.35),
        )
    if rng.random() < 0.20:
        augmented = add_reverb(augmented, rng)
    if rng.random() < 0.20:
        augmented = apply_compression(augmented, rng)

    augmented = pad_or_trim(augmented, len(audio_chunk)).astype(np.float32)
    peak = np.max(np.abs(augmented))
    if peak > 1.0:
        augmented /= peak
    return augmented


def normalize_spectrogram(db_spectrogram):
    """
    Normaliza cada banda de frecuencia del espectrograma.

    Esto reduce diferencias de nivel entre grabaciones y ayuda a que la red
    aprenda mas por patron espectral que por volumen absoluto.
    """
    freq_mean = np.mean(db_spectrogram, axis=1, keepdims=True)
    freq_std = np.std(db_spectrogram, axis=1, keepdims=True)
    normalized = (db_spectrogram - freq_mean) / (freq_std + 1e-6)
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def build_feature_tensor_from_stft(stft_matrix):
    """
    Construye la entrada final de la CNN a partir de una STFT.

    Segun `FEATURE_REPRESENTATION`, devuelve:
    - un canal armonico
    - un canal completo
    - o ambos canales apilados
    """
    full_sliced = stft_matrix[:359, :17]
    harmonic, _ = librosa.decompose.hpss(stft_matrix, margin=3.0)
    harmonic_sliced = harmonic[:359, :17]

    full_db = librosa.amplitude_to_db(np.abs(full_sliced), ref=np.max)
    harmonic_db = librosa.amplitude_to_db(np.abs(harmonic_sliced), ref=np.max)

    if USE_FREQUENCY_NORMALIZATION:
        full_db = normalize_spectrogram(full_db)
        harmonic_db = normalize_spectrogram(harmonic_db)

    if FEATURE_REPRESENTATION == "harmonic":
        return np.expand_dims(harmonic_db, axis=-1)
    if FEATURE_REPRESENTATION == "full":
        return np.expand_dims(full_db, axis=-1)
    if FEATURE_REPRESENTATION == "harmonic_full":
        return np.stack([harmonic_db, full_db], axis=-1).astype(np.float32)

    raise ValueError(
        "FEATURE_REPRESENTATION debe ser 'harmonic', 'full' o 'harmonic_full'."
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
    2. Aplicar augmentacion opcional si estamos en entrenamiento.
    3. Calcular la STFT.
    4. Construir la representacion espectral elegida.
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

            # La augmentacion solo se usa durante entrenamiento.
            if augment:
                audio_chunk = augment_audio_chunk(audio_chunk, sr, rng)

            # Se rellena hasta 8192 muestras para obtener siempre 17 frames
            # temporales al aplicar la STFT con estos parametros.
            audio_chunk_padded = np.pad(audio_chunk, (0, 8192 - len(audio_chunk)))
            stft = librosa.stft(audio_chunk_padded, n_fft=1024, hop_length=512, window="hamming")

            # A partir de la STFT se genera la entrada final para la CNN.
            features = build_feature_tensor_from_stft(stft)
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


def make_stratum_keys(df, columns):
    """Combina varias columnas categoricas en una clave de estratificacion."""
    return df.loc[:, columns].fillna("missing").astype(str).agg("|".join, axis=1)


def split_assignment_cost(current_counts, current_rows, target_counts, target_rows):
    """Mide lo lejos que esta un split de sus proporciones objetivo."""
    row_error = ((current_rows - target_rows) / max(1.0, target_rows)) ** 2
    stratum_error = 0.0

    for key, target_value in target_counts.items():
        current_value = current_counts.get(key, 0)
        stratum_error += ((current_value - target_value) / max(1.0, target_value)) ** 2

    return row_error + stratum_error


def grouped_stratified_split(
    df,
    group_col,
    test_size,
    stratify_columns=SPLIT_STRATIFY_COLUMNS,
    random_state=RANDOM_SEED,
):
    """
    Divide un DataFrame en dos subconjuntos manteniendo grupos completos y
    aproximando las proporciones de las columnas indicadas en `stratify_columns`.
    """
    df_local = df.copy()
    df_local["_stratum_key"] = make_stratum_keys(df_local, list(stratify_columns))

    target_test_rows = len(df_local) * float(test_size)
    target_train_rows = len(df_local) - target_test_rows

    total_stratum_counts = df_local["_stratum_key"].value_counts().to_dict()
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
                "counts": group_df["_stratum_key"].value_counts().to_dict(),
                "tie_breaker": float(rng.random()),
            }
        )

    group_summaries.sort(key=lambda item: (-item["size"], item["tie_breaker"]))

    train_indices = []
    test_indices = []
    train_rows = 0
    test_rows = 0
    train_counts = {}
    test_counts = {}

    for summary in group_summaries:
        candidate_train_rows = train_rows + summary["size"]
        candidate_test_rows = test_rows + summary["size"]

        candidate_train_counts = train_counts.copy()
        candidate_test_counts = test_counts.copy()
        for key, value in summary["counts"].items():
            candidate_train_counts[key] = candidate_train_counts.get(key, 0) + value
            candidate_test_counts[key] = candidate_test_counts.get(key, 0) + value

        train_cost = split_assignment_cost(
            candidate_train_counts,
            candidate_train_rows,
            target_train_counts,
            target_train_rows,
        ) + split_assignment_cost(
            test_counts,
            test_rows,
            target_test_counts,
            target_test_rows,
        )
        test_cost = split_assignment_cost(
            train_counts,
            train_rows,
            target_train_counts,
            target_train_rows,
        ) + split_assignment_cost(
            candidate_test_counts,
            candidate_test_rows,
            target_test_counts,
            target_test_rows,
        )

        if test_cost < train_cost:
            test_indices.extend(summary["indices"].tolist())
            test_rows = candidate_test_rows
            test_counts = candidate_test_counts
        else:
            train_indices.extend(summary["indices"].tolist())
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
    print(df.groupby("label")["num_chunks"].sum())
    print(f"Distribucion por audio ({', '.join(stratify_columns)}):")
    print(df.groupby(list(stratify_columns)).size())
    print(f"Distribucion por chunk ({', '.join(stratify_columns)}):")
    print(df.groupby(list(stratify_columns))["num_chunks"].sum())
    print("Top fuentes por audio:")
    print(df["source"].value_counts().head(10))
    print("Top fuentes por chunk:")
    print(df.groupby("source")["num_chunks"].sum().sort_values(ascending=False).head(10))


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
    ):
        super().__init__()
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
    auc_pr = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    negative_chunks = max(1, int(np.sum(y_true == 0)))
    false_alarms_per_min = (fp / negative_chunks) * (60.0 / chunk_step_s)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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
            by=["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    else:
        best_row = threshold_df.sort_values(
            by=["false_alarms_per_min", "f1", "recall"],
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
        "Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
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
            "Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
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
    print("Cargando metadata...")
    df_master = pd.read_csv(METADATA_PATH)

    label_encoder = LabelEncoder()
    df_master["target"] = label_encoder.fit_transform(df_master["label"])
    df_master["grupo_seguro"] = df_master["siren_id"].fillna(df_master["group_id"])

    print(
        "Generando splits agrupados y estratificados por: "
        f"{', '.join(SPLIT_STRATIFY_COLUMNS)}"
    )
    train_idx, temp_idx = grouped_stratified_split(
        df_master,
        group_col="grupo_seguro",
        test_size=0.3,
        stratify_columns=SPLIT_STRATIFY_COLUMNS,
        random_state=RANDOM_SEED,
    )

    train_df = df_master.iloc[train_idx].reset_index(drop=True)
    temp_df = df_master.iloc[temp_idx].reset_index(drop=True)

    val_idx, test_idx = grouped_stratified_split(
        temp_df,
        group_col="grupo_seguro",
        test_size=0.5,
        stratify_columns=SPLIT_STRATIFY_COLUMNS,
        random_state=RANDOM_SEED,
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print("Contando chunks validos por split. Esto puede tardar unos minutos...")
    train_df = add_chunk_count_column(train_df, base_path=DATASET_DIR)
    val_df = add_chunk_count_column(val_df, base_path=DATASET_DIR)
    test_df = add_chunk_count_column(test_df, base_path=DATASET_DIR)

    print(
        f"Archivos originales distribuidos en -> Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | Test: {len(test_df)}"
    )
    print(
        f"Configuracion temporal -> chunk: {CHUNK_LENGTH_S:.2f} s | "
        f"solapamiento: {OVERLAP_S:.2f} s | paso entre decisiones: {CHUNK_STEP_S:.2f} s"
    )
    print("Distribucion de clases en train:")
    print(train_df["label"].value_counts())
    print_split_diagnostics("Train", train_df)
    print_split_diagnostics("Validation", val_df)
    print_split_diagnostics("Test", test_df)

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
    )
    val_gen = AudioDataGenerator(
        val_df,
        base_path=DATASET_DIR,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False,
        balance_chunks=False,
        seed=RANDOM_SEED,
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
            ["threshold", "precision", "recall", "f1", "auc_pr", "false_alarms_per_min"]
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
            f"Feature representation: {FEATURE_REPRESENTATION}",
            f"Split stratify columns: {', '.join(SPLIT_STRATIFY_COLUMNS)}",
            f"Balanced chunk batches: {USE_BALANCED_CHUNK_BATCHES}",
            f"Train chunk batch size: {TRAIN_CHUNK_BATCH_SIZE}",
            f"Effective class weights: {effective_use_class_weights}",
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
                    "feature_representation": FEATURE_REPRESENTATION,
                    "split_stratify_columns": list(SPLIT_STRATIFY_COLUMNS),
                    "conv_filters": list(CONV_FILTERS),
                    "dense_units": DENSE_UNITS,
                    "labels": label_encoder.classes_.tolist(),
                    "output_mode": "chunk_probability",
                    "use_class_weights": USE_CLASS_WEIGHTS,
                    "effective_use_class_weights": effective_use_class_weights,
                    "use_data_augmentation": USE_DATA_AUGMENTATION,
                    "use_frequency_normalization": USE_FREQUENCY_NORMALIZATION,
                    "use_threshold_analysis": USE_THRESHOLD_ANALYSIS,
                    "use_balanced_chunk_batches": USE_BALANCED_CHUNK_BATCHES,
                    "train_chunk_batch_size": TRAIN_CHUNK_BATCH_SIZE,
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
