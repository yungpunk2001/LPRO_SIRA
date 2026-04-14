import csv
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Barrido dirigido para `entrenar_modelo_margin_3.py`.
#
# Objetivo:
# 1. Partir de las dos semillas mas relevantes del barrido anterior:
#    - exp002: harmonic_full + balanceo por chunks
#    - exp006: full + sin balanceo por chunks
# 2. Comparar de forma limpia:
#    - ventanas de 0.5 s frente a 1.0 s
#    - frontend lineal frente a log-mel
# 3. Espejar la matriz anterior con data augmentation para medir si aqui ayuda.
# 4. Recuperar la familia de exp008 "combinacion_prometedora" con la misma
#    matriz temporal/espectral para compararla contra los seeds anteriores.
# 5. Mantener unas pocas variantes exploratorias separadas del bloque central.
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT_PATH = SCRIPT_DIR / "entrenar_modelo_margin_3.py"
RUNS_ROOT_DIR = SCRIPT_DIR / "Modelos Atlas"


# ---------------------------------------------------------------------------
# Configuracion general del barrido
# ---------------------------------------------------------------------------
PYTHON_COMMAND = ["py", "-3"]
CONTINUE_ON_ERROR = True
STREAM_TRAINING_LOGS = True
MAX_EXPERIMENTS = None

PRIMARY_RANK_METRIC = "validation_f2"
SECONDARY_RANK_METRIC = "validation_recall"
TERTIARY_RANK_METRIC = "validation_auc_pr"

# Si es True, al terminar una ronda se genera un reporte y, si hay variantes
# nuevas no probadas, se lanza otra ronda automaticamente a partir de las
# mejores configuraciones encontradas.
AUTO_ITERATIVE_REFINEMENT = True

# Numero maximo total de rondas. 1 = solo barrido inicial; 2 = inicial + una
# ronda de refinamiento; 3 = dos refinamientos, etc.
MAX_SWEEP_ROUNDS = 2

# Cuantos padres se eligen tras cada ronda para generar la siguiente.
REFINEMENT_TOP_K = 4

# Numero maximo de hijos derivados de cada configuracion seleccionada.
REFINEMENT_MAX_CHILDREN_PER_PARENT = 6

# Cuantos experimentos mostrar en cada reporte de ronda.
ROUND_REPORT_TOP_N = 8


# ---------------------------------------------------------------------------
# Constantes compartidas para generar comparaciones coherentes.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
HOP_LENGTH = 512
DEFAULT_MEL_BINS = 128
CORE_CHUNK_LENGTHS_S = [0.5, 1.0]
CORE_FRONTENDS = ["linear_stft", "log_mel"]
CORE_SEED_KEYS = ["exp002", "exp006"]
AUGMENTATION_MIRROR_SEED_KEYS = ["exp002", "exp006"]
EXP008_SEED_KEYS = ["exp008"]


# ---------------------------------------------------------------------------
# Overrides fijos aplicados a todos los experimentos del barrido.
# ---------------------------------------------------------------------------
FIXED_OVERRIDES = {
    "RANDOM_SEED": 42,
    "SHOW_TRAINING_PLOTS": False,
    "USE_THRESHOLD_ANALYSIS": True,
    "SAVE_POSTPROCESSING_CONFIG": True,
    "SAVE_CONFUSION_REPORT": False,
    "USE_CLASS_WEIGHTS": True,
    "TRAIN_CHUNK_BATCH_SIZE": 64,
    "OVERLAP_S": 0.0,
}


# ---------------------------------------------------------------------------
# Semillas heredadas del barrido anterior.
# ---------------------------------------------------------------------------
BASE_EXPERIMENT_SEEDS = {
    "exp002": {
        "base_experiment_label": "solo_balanceo_chunks",
        "base_summary": "Seed exp002: harmonic_full con balanceo por chunks.",
        "overrides": {
            "FEATURE_REPRESENTATION": "harmonic_full",
            "SPECTROGRAM_NORMALIZATION": "minmax",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": True,
            "USE_DATA_AUGMENTATION": False,
        },
    },
    "exp006": {
        "base_experiment_label": "solo_espectrograma_completo",
        "base_summary": "Seed exp006: espectrograma completo sin balanceo por chunks.",
        "overrides": {
            "FEATURE_REPRESENTATION": "full",
            "SPECTROGRAM_NORMALIZATION": "minmax",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": False,
            "USE_DATA_AUGMENTATION": False,
        },
    },
    "exp008": {
        "base_experiment_label": "combinacion_prometedora",
        "base_summary": (
            "Seed exp008: harmonic_full con normalizacion frequency, "
            "balanceo por chunks y data augmentation."
        ),
        "overrides": {
            "FEATURE_REPRESENTATION": "harmonic_full",
            "SPECTROGRAM_NORMALIZATION": "frequency",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": True,
            "USE_DATA_AUGMENTATION": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Variantes extra fuera de las matrices principales.
# ---------------------------------------------------------------------------
EXPLORATORY_VARIANTS = [
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "linear_stft",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Transfiere al mejor seed el balanceo por chunks que si ayudo en exp002.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 1.0,
        "frontend": "linear_stft",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Comprueba si 1.0 s y balanceo por chunks se refuerzan mutuamente.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "linear_stft",
        "label_suffix": "frequency_norm",
        "extra_overrides": {
            "SPECTROGRAM_NORMALIZATION": "frequency",
        },
        "notes": "Reevalua la normalizacion por frecuencia sobre la familia full.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "log_mel",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Comprueba si log-mel gana recall cuando se equilibra por chunks.",
    },
]


def seconds_tag(seconds):
    """Convierte 0.5 -> 0p5s para etiquetas de fichero y CSV."""
    return f"{seconds:.1f}".replace(".", "p") + "s"


def frontend_tag(frontend):
    """Nombre corto del frontend para etiquetas legibles."""
    if frontend == "linear_stft":
        return "linear"
    if frontend == "log_mel":
        return "logmel"
    raise ValueError("Frontend no soportado.")


def compute_time_frames(chunk_length_s, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Ajusta TIME_FRAMES a la duracion real del chunk para que 1.0 s no recorte
    la mitad del contexto temporal.
    """
    chunk_samples = int(round(chunk_length_s * sample_rate))
    return max(1, int(math.ceil(chunk_samples / hop_length)) + 1)


def build_temporal_overrides(chunk_length_s):
    """Genera overrides temporales coherentes para cada tamano de ventana."""
    return {
        "CHUNK_LENGTH_S": chunk_length_s,
        "OVERLAP_S": 0.0,
        "TIME_FRAMES": compute_time_frames(chunk_length_s),
    }


def build_frontend_overrides(frontend):
    """Devuelve los overrides necesarios para cada frontend espectral."""
    if frontend == "linear_stft":
        return {
            "SPECTRAL_FRONTEND": "linear_stft",
        }

    if frontend == "log_mel":
        return {
            "SPECTRAL_FRONTEND": "log_mel",
            "MEL_BINS": DEFAULT_MEL_BINS,
        }

    raise ValueError("SPECTRAL_FRONTEND debe ser 'linear_stft' o 'log_mel'.")


def build_experiment_notes(base_seed_key, frontend, notes):
    """Une una descripcion corta del seed y del cambio que se esta probando."""
    seed_summary = BASE_EXPERIMENT_SEEDS[base_seed_key]["base_summary"]
    frontend_note = (
        "Frontend lineal con la representacion heredada del seed."
        if frontend == "linear_stft"
        else "Frontend log-mel; FEATURE_REPRESENTATION queda solo como trazabilidad."
    )
    note_parts = [seed_summary, frontend_note]
    if notes:
        note_parts.append(notes)
    return " ".join(note_parts)


def make_experiment(
    base_seed_key,
    chunk_length_s,
    frontend,
    tier,
    notes=None,
    label_suffix=None,
    extra_overrides=None,
):
    """Construye una configuracion de experimento autocontenida y trazable."""
    base_seed = BASE_EXPERIMENT_SEEDS[base_seed_key]
    chunk_tag = seconds_tag(chunk_length_s)
    short_frontend = frontend_tag(frontend)
    label_parts = [base_seed_key, chunk_tag, short_frontend, tier]
    if label_suffix:
        label_parts.append(label_suffix)
    experiment_label = "_".join(label_parts)

    overrides = {
        "EXPERIMENT_LABEL": experiment_label,
        "EXPERIMENT_TIER": tier,
        "BASE_EXPERIMENT_ID": base_seed_key,
        "BASE_EXPERIMENT_LABEL": base_seed["base_experiment_label"],
        "COMPARISON_WINDOW_S": chunk_length_s,
        "COMPARISON_WINDOW_TAG": chunk_tag,
        "COMPARISON_FRONTEND": frontend,
        "COMPARISON_FRONTEND_TAG": short_frontend,
        "EXPERIMENT_NOTES": build_experiment_notes(base_seed_key, frontend, notes),
    }

    overrides.update(base_seed["overrides"])
    overrides.update(build_temporal_overrides(chunk_length_s))
    overrides.update(build_frontend_overrides(frontend))

    if extra_overrides:
        overrides.update(extra_overrides)

    return overrides


def build_family_matrix(seed_keys, tier, notes, extra_overrides=None, label_suffix=None):
    """Genera una matriz completa seed x ventana x frontend con una etiqueta comun."""
    experiments = []

    for base_seed_key in seed_keys:
        for chunk_length_s in CORE_CHUNK_LENGTHS_S:
            for frontend in CORE_FRONTENDS:
                experiments.append(
                    make_experiment(
                        base_seed_key=base_seed_key,
                        chunk_length_s=chunk_length_s,
                        frontend=frontend,
                        tier=tier,
                        notes=notes,
                        label_suffix=label_suffix,
                        extra_overrides=extra_overrides,
                    )
                )
    return experiments


def build_experiment_grid():
    """Genera las matrices principales y las variantes exploratorias adicionales."""
    experiments = []

    experiments.extend(
        build_family_matrix(
            seed_keys=CORE_SEED_KEYS,
            tier="core",
            notes="Parte de la matriz principal para comparar ventana y frontend.",
        )
    )

    experiments.extend(
        build_family_matrix(
            seed_keys=AUGMENTATION_MIRROR_SEED_KEYS,
            tier="augment",
            notes=(
                "Espejo directo del bloque core con data augmentation activada "
                "para aislar su impacto."
            ),
            extra_overrides={"USE_DATA_AUGMENTATION": True},
        )
    )

    experiments.extend(
        build_family_matrix(
            seed_keys=EXP008_SEED_KEYS,
            tier="exp008",
            notes=(
                "Familia derivada de exp008 combinacion_prometedora para probar "
                "ventana y frontend sobre esa configuracion base."
            ),
        )
    )

    for variant in EXPLORATORY_VARIANTS:
        experiments.append(
            make_experiment(
                base_seed_key=variant["base_seed_key"],
                chunk_length_s=variant["chunk_length_s"],
                frontend=variant["frontend"],
                tier="explore",
                notes=variant["notes"],
                label_suffix=variant.get("label_suffix"),
                extra_overrides=variant.get("extra_overrides"),
            )
        )

    return experiments


def parse_conv_filters(value):
    """Recupera la lista de filtros desde el resumen compacto."""
    if isinstance(value, str):
        return list(json.loads(value))
    if value is None:
        return [16, 32, 64]
    return list(value)


def build_experiment_signature_from_overrides(overrides):
    """Devuelve una firma canonica para evitar repetir la misma configuracion."""
    frontend = overrides.get("SPECTRAL_FRONTEND", "linear_stft")
    use_augmentation = bool(overrides.get("USE_DATA_AUGMENTATION", False))
    use_spectral_eq = bool(overrides.get("USE_SPECTRAL_EQ_AUGMENTATION", True))

    return (
        frontend,
        overrides.get("FEATURE_REPRESENTATION"),
        overrides.get("SPECTROGRAM_NORMALIZATION"),
        tuple(overrides.get("CONV_FILTERS", [])),
        int(overrides.get("DENSE_UNITS", 32)),
        round(float(overrides.get("CHUNK_LENGTH_S", 0.5)), 4),
        round(float(overrides.get("OVERLAP_S", 0.0)), 4),
        int(overrides.get("TIME_FRAMES", compute_time_frames(overrides.get("CHUNK_LENGTH_S", 0.5)))),
        int(overrides.get("MEL_BINS", DEFAULT_MEL_BINS)) if frontend == "log_mel" else None,
        bool(overrides.get("USE_BALANCED_CHUNK_BATCHES", False)),
        bool(overrides.get("USE_CLASS_WEIGHTS", True)),
        use_augmentation,
        (
            round(float(overrides.get("AUGMENTATION_APPLY_PROB", 0.5)), 4)
            if use_augmentation
            else None
        ),
        use_spectral_eq if use_augmentation else None,
        (
            round(float(overrides.get("EQ_AUGMENTATION_PROB", 0.20)), 4)
            if use_augmentation and use_spectral_eq
            else None
        ),
    )


def derive_overrides_from_row(row):
    """Reconstruye una configuracion reutilizable a partir de la fila resumen."""
    chunk_length_s = float(row.get("config_chunk_length_s") or row.get("comparison_window_s") or 0.5)
    frontend = row.get("config_spectral_frontend") or row.get("comparison_frontend") or "linear_stft"
    mel_bins = row.get("config_mel_bins")
    augmentation_apply_probability = row.get("config_augmentation_apply_probability")
    eq_augmentation_probability = row.get("config_eq_augmentation_probability")
    use_spectral_eq_augmentation = row.get("config_use_spectral_eq_augmentation")

    overrides = {
        "FEATURE_REPRESENTATION": row.get("config_feature_representation_requested"),
        "SPECTROGRAM_NORMALIZATION": row.get("config_spectrogram_normalization"),
        "CONV_FILTERS": parse_conv_filters(row.get("config_conv_filters")),
        "DENSE_UNITS": int(row.get("config_dense_units") or 32),
        "USE_BALANCED_CHUNK_BATCHES": bool(row.get("config_use_balanced_chunk_batches")),
        "USE_DATA_AUGMENTATION": bool(row.get("config_use_data_augmentation")),
        "USE_CLASS_WEIGHTS": bool(
            FIXED_OVERRIDES.get("USE_CLASS_WEIGHTS", True)
            if row.get("config_use_class_weights") is None
            else row.get("config_use_class_weights")
        ),
        "TRAIN_CHUNK_BATCH_SIZE": int(
            row.get("config_train_chunk_batch_size") or FIXED_OVERRIDES["TRAIN_CHUNK_BATCH_SIZE"]
        ),
        "SPECTRAL_FRONTEND": frontend,
    }
    overrides.update(build_temporal_overrides(chunk_length_s))

    if frontend == "log_mel":
        overrides["MEL_BINS"] = int(mel_bins or DEFAULT_MEL_BINS)

    if overrides["USE_DATA_AUGMENTATION"]:
        overrides["AUGMENTATION_APPLY_PROB"] = float(
            0.5 if augmentation_apply_probability is None else augmentation_apply_probability
        )
        overrides["USE_SPECTRAL_EQ_AUGMENTATION"] = bool(
            True if use_spectral_eq_augmentation is None else use_spectral_eq_augmentation
        )
        if overrides["USE_SPECTRAL_EQ_AUGMENTATION"]:
            overrides["EQ_AUGMENTATION_PROB"] = float(
                0.20 if eq_augmentation_probability is None else eq_augmentation_probability
            )

    return overrides


def build_experiment_signature_from_row(row):
    """Firma canonica a partir de una fila de resultados ya evaluada."""
    return build_experiment_signature_from_overrides(derive_overrides_from_row(row))


def build_round_dir_name(round_number):
    """Nombre de directorio estable para cada ronda."""
    return f"ronda_{round_number:02d}"


def format_row_compact(row, rank=None):
    """Representacion corta y legible de una fila para consola y reportes."""
    rank_prefix = f"{rank}. " if rank is not None else ""
    return (
        f"{rank_prefix}{row['experiment_id']} ({row['experiment_label']}) | "
        f"round={row.get('sweep_round')} | "
        f"tier={row.get('experiment_tier')} | "
        f"base={row.get('base_experiment_id')} | "
        f"frontend={row.get('config_spectral_frontend')} | "
        f"window={row.get('config_chunk_length_s')} s | "
        f"norm={row.get('config_spectrogram_normalization')} | "
        f"balanced={row.get('config_use_balanced_chunk_batches')} | "
        f"augment={row.get('config_use_data_augmentation')} | "
        f"val_f2={safe_metric(row, 'validation_f2', default=0.0):.4f} | "
        f"val_recall={safe_metric(row, 'validation_recall', default=0.0):.4f} | "
        f"val_auc_pr={safe_metric(row, 'validation_auc_pr', default=0.0):.4f} | "
        f"fa/min={safe_metric(row, 'validation_false_alarms_per_min', default=0.0):.2f}"
    )


def safe_metric(row, key, default=-1.0):
    """Devuelve una metrica numerica o un valor por defecto si falta."""
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def get_effective_frontend(overrides, postprocess_data):
    """Obtiene el frontend real del experimento aunque falten datos en el JSON."""
    return postprocess_data.get(
        "spectral_frontend",
        overrides.get("SPECTRAL_FRONTEND", "linear_stft"),
    )


def get_effective_feature_representation(overrides, postprocess_data):
    """Indica la representacion efectiva usada por la CNN en cada frontend."""
    frontend = get_effective_frontend(overrides, postprocess_data)
    if frontend == "log_mel":
        return "n/a_log_mel"
    return postprocess_data.get(
        "feature_representation",
        overrides.get("FEATURE_REPRESENTATION"),
    )


def build_summary_row(
    experiment_id,
    overrides,
    postprocess_data,
    status,
    return_code,
    log_path,
    sweep_round,
    round_dir,
):
    """Compacta los resultados de un experimento en una sola fila."""
    validation_metrics = postprocess_data.get("validation_metrics") or {}
    test_metrics = postprocess_data.get("test_metrics") or {}

    return {
        "experiment_id": experiment_id,
        "experiment_label": overrides.get("EXPERIMENT_LABEL"),
        "sweep_round": sweep_round,
        "round_dir": str(round_dir),
        "experiment_tier": overrides.get("EXPERIMENT_TIER"),
        "base_experiment_id": overrides.get("BASE_EXPERIMENT_ID"),
        "base_experiment_label": overrides.get("BASE_EXPERIMENT_LABEL"),
        "parent_experiment_id": overrides.get("PARENT_EXPERIMENT_ID"),
        "parent_experiment_label": overrides.get("PARENT_EXPERIMENT_LABEL"),
        "parent_sweep_round": overrides.get("PARENT_SWEEP_ROUND"),
        "comparison_window_s": overrides.get("COMPARISON_WINDOW_S"),
        "comparison_window_tag": overrides.get("COMPARISON_WINDOW_TAG"),
        "comparison_frontend": overrides.get("COMPARISON_FRONTEND"),
        "comparison_frontend_tag": overrides.get("COMPARISON_FRONTEND_TAG"),
        "experiment_notes": overrides.get("EXPERIMENT_NOTES"),
        "status": status,
        "return_code": return_code,
        "config_chunk_length_s": postprocess_data.get(
            "chunk_length_s",
            overrides.get("CHUNK_LENGTH_S"),
        ),
        "config_overlap_s": postprocess_data.get(
            "overlap_s",
            overrides.get("OVERLAP_S"),
        ),
        "config_decision_step_s": postprocess_data.get("decision_step_s"),
        "config_spectral_frontend": get_effective_frontend(overrides, postprocess_data),
        "config_feature_representation_requested": overrides.get("FEATURE_REPRESENTATION"),
        "config_feature_representation_effective": get_effective_feature_representation(
            overrides,
            postprocess_data,
        ),
        "config_spectrogram_normalization": postprocess_data.get(
            "spectrogram_normalization",
            overrides.get("SPECTROGRAM_NORMALIZATION"),
        ),
        "config_conv_filters": json.dumps(
            postprocess_data.get("conv_filters", overrides.get("CONV_FILTERS"))
        ),
        "config_dense_units": postprocess_data.get(
            "dense_units",
            overrides.get("DENSE_UNITS"),
        ),
        "config_time_frames": postprocess_data.get(
            "time_frames",
            overrides.get("TIME_FRAMES"),
        ),
        "config_padded_chunk_samples": postprocess_data.get("padded_chunk_samples"),
        "config_mel_bins": postprocess_data.get(
            "mel_bins",
            overrides.get("MEL_BINS"),
        ),
        "config_use_class_weights": postprocess_data.get(
            "use_class_weights",
            overrides.get("USE_CLASS_WEIGHTS"),
        ),
        "config_effective_use_class_weights": postprocess_data.get(
            "effective_use_class_weights"
        ),
        "config_use_balanced_chunk_batches": postprocess_data.get(
            "use_balanced_chunk_batches",
            overrides.get("USE_BALANCED_CHUNK_BATCHES"),
        ),
        "config_train_chunk_batch_size": postprocess_data.get(
            "train_chunk_batch_size",
            overrides.get("TRAIN_CHUNK_BATCH_SIZE"),
        ),
        "config_use_data_augmentation": postprocess_data.get(
            "use_data_augmentation",
            overrides.get("USE_DATA_AUGMENTATION"),
        ),
        "config_augmentation_apply_probability": postprocess_data.get(
            "augmentation_apply_probability"
        ),
        "config_use_spectral_eq_augmentation": postprocess_data.get(
            "use_spectral_eq_augmentation"
        ),
        "config_eq_augmentation_probability": postprocess_data.get(
            "eq_augmentation_probability"
        ),
        "config_eq_effective_conditional_probability": postprocess_data.get(
            "eq_effective_conditional_probability"
        ),
        "recommended_chunk_threshold": postprocess_data.get("recommended_chunk_threshold"),
        "model_path": postprocess_data.get("model_path"),
        "postprocess_json_path": postprocess_data.get("postprocess_json_path"),
        "log_path": str(log_path),
        "validation_precision": validation_metrics.get("precision"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1"),
        "validation_f2": validation_metrics.get("f2"),
        "validation_auc_pr": validation_metrics.get("auc_pr"),
        "validation_false_alarms_per_min": validation_metrics.get("false_alarms_per_min"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_f2": test_metrics.get("f2"),
        "test_auc_pr": test_metrics.get("auc_pr"),
        "test_false_alarms_per_min": test_metrics.get("false_alarms_per_min"),
    }


def stream_process_output(process, log_path):
    """Guarda la salida de un entrenamiento en log y opcionalmente la muestra."""
    with open(log_path, "w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            if STREAM_TRAINING_LOGS:
                print(line, end="")


def find_new_postprocess_json(artifacts_dir, run_name_prefix, previous_matches):
    """Localiza el JSON generado por el experimento recien terminado."""
    current_matches = set(
        path.resolve()
        for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
    )
    new_matches = sorted(current_matches - previous_matches)

    if not new_matches:
        return None

    return Path(new_matches[-1])


def rank_rows(rows):
    """Ordena experimentos exitosos por las metricas principales del proyecto."""
    successful_rows = [row for row in rows if row["status"] == "ok"]

    successful_rows.sort(
        key=lambda row: (
            safe_metric(row, PRIMARY_RANK_METRIC),
            safe_metric(row, SECONDARY_RANK_METRIC),
            safe_metric(row, TERTIARY_RANK_METRIC),
            -safe_metric(row, "validation_false_alarms_per_min", default=1e9),
        ),
        reverse=True,
    )
    return successful_rows


def save_summary_files(output_dir, rows, base_name="resumen_experimentos"):
    """Guarda un resumen de experimentos en CSV y JSON."""
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    fieldnames = list(rows[0].keys()) if rows else []

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as json_handle:
        json.dump(rows, json_handle, indent=2)

    return csv_path, json_path


def get_best_result_for_group(rows, predicate):
    """Devuelve el mejor experimento dentro de un subconjunto concreto."""
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return None
    return subset[0]


def print_best_result_for_group(rows, title, predicate):
    """Imprime el mejor experimento dentro de un subconjunto concreto."""
    best_row = get_best_result_for_group(rows, predicate)
    if best_row is None:
        return
    print(f"\n{title}:")
    print(format_row_compact(best_row))


def build_refinement_mutations(parent_row):
    """Devuelve mutaciones ordenadas por prioridad para refinar un buen candidato."""
    parent_overrides = derive_overrides_from_row(parent_row)
    current_chunk_length_s = float(parent_overrides["CHUNK_LENGTH_S"])
    current_frontend = parent_overrides["SPECTRAL_FRONTEND"]
    current_normalization = parent_overrides["SPECTROGRAM_NORMALIZATION"]
    current_balanced_batches = bool(parent_overrides["USE_BALANCED_CHUNK_BATCHES"])
    current_use_augmentation = bool(parent_overrides["USE_DATA_AUGMENTATION"])
    current_augmentation_prob = float(parent_overrides.get("AUGMENTATION_APPLY_PROB", 0.5))
    current_use_spectral_eq = bool(parent_overrides.get("USE_SPECTRAL_EQ_AUGMENTATION", True))
    current_eq_probability = float(parent_overrides.get("EQ_AUGMENTATION_PROB", 0.20))

    target_chunk_length_s = 1.0 if current_chunk_length_s < 0.75 else 0.5
    target_frontend = "log_mel" if current_frontend == "linear_stft" else "linear_stft"
    target_normalization = "frequency" if current_normalization == "minmax" else "minmax"

    mutations = []

    if current_use_augmentation:
        if not math.isclose(current_augmentation_prob, 0.35, rel_tol=0.0, abs_tol=1e-9):
            mutations.append(
                (
                    "aug_p35",
                    {"AUGMENTATION_APPLY_PROB": 0.35},
                    "Reduce la probabilidad de augmentacion para conservar mas chunks limpios.",
                )
            )
        if not math.isclose(current_augmentation_prob, 0.65, rel_tol=0.0, abs_tol=1e-9):
            mutations.append(
                (
                    "aug_p65",
                    {"AUGMENTATION_APPLY_PROB": 0.65},
                    "Aumenta la presencia de augmentacion para exigir mas robustez al modelo.",
                )
            )
        if current_use_spectral_eq:
            mutations.append(
                (
                    "eq_off",
                    {"USE_SPECTRAL_EQ_AUGMENTATION": False},
                    "Mantiene augmentacion general pero elimina la EQ espectral para aislar su efecto.",
                )
            )
            if not math.isclose(current_eq_probability, 0.10, rel_tol=0.0, abs_tol=1e-9):
                mutations.append(
                    (
                        "eq_p10",
                        {"EQ_AUGMENTATION_PROB": 0.10},
                        "Reduce la frecuencia de la EQ espectral dentro de la augmentacion.",
                    )
                )
    else:
        mutations.append(
            (
                "augment_p35",
                {
                    "USE_DATA_AUGMENTATION": True,
                    "AUGMENTATION_APPLY_PROB": 0.35,
                },
                "Activa data augmentation suave para comprobar si ayuda sin dominar el train.",
            )
        )
        mutations.append(
            (
                "augment_p65",
                {
                    "USE_DATA_AUGMENTATION": True,
                    "AUGMENTATION_APPLY_PROB": 0.65,
                },
                "Activa data augmentation intensa para comprobar si mejora la robustez.",
            )
        )

    mutations.append(
        (
            f"norm_{target_normalization}",
            {"SPECTROGRAM_NORMALIZATION": target_normalization},
            "Invierte la normalizacion del espectrograma manteniendo el resto constante.",
        )
    )
    mutations.append(
        (
            "balanced_on" if not current_balanced_batches else "balanced_off",
            {"USE_BALANCED_CHUNK_BATCHES": not current_balanced_batches},
            "Invierte el balanceo por chunks para revisar su impacto local.",
        )
    )
    mutations.append(
        (
            f"window_{seconds_tag(target_chunk_length_s)}",
            build_temporal_overrides(target_chunk_length_s),
            "Cambia la ventana temporal manteniendo el resto de hiperparametros.",
        )
    )
    mutations.append(
        (
            f"frontend_{frontend_tag(target_frontend)}",
            build_frontend_overrides(target_frontend),
            "Cambia solo el frontend espectral para contrastar lineal frente a log-mel.",
        )
    )

    return mutations


def make_refinement_experiment(parent_row, round_number, label_suffix, extra_overrides, notes):
    """Construye un hijo de refinamiento a partir de una configuracion ganadora."""
    base_overrides = derive_overrides_from_row(parent_row)
    merged_overrides = dict(base_overrides)
    merged_overrides.update(extra_overrides)

    chunk_length_s = float(merged_overrides.get("CHUNK_LENGTH_S", 0.5))
    merged_overrides.update(build_temporal_overrides(chunk_length_s))

    frontend = merged_overrides.get("SPECTRAL_FRONTEND", "linear_stft")
    if frontend == "log_mel":
        merged_overrides.setdefault("MEL_BINS", DEFAULT_MEL_BINS)

    parent_round = int(parent_row.get("sweep_round") or max(1, round_number - 1))
    base_experiment_id = parent_row.get("base_experiment_id") or "derived"
    experiment_label = (
        f"{base_experiment_id}_r{round_number:02d}_"
        f"from_r{parent_round:02d}_{parent_row['experiment_id']}_{label_suffix}"
    )

    overrides = {
        "EXPERIMENT_LABEL": experiment_label,
        "EXPERIMENT_TIER": f"refine_r{round_number:02d}",
        "BASE_EXPERIMENT_ID": parent_row.get("base_experiment_id"),
        "BASE_EXPERIMENT_LABEL": parent_row.get("base_experiment_label"),
        "PARENT_EXPERIMENT_ID": parent_row.get("experiment_id"),
        "PARENT_EXPERIMENT_LABEL": parent_row.get("experiment_label"),
        "PARENT_SWEEP_ROUND": parent_row.get("sweep_round"),
        "COMPARISON_WINDOW_S": chunk_length_s,
        "COMPARISON_WINDOW_TAG": seconds_tag(chunk_length_s),
        "COMPARISON_FRONTEND": frontend,
        "COMPARISON_FRONTEND_TAG": frontend_tag(frontend),
        "EXPERIMENT_NOTES": (
            f"Refinamiento automatico desde {parent_row['experiment_label']} "
            f"(ronda {parent_round}). {notes}"
        ),
    }
    overrides.update(merged_overrides)
    return overrides


def select_refinement_parents(ranked_rows, max_parents):
    """Selecciona padres prometedores preservando algo de diversidad por seed base."""
    selected_rows = []
    seen_signatures = set()
    covered_base_experiment_ids = set()

    def try_add_row(row):
        signature = build_experiment_signature_from_row(row)
        if signature in seen_signatures:
            return False
        selected_rows.append(row)
        seen_signatures.add(signature)
        base_experiment_id = row.get("base_experiment_id")
        if base_experiment_id:
            covered_base_experiment_ids.add(base_experiment_id)
        return True

    if ranked_rows:
        try_add_row(ranked_rows[0])

    for row in ranked_rows:
        if len(selected_rows) >= max_parents:
            break
        base_experiment_id = row.get("base_experiment_id")
        if base_experiment_id and base_experiment_id not in covered_base_experiment_ids:
            try_add_row(row)

    for row in ranked_rows:
        if len(selected_rows) >= max_parents:
            break
        try_add_row(row)

    return selected_rows[:max_parents]


def build_refinement_experiment_grid(parent_rows, seen_signatures, round_number):
    """Construye la siguiente ronda a partir de las mejores configuraciones."""
    refinement_experiments = []
    round_signatures = set()

    for parent_row in parent_rows:
        children_added = 0
        for label_suffix, extra_overrides, notes in build_refinement_mutations(parent_row):
            experiment = make_refinement_experiment(
                parent_row=parent_row,
                round_number=round_number,
                label_suffix=label_suffix,
                extra_overrides=extra_overrides,
                notes=notes,
            )
            signature = build_experiment_signature_from_overrides(experiment)
            if signature in seen_signatures or signature in round_signatures:
                continue

            refinement_experiments.append(experiment)
            round_signatures.add(signature)
            children_added += 1

            if children_added >= REFINEMENT_MAX_CHILDREN_PER_PARENT:
                break

    return refinement_experiments


def save_round_report(round_dir, round_number, round_rows, ranked_rows, selected_parents, next_round_grid):
    """Guarda un reporte Markdown corto con lo mejor de la ronda."""
    report_path = round_dir / "reporte_ronda.md"
    successful_rows = [row for row in round_rows if row["status"] == "ok"]
    failed_rows = [row for row in round_rows if row["status"] != "ok"]
    report_lines = [
        f"# Reporte ronda {round_number:02d}",
        "",
        f"- Experimentos lanzados: {len(round_rows)}",
        f"- Experimentos exitosos: {len(successful_rows)}",
        f"- Experimentos con error o sin JSON: {len(failed_rows)}",
    ]

    if ranked_rows:
        best_row = ranked_rows[0]
        report_lines.extend(
            [
                f"- Mejor experimento: {format_row_compact(best_row)}",
                "",
                f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))}",
            ]
        )
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

        report_lines.extend(
            [
                "",
                "## Mejores por grupo",
            ]
        )
        group_definitions = [
            ("Matriz core", lambda row: row["experiment_tier"] == "core"),
            ("Bloque augment", lambda row: row["experiment_tier"] == "augment"),
            ("Familia exp008", lambda row: row["experiment_tier"] == "exp008"),
        ]
        for title, predicate in group_definitions:
            best_row = get_best_result_for_group(ranked_rows, predicate)
            if best_row is not None:
                report_lines.append(f"- {title}: {format_row_compact(best_row)}")
    else:
        report_lines.extend(["", "No hay experimentos exitosos en esta ronda."])

    if selected_parents:
        report_lines.extend(["", "## Padres elegidos para la siguiente ronda"])
        for rank, row in enumerate(selected_parents, start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    if next_round_grid is not None:
        report_lines.extend(
            [
                "",
                "## Siguiente ronda automatica",
                f"- Configuraciones nuevas generadas: {len(next_round_grid)}",
            ]
        )
        for experiment in next_round_grid[:ROUND_REPORT_TOP_N]:
            report_lines.append(
                f"- {experiment['EXPERIMENT_LABEL']} | "
                f"frontend={experiment['SPECTRAL_FRONTEND']} | "
                f"window={experiment['CHUNK_LENGTH_S']} s | "
                f"norm={experiment['SPECTROGRAM_NORMALIZATION']} | "
                f"balanced={experiment['USE_BALANCED_CHUNK_BATCHES']} | "
                f"augment={experiment['USE_DATA_AUGMENTATION']}"
            )

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def save_global_report(run_group_dir, all_rows, round_results):
    """Guarda un reporte global con el consolidado de todas las rondas."""
    report_path = run_group_dir / "reporte_global.md"
    ranked_rows = rank_rows(all_rows)
    report_lines = [
        "# Reporte global del barrido iterativo",
        "",
        f"- Rondas ejecutadas: {len(round_results)}",
        f"- Experimentos totales: {len(all_rows)}",
        f"- Experimentos exitosos: {len(ranked_rows)}",
    ]

    report_lines.extend(["", "## Resumen por ronda"])
    for round_result in round_results:
        round_rows = round_result["rows"]
        round_ranked_rows = round_result["ranked_rows"]
        report_lines.append(
            f"- Ronda {round_result['round_number']:02d}: "
            f"{len(round_rows)} experimentos, "
            f"{len(round_ranked_rows)} exitosos, "
            f"dir={round_result['round_dir']}"
        )
        if round_ranked_rows:
            report_lines.append(f"  Mejor: {format_row_compact(round_ranked_rows[0])}")

    if ranked_rows:
        report_lines.extend(["", f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))} global"])
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def execute_round(experiment_grid, round_number, run_group_dir):
    """Ejecuta una ronda completa de barrido y devuelve su resumen."""
    if MAX_EXPERIMENTS is not None:
        experiment_grid = experiment_grid[:MAX_EXPERIMENTS]

    round_dir = run_group_dir / build_round_dir_name(round_number)
    configs_dir = round_dir / "configs"
    logs_dir = round_dir / "logs"
    artifacts_dir = round_dir / "artefactos"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"Ronda {round_number:02d}")
    print(f"Script de entrenamiento: {TRAINING_SCRIPT_PATH}")
    print(f"Directorio de la ronda: {round_dir}")
    print(f"Numero de experimentos a ejecutar: {len(experiment_grid)}")

    all_rows = []

    for experiment_number, experiment_overrides in enumerate(experiment_grid, start=1):
        experiment_id = f"exp_{experiment_number:03d}"
        experiment_label = experiment_overrides.get("EXPERIMENT_LABEL", experiment_id)
        run_name_prefix = f"{experiment_id}_{experiment_label}"

        applied_overrides = dict(FIXED_OVERRIDES)
        applied_overrides.update(experiment_overrides)
        applied_overrides["RUN_OUTPUT_DIR"] = str(artifacts_dir)
        applied_overrides["RUN_NAME_PREFIX"] = run_name_prefix

        config_path = configs_dir / f"{experiment_id}.json"
        log_path = logs_dir / f"{experiment_id}.log"

        with open(config_path, "w", encoding="utf-8") as config_handle:
            json.dump(applied_overrides, config_handle, indent=2)

        print("\n" + "=" * 80)
        print(f"Ejecutando {experiment_id} -> {experiment_label}")
        print(json.dumps(applied_overrides, indent=2))

        previous_matches = set(
            path.resolve()
            for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
        )

        process_env = os.environ.copy()
        process_env["SIREN_TRAINING_CONFIG_PATH"] = str(config_path)

        process = subprocess.Popen(
            PYTHON_COMMAND + [str(TRAINING_SCRIPT_PATH)],
            cwd=str(SCRIPT_DIR),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stream_process_output(process, log_path)
        return_code = process.wait()

        postprocess_json_path = find_new_postprocess_json(
            artifacts_dir,
            run_name_prefix,
            previous_matches,
        )

        if return_code != 0:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={
                    "postprocess_json_path": (
                        str(postprocess_json_path) if postprocess_json_path else None
                    )
                },
                status="failed",
                return_code=return_code,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(f"El experimento {experiment_id} ha fallado con codigo {return_code}.")

            if not CONTINUE_ON_ERROR:
                break
            continue

        if postprocess_json_path is None:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={},
                status="missing_json",
                return_code=return_code,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(f"El experimento {experiment_id} termino pero no genero JSON de resultados.")

            if not CONTINUE_ON_ERROR:
                break
            continue

        with open(postprocess_json_path, "r", encoding="utf-8") as result_handle:
            postprocess_data = json.load(result_handle)
        postprocess_data["postprocess_json_path"] = str(postprocess_json_path)

        row = build_summary_row(
            experiment_id=experiment_id,
            overrides=applied_overrides,
            postprocess_data=postprocess_data,
            status="ok",
            return_code=return_code,
            log_path=log_path,
            sweep_round=round_number,
            round_dir=round_dir,
        )
        all_rows.append(row)

    ranked_rows = rank_rows(all_rows)
    csv_path, json_path = save_summary_files(round_dir, all_rows)

    print("\n" + "=" * 80)
    print(f"Resumen CSV guardado en: {csv_path}")
    print(f"Resumen JSON guardado en: {json_path}")

    if ranked_rows:
        print("\nMejor experimento de la ronda:")
        print(format_row_compact(ranked_rows[0]))

        print(f"\nTop {min(ROUND_REPORT_TOP_N, len(ranked_rows))} de la ronda:")
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            print(format_row_compact(row, rank=rank))

        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento de la matriz core",
            predicate=lambda row: row["experiment_tier"] == "core",
        )
        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento con data augmentation",
            predicate=lambda row: row["experiment_tier"] == "augment",
        )
        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento de la familia exp008",
            predicate=lambda row: row["experiment_tier"] == "exp008",
        )
    else:
        print("\nNo hay experimentos exitosos para comparar en esta ronda.")

    return {
        "round_number": round_number,
        "round_dir": round_dir,
        "experiment_grid": experiment_grid,
        "rows": all_rows,
        "ranked_rows": ranked_rows,
        "csv_path": csv_path,
        "json_path": json_path,
    }


if __name__ == "__main__":
    run_group_name = datetime.now().strftime(
        "barrido_margin_3_iterativo_%Y%m%d_%H%M%S"
    )
    run_group_dir = RUNS_ROOT_DIR / run_group_name
    run_group_dir.mkdir(parents=True, exist_ok=True)

    current_experiment_grid = build_experiment_grid()
    all_rows_global = []
    round_results = []
    seen_signatures = set()

    print(f"Directorio raiz del barrido: {run_group_dir}")
    print(
        "Modo iterativo: "
        f"{AUTO_ITERATIVE_REFINEMENT} | max rondas: {MAX_SWEEP_ROUNDS} | "
        f"top padres: {REFINEMENT_TOP_K}"
    )

    for round_number in range(1, MAX_SWEEP_ROUNDS + 1):
        if not current_experiment_grid:
            print("\nNo hay configuraciones nuevas que ejecutar. Fin del barrido.")
            break

        round_result = execute_round(
            experiment_grid=current_experiment_grid,
            round_number=round_number,
            run_group_dir=run_group_dir,
        )
        round_results.append(round_result)
        all_rows_global.extend(round_result["rows"])

        for experiment in round_result["experiment_grid"]:
            seen_signatures.add(build_experiment_signature_from_overrides(experiment))

        selected_parents = []
        next_round_grid = None

        if AUTO_ITERATIVE_REFINEMENT and round_number < MAX_SWEEP_ROUNDS:
            selected_parents = select_refinement_parents(
                round_result["ranked_rows"],
                max_parents=REFINEMENT_TOP_K,
            )
            next_round_grid = build_refinement_experiment_grid(
                parent_rows=selected_parents,
                seen_signatures=seen_signatures,
                round_number=round_number + 1,
            )

        round_report_path = save_round_report(
            round_dir=round_result["round_dir"],
            round_number=round_number,
            round_rows=round_result["rows"],
            ranked_rows=round_result["ranked_rows"],
            selected_parents=selected_parents,
            next_round_grid=next_round_grid,
        )
        print(f"\nReporte de ronda guardado en: {round_report_path}")

        if not AUTO_ITERATIVE_REFINEMENT or round_number >= MAX_SWEEP_ROUNDS:
            break

        if not next_round_grid:
            print(
                "\nNo se han generado configuraciones nuevas para la siguiente ronda. "
                "Fin del barrido iterativo."
            )
            break

        print(
            f"\nArrancando automaticamente la ronda {round_number + 1:02d} "
            f"con {len(next_round_grid)} nuevas configuraciones..."
        )
        current_experiment_grid = next_round_grid

    global_csv_path, global_json_path = save_summary_files(
        run_group_dir,
        all_rows_global,
        base_name="resumen_global",
    )
    global_report_path = save_global_report(
        run_group_dir=run_group_dir,
        all_rows=all_rows_global,
        round_results=round_results,
    )

    print("\n" + "#" * 80)
    print(f"Resumen global CSV guardado en: {global_csv_path}")
    print(f"Resumen global JSON guardado en: {global_json_path}")
    print(f"Reporte global guardado en: {global_report_path}")

    global_ranked_rows = rank_rows(all_rows_global)
    if global_ranked_rows:
        print("\nMejor experimento global:")
        print(format_row_compact(global_ranked_rows[0]))
    else:
        print("\nNo hay experimentos exitosos en el barrido iterativo.")
