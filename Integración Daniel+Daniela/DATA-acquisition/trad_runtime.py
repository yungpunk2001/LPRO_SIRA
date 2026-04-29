from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import librosa
import numpy as np

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = UserWarning


DEFAULT_SR = 16000
DEFAULT_N_MFCC = 13
FEATURE_VECTOR_SIZE = 50
POSITIVE_LABEL = "siren"
VALID_POSITIVE_LABELS = {"siren", "sirena"}
REQUIRED_JSON_KEYS = ("sample_rate", "chunk_length_s", "recommended_chunk_threshold")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON de postprocesado no contiene un objeto: {path}")
    return payload


def path_name_from_json_value(value) -> str:
    return Path(str(value).replace("\\", "/")).name


def normalized_bundle_filename(name: str) -> str:
    marker = "clasificador_tradicional_"
    if marker in name:
        return f"{marker}{name.split(marker, 1)[1]}"
    return name


def bundle_filename_prefix(name: str) -> str:
    marker = "clasificador_tradicional_"
    if marker not in name:
        return ""
    return name.split(marker, 1)[0].rstrip("_- ")


def json_references_bundle(payload: dict, bundle_path: Path) -> bool:
    expected_name = normalized_bundle_filename(bundle_path.name)
    saved_bundle_names = {
        normalized_bundle_filename(path_name_from_json_value(value))
        for value in (payload.get("saved_bundles") or {}).values()
    }
    default_bundle_path = payload.get("bundle_path")
    if default_bundle_path:
        saved_bundle_names.add(
            normalized_bundle_filename(path_name_from_json_value(default_bundle_path))
        )

    if not saved_bundle_names:
        return True
    return expected_name in saved_bundle_names


def find_postprocessing_config(bundle_path: Path) -> tuple[dict, Path]:
    same_stem_path = bundle_path.with_name(f"{bundle_path.stem}_postprocesado.json")
    if same_stem_path.exists():
        payload = load_json(same_stem_path)
        if not json_references_bundle(payload, bundle_path):
            raise ValueError(
                "El JSON de postprocesado no corresponde al bundle tradicional.\n"
                f"Bundle: {bundle_path}\n"
                f"JSON: {same_stem_path}"
            )
        return payload, same_stem_path

    candidates = sorted(bundle_path.parent.glob("*_postprocesado.json"))
    if not candidates:
        raise FileNotFoundError(
            "No se encontro el JSON de postprocesado asociado al bundle tradicional.\n"
            f"Bundle: {bundle_path}\n"
            "Copia en la misma carpeta el *_postprocesado.json que genera "
            "entrenar_modelo_clasif_trad.py."
        )

    bundle_prefix = bundle_filename_prefix(bundle_path.name)
    if bundle_prefix:
        prefixed_matches: list[tuple[dict, Path]] = []
        for candidate in candidates:
            if not candidate.name.startswith(bundle_prefix):
                continue
            payload = load_json(candidate)
            if json_references_bundle(payload, bundle_path):
                prefixed_matches.append((payload, candidate))

        if len(prefixed_matches) == 1:
            return prefixed_matches[0]
        if len(prefixed_matches) > 1:
            candidate_text = "\n".join(f"  - {path}" for _, path in prefixed_matches)
            raise ValueError(
                "Hay varios JSON de postprocesado con el mismo prefijo del bundle.\n"
                f"Bundle: {bundle_path}\n"
                f"Prefijo: {bundle_prefix}\n"
                f"Candidatos:\n{candidate_text}"
            )

    matching_candidates: list[tuple[dict, Path]] = []
    for candidate in candidates:
        payload = load_json(candidate)
        if json_references_bundle(payload, bundle_path):
            matching_candidates.append((payload, candidate))

    if len(matching_candidates) == 1:
        return matching_candidates[0]

    if len(matching_candidates) > 1:
        candidate_text = "\n".join(f"  - {path}" for _, path in matching_candidates)
        raise ValueError(
            "Hay varios JSON de postprocesado compatibles con el mismo bundle.\n"
            f"Bundle: {bundle_path}\n"
            f"Candidatos:\n{candidate_text}"
        )

    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    raise ValueError(
        "Ningun JSON de postprocesado de la carpeta referencia el bundle indicado.\n"
        f"Bundle: {bundle_path}\n"
        f"JSON encontrados:\n{candidate_text}"
    )


def validate_postprocessing_config(payload: dict, config_path: Path) -> dict:
    missing_keys = [key for key in REQUIRED_JSON_KEYS if key not in payload]
    if "overlap_s" not in payload and "decision_step_s" not in payload:
        missing_keys.append("overlap_s|decision_step_s")
    if missing_keys:
        raise ValueError(
            "El JSON de postprocesado tradicional no incluye toda la configuracion "
            "necesaria para inferir con seguridad.\n"
            f"JSON: {config_path}\n"
            f"Campos ausentes: {', '.join(missing_keys)}"
        )

    positive_label = str(payload.get("positive_label", POSITIVE_LABEL)).lower()
    if positive_label not in VALID_POSITIVE_LABELS:
        raise ValueError(
            "El JSON de postprocesado tradicional no usa 'siren' como clase positiva.\n"
            f"JSON: {config_path}\n"
            f"positive_label={payload.get('positive_label')!r}"
        )

    sample_rate = int(payload["sample_rate"])
    chunk_length_s = float(payload["chunk_length_s"])
    if "overlap_s" in payload:
        overlap_s = float(payload["overlap_s"])
        decision_step_s = float(payload.get("decision_step_s", chunk_length_s - overlap_s))
    else:
        decision_step_s = float(payload["decision_step_s"])
        overlap_s = chunk_length_s - decision_step_s

    if sample_rate <= 0:
        raise ValueError(f"sample_rate debe ser positivo en {config_path}.")
    if chunk_length_s <= 0:
        raise ValueError(f"chunk_length_s debe ser positivo en {config_path}.")
    if overlap_s < 0:
        raise ValueError(f"overlap_s no puede ser negativo en {config_path}.")
    if overlap_s >= chunk_length_s:
        raise ValueError(f"overlap_s debe ser menor que chunk_length_s en {config_path}.")
    if decision_step_s <= 0:
        raise ValueError(f"decision_step_s debe ser positivo en {config_path}.")

    return {
        "sample_rate": sample_rate,
        "chunk_length_s": chunk_length_s,
        "overlap_s": overlap_s,
        "decision_step_s": decision_step_s,
        "json_chunk_threshold": float(payload["recommended_chunk_threshold"]),
        "positive_label": POSITIVE_LABEL,
        "labels": [str(label) for label in payload.get("labels", ["background", "siren"])],
        "raw_payload": payload,
    }


def values_equal(left, right) -> bool:
    if left == right:
        return True
    if str(left).lower() == str(right).lower():
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def resolve_probability_index(classes, positive_class) -> int:
    for index, class_value in enumerate(np.asarray(classes).reshape(-1)):
        if values_equal(class_value, positive_class):
            return int(index)
    raise ValueError(
        "No se pudo localizar la clase positiva en model.classes_. "
        f"classes_={list(classes)!r}, positive_class={positive_class!r}"
    )


def resolve_positive_class_value(bundle: dict, metadata: dict):
    if metadata.get("positive_class_encoded") is not None:
        return metadata["positive_class_encoded"]

    positive_label = metadata.get("positive_label", POSITIVE_LABEL)
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "transform"):
        try:
            return label_encoder.transform([positive_label])[0]
        except Exception:
            pass
    return positive_label


def force_sequential_inference(model) -> None:
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass


def load_and_prepare_bundle(bundle_path: Path) -> dict:
    load_warnings = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(bundle_path)
    for warning in caught_warnings:
        if issubclass(warning.category, InconsistentVersionWarning):
            warning_text = str(warning.message).splitlines()[0]
            if warning_text not in load_warnings:
                load_warnings.append(warning_text)

    if not isinstance(bundle, dict):
        raise ValueError(f"El bundle tradicional no contiene un diccionario: {bundle_path}")
    for required_key in ("model", "scaler", "metadata"):
        if required_key not in bundle:
            raise ValueError(
                f"El bundle tradicional no incluye '{required_key}': {bundle_path}"
            )

    metadata = dict(bundle.get("metadata") or {})
    positive_label = str(metadata.get("positive_label", POSITIVE_LABEL)).lower()
    if positive_label not in VALID_POSITIVE_LABELS:
        raise ValueError(
            "El bundle tradicional no usa 'siren' como clase positiva.\n"
            f"Bundle: {bundle_path}\n"
            f"positive_label={metadata.get('positive_label')!r}"
        )

    model = bundle["model"]
    force_sequential_inference(model)
    positive_class_value = resolve_positive_class_value(bundle, metadata)
    bundle["metadata"] = metadata
    bundle["positive_class_value"] = positive_class_value
    bundle["positive_label"] = POSITIVE_LABEL
    bundle["has_predict_proba"] = hasattr(model, "predict_proba")
    bundle["load_warnings"] = load_warnings
    if bundle["has_predict_proba"]:
        bundle["positive_probability_index"] = resolve_probability_index(
            model.classes_,
            positive_class_value,
        )
    else:
        bundle["positive_probability_index"] = None
    return bundle


def assert_temporal_consistency(metadata: dict, runtime_config: dict, bundle_path: Path) -> None:
    checks = (
        ("sample_rate", "sample_rate"),
        ("chunk_seconds", "chunk_length_s"),
        ("overlap_seconds", "overlap_s"),
        ("decision_step_seconds", "decision_step_s"),
    )
    for metadata_key, runtime_key in checks:
        if metadata_key not in metadata:
            continue
        metadata_value = float(metadata[metadata_key])
        runtime_value = float(runtime_config[runtime_key])
        if abs(metadata_value - runtime_value) > 1e-6:
            raise ValueError(
                "El JSON de postprocesado y la metadata del bundle no coinciden.\n"
                f"Bundle: {bundle_path}\n"
                f"Campo: {metadata_key}={metadata_value}, JSON={runtime_value}"
            )


def build_runtime_config(bundle: dict, json_config: dict, bundle_path: Path) -> dict:
    metadata = bundle["metadata"]
    assert_temporal_consistency(metadata, json_config, bundle_path)

    if metadata.get("recommended_threshold") is None:
        raise ValueError(
            "El bundle no incluye recommended_threshold. El JSON tradicional actual "
            "solo guarda el umbral ganador del entrenamiento, asi que no permite "
            "recuperar con seguridad el umbral especifico de este modelo.\n"
            f"Bundle: {bundle_path}"
        )

    chunk_threshold = float(metadata["recommended_threshold"])
    sample_rate = int(json_config["sample_rate"])
    chunk_length_s = float(json_config["chunk_length_s"])
    decision_step_s = float(json_config["decision_step_s"])
    chunk_samples = int(round(chunk_length_s * sample_rate))
    step_samples = int(round(decision_step_s * sample_rate))
    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError(
            "La configuracion temporal tradicional produce tamanos invalidos.\n"
            f"Bundle: {bundle_path}\n"
            f"chunk_samples={chunk_samples}, step_samples={step_samples}"
        )

    return {
        "model_name": str(metadata.get("model_name") or metadata.get("winner_name") or bundle_path.stem),
        "winner_name": str(metadata.get("winner_name") or ""),
        "bundle_role": str(metadata.get("bundle_role") or ""),
        "sample_rate": sample_rate,
        "chunk_length_s": chunk_length_s,
        "overlap_s": float(json_config["overlap_s"]),
        "decision_step_s": decision_step_s,
        "chunk_samples": chunk_samples,
        "step_samples": step_samples,
        "chunk_threshold": chunk_threshold,
        "json_chunk_threshold": float(json_config["json_chunk_threshold"]),
        "threshold_source": "bundle_metadata",
        "positive_label": POSITIVE_LABEL,
        "has_probability": bool(bundle["has_predict_proba"]),
        "load_warnings": list(bundle.get("load_warnings", [])),
    }


def load_traditional_bundle(bundle_path: str | Path) -> tuple[dict, dict, str]:
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"No se encontro el bundle tradicional: {bundle_path}")

    payload, config_path = find_postprocessing_config(bundle_path)
    json_config = validate_postprocessing_config(payload, config_path)
    bundle = load_and_prepare_bundle(bundle_path)
    runtime_config = build_runtime_config(bundle, json_config, bundle_path)
    return bundle, runtime_config, str(config_path)


def extract_feature_vector(
    y_chunk: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
) -> np.ndarray:
    y_chunk = np.asarray(y_chunk, dtype=np.float32).reshape(-1)
    if y_chunk.size == 0:
        raise ValueError("El fragmento de audio esta vacio.")

    spectrum = np.abs(librosa.stft(y_chunk))
    mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=n_mfcc)
    centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y_chunk, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=spectrum, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_chunk)
    rms = librosa.feature.rms(y=y_chunk)

    features_vector = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(centroid, axis=1),
            np.std(centroid, axis=1),
            np.mean(bandwidth, axis=1),
            np.std(bandwidth, axis=1),
            np.mean(rolloff, axis=1),
            np.std(rolloff, axis=1),
            np.mean(contrast, axis=1),
            np.std(contrast, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1),
            np.mean(rms, axis=1),
            np.std(rms, axis=1),
        ]
    ).astype(np.float32)

    if features_vector.shape[0] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"Se esperaban {FEATURE_VECTOR_SIZE} features y se obtuvieron "
            f"{features_vector.shape[0]}."
        )
    return features_vector


def class_value_is_positive(value, bundle: dict) -> bool:
    if values_equal(value, bundle["positive_class_value"]):
        return True
    return values_equal(value, bundle["positive_label"])


def predict_traditional(bundle: dict, features: np.ndarray, threshold: float) -> dict:
    x_scaled = bundle["scaler"].transform(features.reshape(1, -1))
    model = bundle["model"]

    if bundle.get("has_predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_scaled))
        # Usamos explicitamente p(siren), no max(predict_proba) ni p(background).
        probability = float(probabilities[0, bundle["positive_probability_index"]])
        return {
            "probability": probability,
            "positive": probability >= float(threshold),
            "label": POSITIVE_LABEL if probability >= float(threshold) else "background",
        }

    prediction = model.predict(x_scaled)[0]
    is_positive = class_value_is_positive(prediction, bundle)
    return {
        "probability": None,
        "positive": bool(is_positive),
        "label": POSITIVE_LABEL if is_positive else "background",
    }
