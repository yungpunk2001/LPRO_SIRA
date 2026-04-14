import warnings

import numpy as np
import sounddevice as sd

from clasif_trad_utils import (
    DEFAULT_CHUNK_SEC,
    DEFAULT_OVERLAP_SEC,
    DEFAULT_SR,
    chunk_signal,
    extract_feature_vector,
    predict_positive_probability,
    prompt_user_to_select_inference_bundle,
)

warnings.filterwarnings("ignore", category=UserWarning)

RECORD_SECONDS = 8
MIN_HITS_RATIO = 0.25

# Si se deja en None, se usa el umbral calibrado en validacion.
THRESHOLD_OVERRIDE = None


def record_audio(seconds: float, sr: int) -> np.ndarray:
    print(f"\nGrabando sonido ambiente durante {seconds:.1f} segundos...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Grabacion finalizada.")
    return audio.squeeze()


def main() -> None:
    try:
        bundle = prompt_user_to_select_inference_bundle()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return
    except ValueError as exc:
        print(f"ERROR de metadatos del modelo: {exc}")
        return
    except (EOFError, KeyboardInterrupt):
        print("\nSeleccion cancelada por el usuario.")
        return

    metadata = bundle["metadata"]
    sr = int(metadata.get("sample_rate", DEFAULT_SR))
    chunk_sec = float(metadata.get("chunk_seconds", DEFAULT_CHUNK_SEC))
    overlap_sec = float(metadata.get("overlap_seconds", DEFAULT_OVERLAP_SEC))
    positive_label = metadata.get("positive_label", "siren")
    runtime_threshold = (
        float(THRESHOLD_OVERRIDE)
        if THRESHOLD_OVERRIDE is not None
        else float(bundle["runtime_threshold"])
    )

    print(
        f"Modelo cargado: {metadata.get('model_name', metadata.get('winner_name', 'desconocido'))} | "
        f"Clase positiva: {positive_label} | "
        f"Umbral por fragmento: {runtime_threshold:.2f} | "
        f"sr={sr} | chunk={chunk_sec:.3f}s | solape={overlap_sec:.3f}s"
    )

    y_full = record_audio(RECORD_SECONDS, sr)
    chunks, times = chunk_signal(
        y_full,
        sr=sr,
        chunk_length_s=chunk_sec,
        overlap_s=overlap_sec,
    )
    if not chunks:
        print("No se pudieron generar fragmentos validos. Revisa la grabacion.")
        return

    print(
        f"\nAnalizando {len(chunks)} fragmentos de {chunk_sec:.3f}s "
        f"con solape de {overlap_sec:.3f}s...\n"
    )

    probs: list[float] = []
    hits = 0
    for index, (chunk, start_time) in enumerate(zip(chunks, times), start=1):
        feature_vector = extract_feature_vector(chunk, sr=sr)
        x_scaled = bundle["scaler"].transform(feature_vector.reshape(1, -1))
        positive_prob = predict_positive_probability(bundle, x_scaled)
        probs.append(positive_prob)

        is_positive = positive_prob >= runtime_threshold
        hits += int(is_positive)
        label = positive_label.upper() if is_positive else "SIN ALERTA"
        print(
            f"[{index:02d}] segundo {start_time:5.2f}s -> {label} "
            f"(prob={positive_prob * 100:.1f}%)"
        )

    ratio = hits / len(probs)
    p_max = max(probs)
    p_mean = float(np.mean(probs))

    print("\n" + "=" * 46)
    print(" RESUMEN DE LA ESCUCHA AMBIENTAL ")
    print("=" * 46)
    print(
        f"Fragmentos positivos: {hits} de {len(probs)} "
        f"({ratio * 100:.1f}%)"
    )
    print(
        f"Probabilidad media: {p_mean * 100:.1f}% | "
        f"Pico maximo: {p_max * 100:.1f}%"
    )

    global_by_ratio = ratio >= MIN_HITS_RATIO
    global_by_peak = p_max >= runtime_threshold

    print("\nDiagnostico final:")
    if global_by_ratio or global_by_peak:
        print(f"ALERTA GLOBAL: posible evento '{positive_label}'.")
    else:
        print("Sin evidencia suficiente de una emergencia acustica.")
    print("=" * 46)


if __name__ == "__main__":
    main()
