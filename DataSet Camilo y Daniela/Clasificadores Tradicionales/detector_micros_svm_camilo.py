import warnings

import sounddevice as sd

from clasif_trad_utils import (
    DEFAULT_CHUNK_SEC,
    DEFAULT_SR,
    extract_feature_vector,
    predict_positive_probability,
    prompt_user_to_select_inference_bundle,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Si se deja en None, se usa el valor calibrado en validacion.
THRESHOLD_OVERRIDE = None


def main() -> None:
    print("Seleccion del clasificador tradicional para inferencia.")
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
    chunk_samples = int(round(chunk_sec * sr))
    positive_label = metadata.get("positive_label", "siren")
    runtime_threshold = (
        float(THRESHOLD_OVERRIDE)
        if THRESHOLD_OVERRIDE is not None
        else float(bundle["runtime_threshold"])
    )

    print(
        "Sistema listo. "
        f"Modelo cargado: {metadata.get('model_name', metadata.get('winner_name', 'desconocido'))} | "
        f"Clase positiva: {positive_label} | "
        f"Umbral: {runtime_threshold:.2f} | "
        f"sr={sr} | chunk={chunk_sec:.3f}s"
    )
    print("\nEscuchando audio en vivo. Pulsa Ctrl+C para detener.\n")

    try:
        while True:
            audio = sd.rec(
                chunk_samples,
                samplerate=sr,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            y_chunk = audio.flatten()

            features = extract_feature_vector(y_chunk, sr=sr)
            x_scaled = bundle["scaler"].transform(features.reshape(1, -1))
            positive_prob = predict_positive_probability(bundle, x_scaled)

            if positive_prob >= runtime_threshold:
                print(
                    f"ALERTA {positive_label.upper()} | "
                    f"probabilidad={positive_prob * 100:.1f}%"
                )
            else:
                print(
                    f"Sin alerta | "
                    f"prob_{positive_label}={positive_prob * 100:.1f}%"
                )
    except KeyboardInterrupt:
        print("\nSistema detenido por el usuario.")


if __name__ == "__main__":
    main()
