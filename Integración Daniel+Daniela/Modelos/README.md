# Modelos de deteccion

`DATA-acquisition/piloto_micro_api.py` carga automaticamente todos los modelos
validos que encuentre en estas subcarpetas:

```text
Modelos/
  CNN/
  Tradicionales/
```

## CNN

Coloca en `CNN/` los modelos `.keras`. Cada modelo debe tener al lado su JSON
de postprocesado generado por `entrenar_modelo_margin_3.py`:

```text
modelo_a.keras
modelo_a_postprocesado.json
modelo_b.keras
modelo_b_postprocesado.json
```

## Clasificadores tradicionales

Coloca en `Tradicionales/` solo bundles de inferencia generados por
`entrenar_modelo_clasif_trad.py`, junto con el JSON de postprocesado del mismo
entrenamiento:

```text
exp_001/
  clasificador_tradicional_random_forest_bundle.joblib
  clasificador_tradicional_svm_bundle.joblib
  clasificador_tradicional_knn_bundle.joblib
  exp_001_..._postprocesado.json
```

Tambien se aceptan bundles con prefijo de experimento, por ejemplo
`exp_012_clasificador_tradicional_random_forest_bundle.joblib`, siempre que el
JSON de postprocesado tenga el mismo prefijo `exp_012`.

El bundle contiene el modelo, el scaler, el label encoder y el umbral especifico
del clasificador. El JSON aporta la configuracion temporal del entrenamiento.

El programa falla al arrancar si no encuentra modelos, si falta algun JSON o si
la forma de entrada del modelo no coincide con su configuracion.
