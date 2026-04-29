# ==========================================
# CONFIGURACION DEL DETECTOR DE SIRENAS
# ==========================================

MODEL = {
    # Carpetas donde se buscan automaticamente los modelos y sus JSON asociados.
    # Si es relativa, se resuelve desde DATA-acquisition.
    'cnn_directory': '../Modelos/CNN',
    'traditional_directory': '../Modelos/Tradicionales',

    # Patrones de busqueda dentro de cada carpeta. Permiten subcarpetas.
    'cnn_pattern': '**/*.keras',
    'traditional_pattern': '**/*clasificador_tradicional_*_bundle.joblib',

    'api_url': 'http://192.168.1.138:8000',

    # ==========================================================
    # AJUSTES MANUALES DE INFERENCIA
    # ==========================================================
    # Umbrales manuales individuales para CNN y clasificadores tradicionales.
    # Si un modelo no aparece aqui, usa el umbral guardado en su JSON/bundle.
    #
    # Claves aceptadas:
    #   - Nombre de archivo: 'modelo.keras' o 'clasificador_..._bundle.joblib'
    #   - Stem del archivo: 'modelo' o 'clasificador_..._bundle'
    #   - Ruta relativa dentro de Modelos/CNN o Modelos/Tradicionales
    #   - Nombre mostrado en consola: 'CNN:...' o 'TRAD:...'
    #
    # Ejemplo:
    # 'threshold_overrides': {
    #     'mi_cnn.keras': 0.82,
    #     'exp_012/clasificador_tradicional_random_forest_bundle': 0.65,
    # }
    'threshold_overrides': {},

    # Segundos consecutivos que cada modelo debe clasificar como sirena antes
    # de aportar un voto positivo. Con 1.0s, un modelo de chunks de 0.5s sin
    # solape necesita 2 positivos seguidos, y uno de chunks de 1.0s necesita 1.
    'required_positive_seconds': 1.0,

    # Overrides individuales del requisito temporal anterior. Usa las mismas
    # claves aceptadas en threshold_overrides.
    #
    # Ejemplo:
    # 'required_positive_seconds_overrides': {
    #     'mi_cnn.keras': 1.5,
    # }
    'required_positive_seconds_overrides': {}
}
