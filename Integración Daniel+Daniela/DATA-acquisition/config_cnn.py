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
    'traditional_pattern': '**/clasificador_tradicional_*_bundle.joblib',

    'api_url': 'http://192.168.1.138:8000',

    # None usa el umbral recomendado de cada artefacto. Define un float solo
    # para pruebas manuales controladas sobre todas las CNN.
    'threshold': None,

    # Overrides puntuales de CNN por nombre de archivo o por stem del modelo.
    # Ejemplo: {'modelo_x.keras': 0.75, 'modelo_y': 0.60}
    'threshold_overrides': {}
}
