import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# --- 1. CONFIGURACION ---
# Definimos las clases que queremos detectar
CLASSES = ['sirena', 'ruido_trafico', 'musica']
# Numero de coeficientes MFCC a extraer (entre 13 y 40 es estandar)
N_MFCC = 13


# --- 2. EL MOTOR DE EXTRACCION (DSP) ---
def extract_features(file_path):
    try:
        # Cargar audio (librosa lo convierte a mono y hace resample a 22050Hz por defecto)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extraer MFCCs (Matriz: n_mfcc x tiempo)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

        # --- AQUI ESTA EL TRUCO PARA KNN ---
        # No podemos pasar la matriz temporal. Colapsamos el tiempo.
        # Calculamos la MEDIA de cada coeficiente en todo el clip
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # Opcional: Podrias anadir tambien la desviacion estandar para capturar la variacion
        # mfccs_std = np.std(mfccs.T, axis=0)
        # return np.hstack([mfccs_mean, mfccs_std]) # Vector mas largo

        return mfccs_mean  # Retornamos un vector 1D de tamano N_MFCC

    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None


# --- 3. CREACION DEL DATASET (Simulado para el ejemplo) ---
# NOTA: Para que esto funcione, necesitas una carpeta 'dataset' con subcarpetas:
# dataset/sirena/audio1.wav...
# dataset/ruido_trafico/audio2.wav...

def load_data(dataset_path):
    features = []
    labels = []

    for label in CLASSES:
        path = os.path.join(dataset_path, label)
        if not os.path.exists(path):
            print(f"Advertencia: No encuentro la carpeta {path}")
            continue

        print(f"Procesando clase: {label}...")
        for file in os.listdir(path):
            if file.endswith('.wav'):
                file_path = os.path.join(path, file)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(label)

    return np.array(features), np.array(labels)


# --- 4. FLUJO PRINCIPAL ---
# Descomenta las siguientes lineas cuando tengas tus carpetas con audios .wav

# A. Cargar datos
X, y = load_data('mi_carpeta_dataset')

# B. Dividir en Entrenamiento (80%) y Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C. Entrenar KNN
# k=3 suele funcionar bien para empezar (busca los 3 vecinos mas cercanos)
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
print("Modelo entrenado")

# D. Evaluar
y_pred = knn.predict(X_test)

# E. Resultados para los tutores
print("\n--- RESULTADOS DEL CLASIFICADOR KNN ---")
print(f"Precision Global: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nMatriz de Confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme Detallado:")
print(classification_report(y_test, y_pred, target_names=CLASSES))
