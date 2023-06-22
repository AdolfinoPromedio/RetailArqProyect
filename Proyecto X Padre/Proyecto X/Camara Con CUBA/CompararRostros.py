import cv2
import numpy as np
import os

def comparar_rostros(foto_carpeta, foto_camara):
    # Cargar la imagen de referencia
    parapeta_img = cv2.imread(foto_carpeta)

    # Convertir la imagen de referencia a escala de grises
    parapeta_gris = cv2.cvtColor(parapeta_img, cv2.COLOR_BGR2GRAY)

    # Convertir la foto de la cámara a escala de grises
    camara_gris = cv2.cvtColor(foto_camara, cv2.COLOR_BGR2GRAY)

    # Crear el detector de rostros
    detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detectar rostros en la imagen de referencia y en la foto de la cámara
    rostros_parapeta = detector_rostros.detectMultiScale(parapeta_gris, scaleFactor=1.1, minNeighbors=5)
    rostros_camara = detector_rostros.detectMultiScale(camara_gris, scaleFactor=1.1, minNeighbors=5)

    # Verificar si se detectó al menos un rostro en la imagen de referencia y en la foto de la cámara
    if len(rostros_parapeta) == 0 or len(rostros_camara) == 0:
        print("No se encontraron rostros en la imagen de referencia o en la foto de la cámara.")
        return False

    # Tomar solo el primer rostro detectado en la imagen de referencia y en la foto de la cámara
    (x_parapeta, y_parapeta, w_parapeta, h_parapeta) = rostros_parapeta[0]
    (x_camara, y_camara, w_camara, h_camara) = rostros_camara[0]

    # Extraer las regiones de interés (ROIs) correspondientes a los rostros
    roi_parapeta = parapeta_gris[y_parapeta:y_parapeta + h_parapeta, x_parapeta:x_parapeta + w_parapeta]
    roi_camara = camara_gris[y_camara:y_camara + h_camara, x_camara:x_camara + w_camara]

    # Redimensionar las ROIs para que tengan el mismo tamaño
    roi_parapeta = cv2.resize(roi_parapeta, (100, 100))
    roi_camara = cv2.resize(roi_camara, (100, 100))

    # Calcular la diferencia absoluta entre las ROIs
    diferencia = cv2.absdiff(roi_parapeta, roi_camara)

    # Calcular el promedio de la diferencia absoluta
    promedio_diferencia = np.mean(diferencia)

    # Definir un umbral de diferencia
    umbral = 30

    # Verificar si la diferencia está por debajo del umbral
    if promedio_diferencia < umbral:
        return True
    else:
        return False


def comparar_con_carpeta(carpeta_path, foto_camara):
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(carpeta_path)

    # Iterar sobre los archivos de la carpeta
    for archivo in archivos:
        # Construir la ruta completa del archivo
        archivo_path = os.path.join(carpeta_path, archivo)

        # Leer el archivo de la carpeta
        frame = cv2.imread(archivo_path)

        # Llamar a la función de comparación de rostros
        if comparar_rostros(archivo_path, foto_camara):
            print()
            return True

    return False


