import cv2
import numpy as np
import os
import Monfodb

def comparar_rostros(carpeta_imagenes, umbral=0.7):
    # Obtener la ruta completa de la carpeta de imágenes
    ruta_carpeta_imagenes = os.path.join(os.path.dirname(__file__), carpeta_imagenes)

    # Cargar las imágenes de referencia y convertirlas a escala de grises
    imagenes_referencia = []
    for archivo in os.listdir(ruta_carpeta_imagenes):
        ruta_imagen = os.path.join(ruta_carpeta_imagenes, archivo)
        img_referencia = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        imagenes_referencia.append(img_referencia)

    # Inicializar el detector de rostros
    detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializar la cámara
    camara = cv2.VideoCapture(0)

    while True:
        # Capturar un fotograma de la cámara
        _, fotograma = camara.read()

        # Convertir el fotograma a escala de grises
        fotograma_gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el fotograma
        rostros = detector_rostros.detectMultiScale(fotograma_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rostros:
            # Recortar el área del rostro en el fotograma
            roi = fotograma_gris[y:y + h, x:x + w]

            # Calcular la similitud entre el rostro recortado y las imágenes de referencia
            similitudes = []
            for img_referencia in imagenes_referencia:
                # Redimensionar la imagen de referencia para que coincida con el tamaño de roi
                img_referencia_resized = cv2.resize(img_referencia, (roi.shape[1], roi.shape[0]))

                diferencia = cv2.absdiff(roi, img_referencia_resized)
                similitud = np.mean(diferencia)
                similitudes.append(similitud)

            # Obtener el índice de la imagen de referencia con la menor similitud
            indice_min_similitud = np.argmin(similitudes)

            # Comparar la similitud con el umbral
            if similitudes[indice_min_similitud] < umbral:
                nombre_imagen = os.listdir(ruta_carpeta_imagenes)[indice_min_similitud]
                print("¡Rostros coincidentes con", nombre_imagen, "!")
                return True

        # Mostrar el fotograma en una ventana
        cv2.imshow('Comparación de rostros', fotograma)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    camara.release()
    cv2.destroyAllWindows()

    # Si no se encontraron rostros coincidentes, devolver False
    return False

# Ruta de la carpeta de imágenes de referencia (debe estar en el mismo directorio que el archivo de código)
carpeta_imagenes = './carpeta_de_imagenes'

# Llamar a la función para comparar los rostros
resultado = comparar_rostros(carpeta_imagenes)

# Imprimir el resultado
print(resultado)
