import cv2
import mediapipe as mp
import math
import uuid
from Monfodb import *
from CompararRostros import *
import os
import torch

def calcular_distancia(p1, p2):
    return torch.sqrt(torch.sum((p1 - p2) ** 2))

def detectar_cara_y_manos_en_video():
    # Inicializar el detector de manos y el detector facial
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection

    # Inicializar la captura de video
    cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada, puedes cambiarlo si tienes varias cámaras

    # Configurar el dibujo de los puntos de las manos
    hand_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

    # Configurar el dibujo del cuadro del rostro
    face_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    # Ruta de la carpeta de imágenes
    carpeta_imagenes = os.path.join(os.path.dirname(__file__), 'carpeta_de_imagenes')
    print(carpeta_imagenes)

    # Crear la carpeta de imágenes si no existe
    if not os.path.exists(carpeta_imagenes):
        os.makedirs(carpeta_imagenes)

    # Configuración de la detección de colisión
    threshold_distance = 0.1  # Ajusta este valor según tus necesidades

    # Verificar si hay una GPU disponible y configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar los detectores
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                # Leer el cuadro del video
                ret, frame = cap.read()
                if not ret:
                    break

                # Convertir la imagen a RGB y moverla a la GPU
                image = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device=device)

                # Detectar las manos
                results_hands = hands.process(image)

                # Detectar los rostros
                results_faces = face_detection.process(image)

                # Dibujar los puntos de las manos
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            hand_drawing_spec,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        )

                        # Obtener la posición de los puntos de referencia de la mano
                        landmarks = hand_landmarks.landmark
                        thumb_tip = torch.tensor([landmarks[mp_hands.HandLandmark.THUMB_TIP].x,
                                                  landmarks[mp_hands.HandLandmark.THUMB_TIP].y],
                                                 device=device)
                        index_tip = torch.tensor([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                                  landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y],
                                                 device=device)
                        middle_tip = torch.tensor([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                                  landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y],
                                                 device=device)
                        ring_tip = torch.tensor([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                                  landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y],
                                                 device=device)
                        pinky_tip = torch.tensor([landmarks[mp_hands.HandLandmark.PINKY_TIP].x,
                                                  landmarks[mp_hands.HandLandmark.PINKY_TIP].y],
                                                 device=device)

                        # Calcular la distancia entre los puntos de referencia de los dedos
                        distance_thumb_index = calcular_distancia(thumb_tip, index_tip)
                        distance_thumb_middle = calcular_distancia(thumb_tip, middle_tip)
                        distance_thumb_ring = calcular_distancia(thumb_tip, ring_tip)
                        distance_thumb_pinky = calcular_distancia(thumb_tip, pinky_tip)

                        # Verificar la colisión de las manos
                        if (distance_thumb_index < threshold_distance or
                            distance_thumb_middle < threshold_distance or
                            distance_thumb_ring < threshold_distance or
                            distance_thumb_pinky < threshold_distance):
                            print("¡Colisión detectada entre las manos!")

                            # Generar un nombre aleatorio para la imagen
                            nombre_imagen = str(uuid.uuid4()) + ".jpg"
                            path_imagen = os.path.join(carpeta_imagenes, nombre_imagen)

                            # Guardar una captura del rostro
                            face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Condición si la imagen se encuentra o no en la base de datos
                            if comparar_con_carpeta(carpeta_imagenes, face_image):
                                print("El rostro ya se encuentra en la base de datos")
                                print("Mandando alerta...")
                            else:
                                # Guardar la imagen
                                print("Guardando la imagen...")
                                cv2.imwrite(path_imagen, face_image)
                                print("Captura del rostro guardada como", path_imagen)
                                # Guardar una Captura del Rostor en MongoDb
                                #mongo_save_image(face_image)

                # Dibujar los cuadros del rostro
                if results_faces.detections:
                    for detection in results_faces.detections:
                        mp_drawing.draw_detection(frame, detection, face_drawing_spec)

                # Mostrar el resultado
                cv2.imshow('Hand and Face Detection', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para detectar cara y manos en tiempo real
detectar_cara_y_manos_en_video()
