"""
Análisis de Marcha - Visión Computacional (Proyecto Final)
==========================================================
Este script procesa un video pregrabado de un ciclo de marcha,
extrae los landmarks cinemáticos usando MediaPipe y calcula
el ángulo de flexión de la rodilla en el plano sagital.

Controles:
- ESPACIO : Iniciar / Detener grabación de la demostración
- Q       : Salir de la ejecución
"""

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# 1. Inicialización del Modelo
# ---------------------------------------------------------------------------
# Se utiliza la versión 'lite' del pose landmarker para garantizar un 
# procesamiento fluido durante la demostración funcional.
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

# Validación rápida para evitar descargas redundantes si el modelo ya está local
if not os.path.exists(MODEL_PATH):
    print("Descargando pesos del modelo...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------------------------------------------------------
# 2. Configuración Visual (Esqueleto)
# ---------------------------------------------------------------------------
# Mapeo de las conexiones articulares de interés.
CONNECTIONS = [
    (11, 13), (13, 15),  # Brazo izquierdo
    (11, 12),            # Hombros
    (23, 24),            # Caderas
    (11, 23), (12, 24),  # Torso
    (23, 25), (25, 27),  # Pierna izquierda
    (24, 26), (26, 28),  # Pierna derecha
]
ARM_CONNECTIONS = [(12, 14), (14, 16)]
ARM_LANDMARKS   = [12, 14, 16]

def draw_skeleton(frame, landmarks):
    """
    Renderiza el modelo cinemático sobre el frame original.
    Se modificó el color por defecto a magenta para mejor contraste visual.
    """
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Conexiones principales en color Magenta (B=255, G=0, R=255)
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 0, 255), 2)
    for cx, cy in pts:
        cv2.circle(frame, (cx, cy), 4, (0, 100, 255), -1)

    # Conexiones secundarias (brazos)
    for a, b in ARM_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 140, 255), 4)
    for idx in ARM_LANDMARKS:
        cv2.circle(frame, pts[idx], 8, (0, 140, 255), -1)

# ---------------------------------------------------------------------------
# 3. Interfaz de Usuario (HUD)
# ---------------------------------------------------------------------------
def draw_hud(frame, recording: bool):
    """Muestra el estado de captura para el entregable de video procesado."""
    h, w = frame.shape[:2]

    if recording:
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 30, 24), 10, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 75, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ESPACIO: Grabar Demo | Q: Salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# 4. Cálculos Biomecánicos
# ---------------------------------------------------------------------------
def calcular_angulo(a, b, c):
    """
    Calcula el ángulo de flexión/extensión de la rodilla usando la diferencia 
    de las arcotangentes de los vectores Cadera-Rodilla y Tobillo-Rodilla.
    Esto evita los errores de cuadrante tradicionales.
    """
    ang_rad = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    ang_deg = abs(math.degrees(ang_rad))
    
    # Corrección para obtener el ángulo interior real de la articulación
    if ang_deg > 180.0:
        ang_deg = 360.0 - ang_deg
        
    return ang_deg

# ---------------------------------------------------------------------------
# 5. Flujo Principal de Procesamiento
# ---------------------------------------------------------------------------
def main():
    # Asignación del video de entrada para los 3 ciclos de marcha
    VIDEO_INPUT = "persona_caminando.mp4" 
    
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error crítico: No se encontró el archivo {VIDEO_INPUT}")
        return

    # Extracción de metadatos del video original para conservar la sincronía
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recording = False
    writer    = None

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Adecuación del espacio de color (OpenCV usa BGR, MediaPipe exige RGB)
rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Obtener el timestamp real en milisegundos
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Evitar error si el timestamp es 0 o repetido en el primer frame
            if timestamp_ms <= 0 and frame_idx > 0:
                timestamp_ms = int((frame_idx / FPS) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)

                # Identificación de landmarks para ambas extremidades inferiores
                cadera_izq, rodilla_izq, tobillo_izq = landmarks[23], landmarks[25], landmarks[27]
                cadera_der, rodilla_der, tobillo_der = landmarks[24], landmarks[26], landmarks[28]

                # LÓGICA DE OCLUSIÓN (Plano Sagital Bidireccional):
                # Se evalúa la coordenada Z (profundidad relativa a la cámara) para 
                # determinar automáticamente qué extremidad está expuesta a la lente.
                if cadera_izq.z < cadera_der.z:
                    cadera, rodilla, tobillo = cadera_izq, rodilla_izq, tobillo_izq
                    pierna_activa = "Izquierda (Regreso)"
                else:
                    cadera, rodilla, tobillo = cadera_der, rodilla_der, tobillo_der
                    pierna_activa = "Derecha (Ida)"

                # Procesamiento matemático del ángulo
                angulo = calcular_angulo(cadera, rodilla, tobillo)

                # Despliegue de datos cinemáticos en pantalla (Entregable visual)
                rx_px = int(rodilla.x * fw)
                ry_px = int(rodilla.y * fh)

                cv2.putText(frame, f"{int(angulo)} grados", (rx_px + 15, ry_px), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.putText(frame, f"Pierna analizada: {pierna_activa}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                print(f"[Frame {frame_idx}] {pierna_activa} | Angulo: {angulo:.1f} grados")

            draw_hud(frame, recording)

            if recording and writer:
                writer.write(frame)

            cv2.imshow("Analisis Cinemático - Proyecto Final", frame)

            # Gestión de eventos de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if not recording:
                    recording  = True
                    writer     = cv2.VideoWriter("demo_procesada.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (fw, fh))
                    print("⏺ Iniciando captura de la demostración...")
                else:
                    recording   = False
                    if writer:
                        writer.release()
                        writer = None
                    print("⏹ Captura finalizada exitosamente.")

    # Liberación segura de memoria y recursos de hardware
    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()