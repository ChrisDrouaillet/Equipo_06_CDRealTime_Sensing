# RealTime Sensing

Repositorio del proyecto final.
# Análisis de Marcha - Visión Computacional 🚶‍♂️⚙️

**Materia:** Programación de Sistemas Embebidos (2026A)  
**Equipo:** 06  

### Integrantes:
* Christian Drouaillet 
* Sayid Alatorre 
* Ivan Cortez 

---

## 🎯 Objetivo del Proyecto
Desarrollar un sistema de análisis de marcha integrando captura cinemática mediante **MediaPipe** y **OpenCV** para caracterizar el patrón de movimiento durante el ciclo de la marcha en el plano sagital.

## 🛠️ Evolución del Código y Modificaciones Principales
El desarrollo de este entregable toma como base estructural el script `EMG_video_4.py` proporcionado en el repositorio original. Para cumplir con los requerimientos cinemáticos de la rúbrica, el código fue refactorizado (ahora guardado como `skeleton_recorder.py`) con los siguientes cambios críticos:

1. **Transición a Procesamiento de Video Pregrabado:** Se modificó la entrada de cámara en vivo (`cv2.VideoCapture(0)`) para procesar el archivo local `persona_caminando.mp4`, asegurando el análisis constante de los 3 ciclos de marcha.
2. **Cálculo Cinemático de Ángulos:** Se implementó una función trigonométrica (basada en arcotangentes) para extraer las coordenadas de cadera, rodilla y tobillo, calculando dinámicamente el ángulo de flexión/extensión de la rodilla y evitando errores de cuadrante.
3. **Manejo de Oclusión Bidireccional (Eje Z):** Se integró una lógica de evaluación de profundidad para determinar automáticamente qué extremidad está expuesta al lente (caminata de ida vs. regreso), previniendo la toma de datos basura.
4. **Filtros de Precisión (Anti-falsos positivos):** Se elevaron los umbrales de confianza del modelo `pose_landmarker_lite` al 75% (`min_pose_detection_confidence=0.75`) para evitar que el algoritmo confundiera objetos del entorno con siluetas humanas durante la captura.
5. **Aislamiento de la Visión Computacional:** Por cuestiones de tiempo y alcance de la demostración, se retiraron los hilos de ejecución (`threading`) y las funciones correspondientes a la señal EMG serial, enfocando los recursos de hardware exclusivamente en el procesamiento visual fluido en tiempo real.

## 📁 Archivos de la Demostración (Carpeta `example/scripts/`)
* `skeleton_recorder.py` -> Código fuente final refactorizado.
* `persona_caminando.mp4` -> Video original para pruebas.
* `demo_procesada.mp4` -> Entregable en video mostrando los ciclos de marcha con el análisis superpuesto.
## Contenido

### Interface — Código de ejemplo

La carpeta `Interface/` contiene un ejemplo funcional para conectar Python con tu tarjeta de desarrollo (ESP32). Incluye el firmware en C para leer el ADC y enviarlo por puerto serial, y un script en Python que recibe los datos y los grafica en tiempo real.

Consulta [`Interface/README.md`](Interface/README.md) para instrucciones de instalación, configuración del puerto serial y cómo correr el ejemplo.

