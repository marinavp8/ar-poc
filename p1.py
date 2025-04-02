import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Definir los nombres de los puntos clave (33 puntos en MediaPipe)
KEYPOINT_NAMES = {
    0: "nariz", 1: "ojo_izq", 2: "ojo_der", 3: "oreja_izq", 4: "oreja_der",
    5: "hombro_izq", 6: "hombro_der", 7: "codo_izq", 8: "codo_der",
    9: "muñeca_izq", 10: "muñeca_der", 11: "cadera_izq", 12: "cadera_der",
    13: "rodilla_izq", 14: "rodilla_der", 15: "tobillo_izq", 16: "tobillo_der",
    17: "talón_izq", 18: "talón_der", 19: "dedo_pie_izq", 20: "dedo_pie_der",
    21: "pulgar_izq", 22: "índice_izq", 23: "medio_izq", 24: "anular_izq",
    25: "meñique_izq", 26: "pulgar_der", 27: "índice_der", 28: "medio_der",
    29: "anular_der", 30: "meñique_der", 31: "cara_izq", 32: "cara_der"
}

# Colores para diferentes partes (en BGR)
COLORS = {
    'cara': (255, 0, 0),    # Azul
    'brazos': (0, 0, 255),  # Rojo
    'tronco': (0, 255, 0),  # Verde
    'piernas': (0, 255, 255), # Amarillo
    'pies': (255, 0, 255),  # Magenta
    'manos': (128, 128, 0)  # Turquesa
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def draw_pose_with_colors(frame, results, confidence_threshold=0.5):
    if not results.pose_landmarks:
        return

    h, w, c = frame.shape
    landmarks = results.pose_landmarks.landmark

    # Dibujar conexiones con colores personalizados
    # Cara
    draw_connection(frame, landmarks, [0,1,2,3,4], COLORS['cara'], confidence_threshold)
    # Brazos
    draw_connection(frame, landmarks, [11,13,15,17,19,21], COLORS['brazos'], confidence_threshold) # Izquierdo
    draw_connection(frame, landmarks, [12,14,16,18,20,22], COLORS['brazos'], confidence_threshold) # Derecho
    # Tronco
    draw_connection(frame, landmarks, [11,12,23,24], COLORS['tronco'], confidence_threshold)
    # Piernas
    draw_connection(frame, landmarks, [23,25,27,29,31], COLORS['piernas'], confidence_threshold) # Izquierda
    draw_connection(frame, landmarks, [24,26,28,30,32], COLORS['piernas'], confidence_threshold) # Derecha

    # Dibujar puntos y etiquetas
    confidences = []
    for idx, landmark in enumerate(landmarks):
        if idx in KEYPOINT_NAMES:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            conf = landmark.visibility
            
            if conf > confidence_threshold:
                confidences.append(conf)
                # Dibujar círculo
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                # Mostrar etiqueta
                label = f"{KEYPOINT_NAMES[idx]}: {conf:.2f}"
                cv2.putText(frame, label, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar confianza promedio
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        cv2.putText(frame, f"Confianza: {avg_conf:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_connection(frame, landmarks, indices, color, threshold):
    h, w, c = frame.shape
    points = []
    for idx in indices:
        landmark = landmarks[idx]
        if landmark.visibility > threshold:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
    
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 3)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    draw_pose_with_colors(frame, results)

    # Mostrar título
    cv2.putText(frame, 'MediaPipe Pose', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Pose Detection', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
