import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

KEYPOINT_NAMES = {
    0: "nariz", 1: "ojo_izq", 2: "ojo_der", 3: "oreja_izq", 4: "oreja_der",
    5: "hombro_izq", 6: "hombro_der", 7: "codo_izq", 8: "codo_der",
    9: "muñeca_izq", 10: "muñeca_der", 11: "cadera_izq", 12: "cadera_der",
    13: "rodilla_izq", 14: "rodilla_der", 15: "tobillo_izq", 16: "tobillo_der"
}

KEYPOINT_EDGES = [
    # Cabeza y cuello (azul)
    (0, 1, (255, 0, 0)), (0, 2, (255, 0, 0)), (1, 3, (255, 0, 0)), (2, 4, (255, 0, 0)),
    # Tronco (verde)
    (5, 6, (0, 255, 0)), (5, 11, (0, 255, 0)), (6, 12, (0, 255, 0)), (11, 12, (0, 255, 0)),
    # Brazo izquierdo (rojo)
    (5, 7, (0, 0, 255)), (7, 9, (0, 0, 255)),
    # Brazo derecho (rojo)
    (6, 8, (0, 0, 255)), (8, 10, (0, 0, 255)),
    # Pierna izquierda (amarillo)
    (11, 13, (0, 255, 255)), (13, 15, (0, 255, 255)),
    # Pierna derecha (amarillo)
    (12, 14, (0, 255, 255)), (14, 16, (0, 255, 255))
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def draw_keypoints(frame, keypoints, confidence_threshold=0.4):
    y, x, c = frame.shape
    shaped = np.squeeze(keypoints)
    kp = shaped.reshape(-1, 3)
    
    for edge in KEYPOINT_EDGES:
        p1, p2, color = edge
        y1, x1, c1 = kp[p1]
        y2, x2, c2 = kp[p2]
        
        if c1 > confidence_threshold and c2 > confidence_threshold:
            x1 = int(x1 * x)
            y1 = int(y1 * y)
            x2 = int(x2 * x)
            y2 = int(y2 * y)
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    
    for idx, kp_data in enumerate(kp):
        ky, kx, kp_conf = kp_data
        if kp_conf > confidence_threshold:
            x_coord = int(kx * x)
            y_coord = int(ky * y)
            
            cv2.circle(frame, (x_coord, y_coord), 6, (255, 255, 255), -1)
            
            label = f"{KEYPOINT_NAMES[idx]}: {kp_conf:.2f}"
            cv2.putText(frame, label, (x_coord + 10, y_coord), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    avg_conf = np.mean([kp[2] for kp in shaped if kp[2] > confidence_threshold])
    cv2.putText(frame, f"Confianza: {avg_conf:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)

    results = movenet(input=img)
    keypoints = results['output_0']

    draw_keypoints(frame, keypoints, confidence_threshold=0.4)

    cv2.putText(frame, 'MoveNet Thunder', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MoveNet Thunder Pose Detection', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 