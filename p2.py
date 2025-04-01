import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(keypoints)
    kp = shaped.reshape(-1, 3)
    
    for edge in KEYPOINT_EDGES:
        p1, p2 = edge
        y1, x1, c1 = kp[p1]
        y2, x2, c2 = kp[p2]
        
        if c1 > confidence_threshold and c2 > confidence_threshold:
            x1 = int(x1 * x)
            y1 = int(y1 * y)
            x2 = int(x2 * x)
            y2 = int(y2 * y)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    for kp in kp:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            x_coord = int(kx * x)
            y_coord = int(ky * y)
            cv2.circle(frame, (x_coord, y_coord), 4, (0, 255, 0), -1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)

    results = movenet(input=img)
    keypoints = results['output_0']

    draw_keypoints(frame, keypoints, 0.3)

    cv2.imshow('MoveNet Pose Detection', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
