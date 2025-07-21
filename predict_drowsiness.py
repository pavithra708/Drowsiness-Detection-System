import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pandas as pd
import winsound  # Windows only
from utils import get_ear, get_mar, LEFT_EYE, RIGHT_EYE, MOUTH

# Load trained model
with open('model.pkl', 'rb') as f:
      model = joblib.load('model.pkl')

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

blink_count = 0
closed_frames = 0
start_time = time.time()

cap = cv2.VideoCapture(0)

print("Press 'q' to quit the program.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame. Exiting.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points]

        # Calculate EAR and MAR
        ear = get_ear(landmarks, LEFT_EYE, RIGHT_EYE)
        mar = get_mar(landmarks, MOUTH)

        # Blink detection logic
        EAR_THRESHOLD = 0.21
        CONSEC_FRAMES = 3
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= CONSEC_FRAMES:
                blink_count += 1
            closed_frames = 0

        # Calculate blink rate
        elapsed_time = time.time() - start_time
        blink_rate = blink_count / elapsed_time if elapsed_time > 0 else 0

        # Create input for ML model
        features = pd.DataFrame([[ear, mar, blink_rate]], columns=['EAR', 'MAR', 'blink_rate'])
        prediction = model.predict(features)[0]

        # Display status
        if prediction == 1:
            status = "Drowsy! Please take a break."
            color = (0, 0, 255)  # Red
            winsound.Beep(1000, 500)
        else:
            status = "Alert"
            color = (0, 255, 0)  # Green

        # Overlay info
        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Blink Rate: {blink_rate:.2f}/s", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Ensures windows close properly
