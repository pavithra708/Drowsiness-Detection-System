import cv2
import mediapipe as mp
import math
import pandas as pd

# ✅ Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# ✅ Helper: Euclidean distance between two points
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ✅ Compute EAR (Eye Aspect Ratio)
def calculate_EAR(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    A = euclidean(p[1], p[5])
    B = euclidean(p[2], p[4])
    C = euclidean(p[0], p[3])
    return (A + B) / (2.0 * C)

# ✅ Compute MAR (Mouth Aspect Ratio)
def calculate_MAR(landmarks, mouth_indices):
    p = [landmarks[i] for i in mouth_indices]
    # Using chosen points inside p list (we have 10 points, so index from p[0]..p[9])
    A = euclidean(p[2], p[9])  # vertical
    B = euclidean(p[4], p[7])  # vertical
    C = euclidean(p[0], p[5])  # vertical
    D = euclidean(p[3], p[8])  # horizontal
    return (A + B + C) / (2.0 * D)

# ✅ Landmark indices from Mediapipe's 468 points
left_eye_indices  = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
mouth_indices     = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

# ✅ Start video capture
cap = cv2.VideoCapture(0)

data = []  # List to store collected data rows

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Compute EAR
            left_EAR = calculate_EAR(landmarks, left_eye_indices)
            right_EAR = calculate_EAR(landmarks, right_eye_indices)
            EAR = (left_EAR + right_EAR) / 2

            # Compute MAR
            MAR = calculate_MAR(landmarks, mouth_indices)

            # Display EAR and MAR on the video frame
            cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show video window
    cv2.imshow("Data Collection - Press a(alert) / d(drowsy) / q(quit)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        print("✅ Recorded ALERT")
        data.append([EAR, MAR, 0, 'alert'])  # blink_rate=0 for now
    elif key == ord('d'):
        print("⚠️  Recorded DROWSY")
        data.append([EAR, MAR, 0, 'drowsy'])
    elif key == ord('q'):
        print("👋 Quitting data collection.")
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()

# ✅ Save collected data to CSV
df = pd.DataFrame(data, columns=['EAR', 'MAR', 'blink_rate', 'label'])
df.to_csv('dataset.csv', index=False)
print("📦 Data saved to dataset.csv ✅")
