from scipy.spatial import distance as dist

# Eye and mouth landmark indexes from Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def get_ear(landmarks, left_eye_idx, right_eye_idx):
    # Compute EAR for both eyes
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    left_eye = [landmarks[i] for i in left_eye_idx]
    right_eye = [landmarks[i] for i in right_eye_idx]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    return (left_ear + right_ear) / 2.0

def get_mar(landmarks, mouth_idx):
    # Compute MAR (Mouth Aspect Ratio)
    top_lip = landmarks[mouth_idx[0]]
    bottom_lip = landmarks[mouth_idx[1]]
    left = landmarks[mouth_idx[2]]
    right = landmarks[mouth_idx[3]]

    vertical = dist.euclidean(top_lip, bottom_lip)
    horizontal = dist.euclidean(left, right)

    mar = vertical / horizontal
    return mar
