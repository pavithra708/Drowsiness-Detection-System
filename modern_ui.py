import sys, cv2, time, pickle
import pandas as pd
import joblib
import mediapipe as mp
import winsound
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from utils import get_ear, get_mar, LEFT_EYE, RIGHT_EYE, MOUTH

class ElegantDrowsinessUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛏️ Drowsiness Detection System")
        self.setGeometry(200, 100, 1000, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #f0f0f0;
                font-family: Segoe UI, sans-serif;
            }
            QLabel {
                font-size: 16px;
            }
            QPushButton {
                background-color: #5c6bc0;
                color: white;
                padding: 10px 20px;
                font-size: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #7986cb;
            }
        """)

        # Load ML Model
        self.model = joblib.load('model.pkl')

        self.init_ui()

        # Webcam and model
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)

        self.blink_count = 0
        self.closed_frames = 0
        self.start_time = time.time()

    def init_ui(self):
        self.video_label = QLabel("Loading video...")
        self.video_label.setFixedSize(800, 500)
        self.video_label.setStyleSheet("border: 2px solid #444; border-radius: 12px;")

        self.status = QLabel("Status: ⏳")
        self.ear_label = QLabel("EAR: --")
        self.mar_label = QLabel("MAR: --")
        self.blink_label = QLabel("Blinks: --")
        self.rate_label = QLabel("Blink Rate: --")

        for label in [self.status, self.ear_label, self.mar_label, self.blink_label, self.rate_label]:
            label.setAlignment(Qt.AlignLeft)
            label.setFont(QFont("Segoe UI", 12))

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.status)
        info_layout.addWidget(self.ear_label)
        info_layout.addWidget(self.mar_label)
        info_layout.addWidget(self.blink_label)
        info_layout.addWidget(self.rate_label)

        frame = QFrame()
        frame.setLayout(info_layout)
        frame.setStyleSheet("background-color: #2e2e3e; border-radius: 15px; padding: 10px;")

        btn_start = QPushButton("Start Detection")
        btn_stop = QPushButton("Stop Detection")
        btn_start.clicked.connect(self.start_detection)
        btn_stop.clicked.connect(self.stop_detection)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_stop)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(frame)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.video_label.clear()
        self.status.setText("Status: ⏳")
        self.ear_label.setText("EAR: --")
        self.mar_label.setText("MAR: --")
        self.blink_label.setText("Blinks: --")
        self.rate_label.setText("Blink Rate: --")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        h, w, _ = frame.shape
        if result.multi_face_landmarks:
            mesh = result.multi_face_landmarks[0].landmark
            points = [(int(p.x * w), int(p.y * h)) for p in mesh]

            ear = get_ear(points, LEFT_EYE, RIGHT_EYE)
            mar = get_mar(points, MOUTH)

            if ear < 0.21:
                self.closed_frames += 1
            else:
                if self.closed_frames >= 3:
                    self.blink_count += 1
                self.closed_frames = 0

            elapsed = time.time() - self.start_time
            blink_rate = self.blink_count / elapsed if elapsed > 0 else 0

            df = pd.DataFrame([[ear, mar, blink_rate]], columns=["EAR", "MAR", "blink_rate"])
            prediction = self.model.predict(df)[0]

            self.status.setText(
                f"Status: {'🟢 Alert' if prediction == 0 else '🔴 Drowsy'}"
            )
            self.ear_label.setText(f"EAR: {ear:.2f}")
            self.mar_label.setText(f"MAR: {mar:.2f}")
            self.blink_label.setText(f"Blinks: {self.blink_count}")
            self.rate_label.setText(f"Blink Rate: {blink_rate:.2f}/s")

        # Display frame
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.stop_detection()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ElegantDrowsinessUI()
    win.show()
    sys.exit(app.exec_())
