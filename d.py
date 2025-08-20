# show_landmarks_webcam.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ==== โหลดโมเดลจำแนกอารมณ์ (ของคุณจาก model.py) ====
model = load_model('emotion_detection_model.h5')

# ==== ตั้งค่า MediaPipe FaceMesh ====
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils          # ใช้ฟังก์ชันวาด
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# ==== เปิดกล้อง ====
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ---------- วาด 468 จุด ----------
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            
            # ---------- ตัดใบหน้าเล็ก ๆ 48×48 แล้วให้โมเดลทำนาย ----------
            ih, iw, _ = frame.shape
            # ดึงกรอบ face จาก landmark 0‑16‑… แบบง่าย ๆ
            xs = [int(lm.x * iw) for lm in face_landmarks.landmark]
            ys = [int(lm.y * ih) for lm in face_landmarks.landmark]
            x_min, x_max = max(min(xs),0), min(max(xs), iw)
            y_min, y_max = max(min(ys),0), min(max(ys), ih)
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size:
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48,48))
                face_input = face_resized.reshape(1,48,48,1) / 255.0
                pred = model.predict(face_input)
                label = "Interested" if np.argmax(pred)==1 else "Not Interested"
                cv2.putText(frame, label, (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0,255,0) if label=="Interested" else (0,0,255), 2)

    cv2.imshow("Landmarks + FER", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
