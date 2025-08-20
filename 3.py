import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# โหลดโมเดลการทำนายอารมณ์
model = load_model('emotion_detection_model.h5')

# สร้างตัวแปรสำหรับ MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=20, min_detection_confidence=0.5)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตรวจสอบ FPS ของกล้อง
fps = cap.get(cv2.CAP_PROP_FPS)
frames_to_skip = int(fps * 10)  # กำหนดให้ตรวจจับทุก 30 วินาที
frame_count = 10

# ตรวจสอบการเชื่อมต่อกล้อง
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    while True:
        # อ่านภาพจากกล้อง
        ret, frame = cap.read()
        frame_count += 1  # เพิ่มจำนวนเฟรมที่ผ่านไป

        # ตรวจสอบว่าอ่านภาพสำเร็จ
        if not ret:
            print("Error: Failed to capture image.")
            break

        # ตรวจจับทุก 30 วินาที
        if frame_count % frames_to_skip == 0:
            # แปลงภาพเป็น RGB (เนื่องจาก MediaPipe ใช้ RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ตรวจจับใบหน้าในภาพ
            results = face_mesh.process(frame_rgb)

            # ถ้ามีใบหน้าปรากฏ
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # หาค่ากรอบใบหน้าคร่าว ๆ โดยใช้ landmark points
                    ih, iw, _ = frame.shape
                    x_min, y_min = iw, ih
                    x_max, y_max = 0, 0
                    
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        x_min, y_min = min(x, x_min), min(y, y_min)
                        x_max, y_max = max(x, x_max), max(y, y_max)
                    
                    # ขยายขอบเขตของกรอบใบหน้าเล็กน้อย
                    x_min, y_min = max(0, x_min - 10), max(0, y_min - 10)
                    x_max, y_max = min(iw, x_max + 10), min(ih, y_max + 10)
                    
                    # ตัดส่วนที่เป็นใบหน้าจากภาพ
                    face = frame[y_min:y_max, x_min:x_max]
                    
                    if face.size > 0:
                        # แปลงใบหน้าเป็น grayscale และรีไซส์ให้ตรงกับขนาดที่โมเดลต้องการ (48x48)
                        img_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        img_resized = cv2.resize(img_gray, (48, 48))
                        
                        # ขยายขนาดไปยัง batch size และ channel ที่เหมาะสม
                        img_array = np.expand_dims(img_resized, axis=-1)  # เพิ่มมิติที่หายไป (48, 48, 1)
                        img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติ batch size (1, 48, 48, 1)
                        
                        # ทำ normalization
                        img_array = img_array / 255.0
                        
                        # ทำนายอารมณ์
                        predictions = model.predict(img_array)
                        
                        # หาคลาสที่มีความน่าจะเป็นสูงสุด
                        predicted_class = np.argmax(predictions, axis=1)
                        
                        # กำหนดข้อความอารมณ์
                        emotion_text = "Interested" if predicted_class == 0 else "Not Interested"
                        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
                        
                        # แสดงผลบนหน้าจอ
                        cv2.putText(frame, emotion_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # วาดจุด Landmark บนใบหน้า
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # แสดงภาพในหน้าต่าง
        cv2.imshow("Emotion Detection with Face Mesh", frame)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดการเชื่อมต่อกับกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()