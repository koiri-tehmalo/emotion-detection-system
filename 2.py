import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# โหลดโมเดลการทำนายอารมณ์
model = load_model('emotion_detection_model.h5')

# สร้างตัวแปรสำหรับ MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตรวจสอบการเชื่อมต่อกล้อง
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while True:
            # อ่านภาพจากกล้อง
            ret, frame = cap.read()

            # ตรวจสอบว่าอ่านภาพสำเร็จ
            if not ret:
                print("Error: Failed to capture image.")
                break

            # แปลงภาพเป็น RGB (เนื่องจาก MediaPipe ใช้ RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ตรวจจับใบหน้าในภาพ
            results = face_detection.process(frame_rgb)

            # ถ้ามีใบหน้าปรากฏ
            if results.detections:
                for detection in results.detections:
                    # วาดกรอบใบหน้า
                    mp_drawing.draw_detection(frame, detection)

                    # เอาขนาดของใบหน้าออกมาเพื่อใช้ในการแปลงภาพ
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # ตัดส่วนที่เป็นใบหน้าจากภาพ
                    face = frame[y:y+h, x:x+w]

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

                    # ถ้าทำนายว่า "สนใจ"
                    if predicted_class == 0:  # 0 = "สนใจ"
                        cv2.putText(frame, "Interested", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:  # 1 = "ไม่สนใจ"
                        cv2.putText(frame, "Not Interested", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # แสดงภาพในหน้าต่าง
            cv2.imshow("Emotion Detection with Face Detection", frame)

            # กด 'q' เพื่อออกจากโปรแกรม
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ปิดการเชื่อมต่อกับกล้องและหน้าต่าง
    cap.release()
    cv2.destroyAllWindows()
