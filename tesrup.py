import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('emotion_detection_model.h5')  # โหลดโมเดลการทำนายอารมณ์
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

def preprocess_image(image):
    """ฟังก์ชันที่ใช้ในการปรับขนาดภาพและแปลงให้เหมาะสมกับโมเดล"""
    # ตรวจสอบขนาดของภาพและปรับขนาดให้เป็นขนาดที่เหมาะสม
    if image is None:
        return None
    img_resized = cv2.resize(image, (48, 48))  # ปรับขนาดภาพเป็น 48x48
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพขาวดำ
    img_array = np.expand_dims(img_gray, axis=-1)  # เพิ่มมิติช่องสี
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # ย่อค่าพิกเซลให้เป็น [0, 1]
    return img_array

def detect_emotion_from_image(image_path):
    # โหลดภาพจากไฟล์
    frame = cv2.imread(image_path)
    
    # ตรวจจับใบหน้าด้วย MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    # ตรวจจับและแสดงผล
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = frame[y:y+h, x:x+w]
            
            if face.size > 0:
                # ใช้ฟังก์ชัน preprocess_image ในการปรับขนาดและแปลงภาพ
                img_array = preprocess_image(face)
                
                if img_array is not None:
                    # ทำนายอารมณ์
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    
                    # แสดงผลการทำนาย
                    emotion_label = "Interested" if predicted_class == 0 else "Not Interested"
                    
                    # วาดกรอบรอบใบหน้าและแสดงอารมณ์
                    color = (0, 255, 0) if emotion_label == "Interested" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # แสดงภาพที่แสดงผล
    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
