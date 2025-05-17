from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time
import threading

app = Flask(__name__)
lock = threading.Lock()

# โหลดโมเดลการทำนายอารมณ์
model = load_model('emotion_detection_model.h5')

# MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=20, min_detection_confidence=0.5)


# ค่าทางสถิติ
history = deque(maxlen=1200)  # เก็บข้อมูล 20 นาที (ถ้าตรวจจับทุกวินาที)

cap = cv2.VideoCapture(0)

def detect_emotion():
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            emotion_label = "ไม่พบใบหน้า"
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face = frame[y:y+h, x:x+w]
                    
                    if face.size > 0:
                        img_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        img_resized = cv2.resize(img_gray, (48, 48))
                        img_array = np.expand_dims(img_resized, axis=-1)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0
                        
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions, axis=1)[0]
                        
                        emotion_label = "Interested" if predicted_class == 0 else "Not Interested"
                        history.append(emotion_label)
                    
                    color = (0, 255, 0) if emotion_label == "Interested" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('2.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    interested = history.count("Interested")
    not_interested = history.count("Not Interested")
    total = interested + not_interested if (interested + not_interested) > 0 else 1
    return jsonify({
        "Interested": round((interested / total) * 100, 2),
        "Not Interested": round((not_interested / total) * 100, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)
