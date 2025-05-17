from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from datetime import datetime
import os
import pandas as pd
from io import BytesIO
app = Flask(__name__)

# โหลดโมเดลการทำนายอารมณ์
model = load_model('emotion_detection_model.h5')

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# ค่าทางสถิติ
history = deque(maxlen=1200)  # เก็บข้อมูล 20 นาที (ถ้าตรวจจับทุกวินาที)
time_history = deque(maxlen=1200)  # เก็บเวลาของการตรวจจับ
face_count_history = deque(maxlen=1200)

cap = cv2.VideoCapture(0)

def detect_emotion():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        num_faces = 0
        interested_count = 0
        not_interested_count = 0
        
        if results.detections:
            num_faces = len(results.detections)
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
                    time_history.append(datetime.now().strftime("%H:%M:%S"))
                    
                    if predicted_class == 0:
                        interested_count += 1
                    else:
                        not_interested_count += 1
                
                color = (0, 255, 0) if emotion_label == "Interested" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        face_count_history.append((datetime.now().strftime("%H:%M:%S"), num_faces, interested_count, not_interested_count))
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang='th'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>ระบบตรวจจับอารมณ์</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
            .container { display: flex; justify-content: center; align-items: center; margin-top: 20px; }
            .video-box { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .stats-box { margin-left: 20px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: left; }
            #emotion-history {        max-height: 400px;  /* จำกัดความสูง */ overflow-y: auto;}
            ul { list-style-type: none; padding: 0; }
            li { padding: 5px; border-bottom: 1px solid #ddd; }
            .btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 5px; }
            .btn:hover { background-color: #45a049; }
        </style>
        
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            function updateHistory() {
                $.getJSON("/history", function(data) {
                    const historyList = $("#emotion-history");
                    historyList.empty();
                    data.reverse().forEach(item => {
                        const newItem = $("<li>").text(`${item.time} - คนทั้งหมด: ${item.num_faces} สนใจ: ${item.interested} ไม่สนใจ: ${item.not_interested} สนใจ: ${item.percent_interested}% ไม่สนใจ: ${item.percent_not_interested}%`);
                        historyList.append(newItem);
                    });
                    // ตรวจสอบจำนวนรายการใน emotion-history
        if (historyList.children().length > 20) {
            // ถ้ามีรายการเกิน 20 รายการ จะให้มีฟังก์ชันการเลื่อน
            historyList.css("overflow-y", "auto");
        } else {
            // ถ้ามีรายการไม่เกิน 20 จะยกเลิกการเลื่อน
            historyList.css("overflow-y", "hidden");
        }
                });
            }
            setInterval(updateHistory, 2000);
        </script>
    </head>
    <body>
        <h1>ระบบตรวจจับอารมณ์จากใบหน้า</h1>
        <div class='container'>
            <div class='video-box'>
                <img src='/video_feed' width="640" height="480">
            </div>
            <div class='stats-box'>
                <h3>รายการย้อนหลัง</h3>
                <ul id="emotion-history"></ul>
            <a href="/export_excel" class="btn">Export to Excel</a>  <!-- เพิ่มปุ่ม Export -->
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history_data():
    result = []
    for time, num_faces, interested, not_interested in face_count_history:
        total_faces = interested + not_interested if (interested + not_interested) > 0 else 1  # ป้องกันหารด้วยศูนย์
        percent_interested = round((interested / total_faces) * 100, 2)  # คำนวณเปอร์เซ็นต์ของ "สนใจ"
        percent_not_interested = round((not_interested / total_faces) * 100, 2)  # คำนวณเปอร์เซ็นต์ของ "ไม่สนใจ"
        
        # เพิ่มข้อมูลในผลลัพธ์
        result.append({
            "time": time,
            "num_faces": num_faces,
            "interested": interested,
            "not_interested": not_interested,
            "percent_interested": percent_interested,
            "percent_not_interested": percent_not_interested
        })
    
    return jsonify(result)


# ฟังก์ชันส่งออกข้อมูลเป็นไฟล์ Excel
@app.route('/export_excel')
def export_excel():
    # สร้าง DataFrame จาก history data
    data = []
    for time, num_faces, interested, not_interested in face_count_history:
        total_faces = interested + not_interested if (interested + not_interested) > 0 else 1
        percent_interested = round((interested / total_faces) * 100, 2)
        percent_not_interested = round((not_interested / total_faces) * 100, 2)
        
        data.append({
            "Time": time,
            "Num Faces": num_faces,
            "Interested": interested,
            "Not Interested": not_interested,
            "Percent Interested": f"{percent_interested}%",
            "Percent Not Interested": f"{percent_not_interested}%"
        })
    
    # สร้าง DataFrame
    df = pd.DataFrame(data)

    # สร้าง Excel ไฟล์ใน memory
    excel_file = BytesIO()
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Emotion History')

    # รีเซ็ตตำแหน่ง pointer ของ BytesIO กลับไปที่เริ่มต้น
    excel_file.seek(0)

    # ส่งไฟล์ Excel ให้ผู้ใช้ดาวน์โหลด
    return Response(
        excel_file,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment;filename=emotion_history.xlsx"}
    )


if __name__ == '__main__':
    app.run(debug=False)
