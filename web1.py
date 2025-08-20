from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
import random  # ใช้สำหรับจำลองผลลัพธ์ emotion detection

app = Flask(__name__)

# เก็บค่าอารมณ์ย้อนหลัง 1 นาที
emotion_history = []
lock = threading.Lock()

# เปิดกล้อง
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # จำลองการตรวจจับอารมณ์
            emotion = random.choice(["Interested", "Not Interested"])
            timestamp = time.time()
            
            with lock:
                emotion_history.append((timestamp, emotion))
                # ลบข้อมูลที่เก่ากว่า 60 วินาที
                emotion_history[:] = [(t, e) for t, e in emotion_history if time.time() - t <= 60]
            
            # แสดงผลบนวิดีโอ
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    with lock:
        interested_count = sum(1 for _, e in emotion_history if e == "Interested")
        not_interested_count = sum(1 for _, e in emotion_history if e == "Not Interested")
    
    total = interested_count + not_interested_count
    if total == 0:
        return jsonify({"interested": 0, "not_interested": 0})
    
    return jsonify({
        "interested": (interested_count / total) * 100,
        "not_interested": (not_interested_count / total) * 100
    })

if __name__ == '__main__':
    app.run(debug=True)

# === HTML (index.html) ===
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Emotion Detection</title>
#     <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
#     <style>
#         body { display: flex; font-family: Arial, sans-serif; justify-content: center; align-items: center; height: 100vh; }
#         .container { display: flex; gap: 20px; }
#         .video-container, .stats-container { width: 50%; }
#         .stats-container { text-align: left; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="video-container">
#             <img src="{{ url_for('video_feed') }}" width="100%">
#         </div>
#         <div class="stats-container">
#             <h2>Emotion Stats (Last 1 Min)</h2>
#             <p>Interested: <span id="interested">0</span>%</p>
#             <p>Not Interested: <span id="not_interested">0</span>%</p>
#         </div>
#     </div>
#     <script>
#         function updateEmotionStats() {
#             $.getJSON("/emotion_data", function(data) {
#                 $("#interested").text(data.interested.toFixed(2));
#                 $("#not_interested").text(data.not_interested.toFixed(2));
#             });
#         }
#         setInterval(updateEmotionStats, 1000);
#     </script>
# </body>
# </html>
