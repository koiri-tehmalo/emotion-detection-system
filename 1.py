import cv2
import numpy as np
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model('emotion_detection_model.h5')

# โหลดภาพ
img = cv2.imread('1.jpg')

# ตรวจสอบว่าไฟล์ภาพถูกโหลดสำเร็จ
if img is None:
    print("Error: Image not found or unable to load.")
else:
    # แปลงภาพเป็น grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # รีไซส์ภาพให้เป็นขนาดที่โมเดลต้องการ (48x48)
    img_resized = cv2.resize(img_gray, (48, 48))

    # ขยายขนาดไปยัง batch size และ channel ที่เหมาะสม
    img_array = np.expand_dims(img_resized, axis=-1)  # เพิ่มมิติที่หายไป (48, 48, 1)
    
    # เพิ่มมิติ batch size (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # ทำ normalization (ค่าให้เป็น [0, 1])
    img_array = img_array / 255.0

    # ทำนายผล
    predictions = model.predict(img_array)

    # แสดงผล
    print(predictions)

    # หาคลาสที่มีความน่าจะเป็นสูงสุด
    predicted_class = np.argmax(predictions, axis=1)

    # แสดงผล
    print("Predicted class:", predicted_class)

    # แสดงภาพ
    cv2.imshow("Emotion Detection with Face Detection", img)

    # รอให้ผู้ใช้กดปุ่ม (ไม่ปิดจนกว่าจะกดปุ่มใดๆ)
    cv2.waitKey(0)

    # ปิดหน้าต่างแสดงภาพ
    cv2.destroyAllWindows()
