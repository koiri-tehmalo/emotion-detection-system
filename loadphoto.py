import os
import numpy as np
import cv2

# กำหนด path ไปที่โฟลเดอร์ที่มีข้อมูล
dataset_dir = 'D:/AROM/fer2013/train'

# ชื่อของอารมณ์ที่ใช้ใน FER-2013
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# กำหนดค่าของอารมณ์ที่เป็น Interested และ Not Interested
interested_labels = ["happy", "neutral", "surprise"]
not_interested_labels = ["angry", "disgust", "fear", "sad"]

# สร้าง list สำหรับเก็บข้อมูลภาพและ label
images = []
labels = []

# อ่านไฟล์ภาพจากแต่ละโฟลเดอร์
for emotion_index, emotion in enumerate(emotion_labels):
    emotion_folder = os.path.join(dataset_dir, emotion)
    
    if not os.path.exists(emotion_folder):
        print(f"โฟลเดอร์ {emotion_folder} ไม่พบ!")
        continue  # ข้ามไปยังอารมณ์ถัดไปหากโฟลเดอร์ไม่พบ
    
    # อ่านไฟล์ภาพในโฟลเดอร์อารมณ์
    for img_file in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_file)

        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (48, 48))  

        # แปลง label เป็น 2 อารมณ์
        if emotion in interested_labels:
            labels.append(1)  # Interested
        else:
            labels.append(0)  # Not Interested
        
        images.append(img)

# แปลง list เป็น numpy array
X = np.array(images)
y = np.array(labels)

X = X.reshape(-1, 48, 48, 1) / 255.0  # Normalize ข้อมูลภาพ
y = np.array([1 if label == 1 else 0 for label in y])  # แปลง labels ให้เป็น 2 อารมณ์

# ตรวจสอบขนาดข้อมูล
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# บันทึกข้อมูลเป็นไฟล์ .npy
np.save('X_data.npy', X)
np.save('y_data.npy', y)

print("ข้อมูลได้ถูกบันทึกแล้วในไฟล์ 'X_data.npy' และ 'y_data.npy'")
