import os
import numpy as np
import cv2

# ชื่อของอารมณ์
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
interested_labels = ["happy", "neutral", "surprise"]

def load_data_from_dir(dataset_dir):
    images, labels = [], []
    for emotion in emotion_labels:
        emotion_folder = os.path.join(dataset_dir, emotion)
        if not os.path.exists(emotion_folder):
            continue
        for img_file in os.listdir(emotion_folder):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(emotion_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            label = 1 if emotion in interested_labels else 0
            images.append(img)
            labels.append(label)
    X = np.array(images).reshape(-1, 48, 48, 1) / 255.0
    y = np.array(labels)
    return X, y

# โหลดข้อมูล
X_train, y_train = load_data_from_dir('D:/AROM/fer2013/train')
X_test, y_test = load_data_from_dir('D:/AROM/fer2013/test')

# บันทึก
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
