import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread
from mtcnn import MTCNN

# -------------- image labels ------------------------

def getImagesAndLabels(path):
    if not os.path.exists(path):
        print(f"❌ Lỗi: Thư mục '{path}' không tồn tại!")
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    detector = MTCNN()

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('RGB')  
            imageNp = np.array(pilImage, 'uint8')
            results = detector.detect_faces(imageNp)
            for result in results:
                x, y, w, h = result['box']
                face = imageNp[y:y+h, x:x+w]  
                faces.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))  
                Id = int(os.path.split(imagePath)[-1].split(".")[1])  
                Ids.append(Id)
        except Exception as e:
            print(f"⚠ Lỗi khi xử lý ảnh '{imagePath}': {e}")

    return faces, Ids

# ----------- train images function ---------------
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    training_path = os.path.join("FRAS", "TrainingImage")
    faces, Id = getImagesAndLabels(training_path)

    if len(faces) == 0 or len(Id) == 0:
        print("❌ Không có dữ liệu hình ảnh để huấn luyện!")
        return

    print("✅ Bắt đầu huấn luyện mô hình...")
    recognizer.train(faces, np.array(Id))  

    output_folder = os.path.join("FRAS", "TrainingImageLabel")
    os.makedirs(output_folder, exist_ok=True)  

    model_path = os.path.join(output_folder, "Trainner.yml")
    recognizer.save(model_path)

    print(f"✅ Huấn luyện hoàn tất! Model đã lưu tại '{model_path}'")
    counter_img(training_path)

def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for _ in imagePaths:
        print(f"{imgcounter} Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1
    print("\n✅ Tất cả hình ảnh đã được huấn luyện.")

if __name__ == "__main__":
    TrainImages()
