import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

# -------------- image labels ------------------------

def getImagesAndLabels(path):
    if not os.path.exists(path):
        print(f"❌ Lỗi: Thư mục '{path}' không tồn tại!")
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Chuyển ảnh sang grayscale
            imageNp = np.array(pilImage, 'uint8')

            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Lấy ID từ tên file
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"⚠ Lỗi khi xử lý ảnh '{imagePath}': {e}")

    return faces, Ids

# ----------- train images function ---------------
def TrainImages():
    # Sử dụng đúng cú pháp OpenCV
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    harcascadePath = os.path.join("FRAS", "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)

    training_path = os.path.join("FRAS", "TrainingImage")
    faces, Id = getImagesAndLabels(training_path)

    if len(faces) == 0 or len(Id) == 0:
        print("❌ Không có dữ liệu hình ảnh để huấn luyện!")
        return

    print("✅ Bắt đầu huấn luyện mô hình...")
    recognizer.train(faces, np.array(Id))  # Không cần `Thread()`

    output_folder = os.path.join("FRAS", "TrainingImageLabel")
    os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

    model_path = os.path.join(output_folder, "Trainner.yml")
    recognizer.save(model_path)

    print(f"✅ Huấn luyện hoàn tất! Model đã lưu tại '{model_path}'")

    # Hiển thị số lượng ảnh đã huấn luyện
    counter_img(training_path)

# Optional: Hiển thị số lượng ảnh đã huấn luyện
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for _ in imagePaths:
        print(f"{imgcounter} Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1
    print("\n✅ Tất cả hình ảnh đã được huấn luyện.")

# Chạy thử
if __name__ == "__main__":
    TrainImages()
