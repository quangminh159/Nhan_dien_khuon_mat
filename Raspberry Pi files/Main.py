import cv2
import numpy as np 
import os
import logging as log
import datetime as dt
from time import sleep

# Thông tin về các tên liên quan đến ID
names = ['None', 'tasnim', 'Amir']

# Khởi tạo bộ nhận diện khuôn mặt và bộ phân loại LBPH
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
log.basicConfig(filename='database.log', level=log.INFO)

# Mở tệp CSV để lưu dữ liệu nhận diện
file_path = "/home/pi/Testnew/data_log.csv"
if not os.path.exists(file_path):
    with open(file_path, "w") as file:
        file.write("Date Time, Student Name\n")

file = open(file_path, "a")

# Chuẩn bị dữ liệu huấn luyện
images = []
labels = []
for filename in os.listdir('Dataset'):
    im = cv2.imread('Dataset/' + filename, 0)
    images.append(im)
    labels.append(int(filename.split('.')[0][0]))

recognizer.train(images, np.array(labels))
print("Training Done . . . ")

# Cài đặt font và khởi tạo webcam
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
log.info("Date Time, Student Name\n")
log.info(f"Date: {str(dt.datetime.now().strftime('%d-%m-%Y'))}\n")
log.info("---------------------------------\n")

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)  # Phát hiện khuôn mặt

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật quanh khuôn mặt

        # Dự đoán khuôn mặt nhận diện
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 40:
            id = names[id]  # Gắn tên tương ứng với ID
            confidence = "  {0}%".format(round(100 - confidence))
            # Ghi thông tin vào file log
            log.info(f"{str(dt.datetime.now())}, {id}\n")
            # Ghi thông tin vào file CSV
            file.write(f"{str(dt.datetime.now().strftime('%H:%M:%S'))}, {id}\n")
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Hiển thị tên và độ tin cậy trên ảnh
        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Hiển thị kết quả nhận diện trên màn hình
    cv2.imshow('frame', frame)

    # Thoát khi nhấn phím 'ESC'
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
file.close()
