import cv2
import numpy as np
import os

# Khởi tạo webcam
cam = cv2.VideoCapture(0)

# Tải bộ phân loại Haar Cascade để phát hiện khuôn mặt
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập ID cho người dùng
Id = input('Enter your ID: ')
sampleNum = 0

# Tạo thư mục để lưu dữ liệu nếu chưa có
if not os.path.exists('Dataset'):
    os.makedirs('Dataset')

while(True):
    ret, img = cam.read()  # Đọc hình ảnh từ webcam
    if not ret:
        break

    # Hiển thị video trực tiếp
    cv2.imshow('frame', img)

    # Chuyển hình ảnh thành ảnh xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện và lưu hình ảnh
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật

        # Lưu ảnh khuôn mặt vào thư mục "Dataset" với tên file theo ID và số mẫu
        cv2.imwrite(f"Dataset/{Id}_{sampleNum}.jpg", gray[y:y + h, x:x + w])

        # Tăng số mẫu đã lưu
        sampleNum += 1

    # Hiển thị ảnh với hình chữ nhật vẽ xung quanh khuôn mặt
    cv2.imshow('frame', img)

    # Dừng lại nếu nhấn phím 'q' hoặc đã chụp đủ 30 ảnh
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif sampleNum > 30:
        break

# Giải phóng webcam và đóng cửa sổ OpenCV
cam.release()
cv2.destroyAllWindows()
