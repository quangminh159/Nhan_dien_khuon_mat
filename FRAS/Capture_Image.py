import csv
import cv2
import os

# Kiểm tra xem chuỗi có phải số không
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Hàm chụp ảnh và lưu dữ liệu
def takeImages():
    Id = input("Enter Your Id: ").strip()
    name = input("Enter Your Name: ").strip()

    # Kiểm tra hợp lệ
    if not is_number(Id):
        print("⚠️ Error: ID must be a number!")
        return
    if not name.isalpha():
        print("⚠️ Error: Name must be alphabetic!")
        return

    # Mở camera
    cam = cv2.VideoCapture(0)
    harcascadePath = os.path.join("FRAS", "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)

    sampleNum = 0
    save_dir = os.path.join("FRAS", "TrainingImage")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n📸 Capturing images... Press 'q' to quit early.\n")

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            sampleNum += 1
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(save_dir, f"{name}.{Id}.{sampleNum}.jpg")
            cv2.imwrite(img_path, face_img)

            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
            cv2.imshow('Frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"✅ {sampleNum} Images Saved for ID: {Id}, Name: {name}")

    # Ghi vào file CSV
    csv_path = os.path.join("FRAS", "StudentDetails", "StudentDetails.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if not file_exists:
            writer.writerow(["Id", "Name"])  # Ghi tiêu đề nếu file chưa tồn tại
        writer.writerow([Id, name])

    print("📝 Student details saved successfully!")

