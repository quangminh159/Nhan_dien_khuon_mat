import csv
import cv2
import os
from mtcnn import MTCNN

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def takeImages():
    Id = input("Enter Your Id: ").strip()
    name = input("Enter Your Name: ").strip()

    if not is_number(Id):
        print("⚠️ Error: ID must be a number!")
        return

    if not name.isalpha():
        print("⚠️ Error: Name must be alphabetic!")
        return

    cam = cv2.VideoCapture(0)
    detector = MTCNN() 

    sampleNum = 0
    save_dir = os.path.join("FRAS", "TrainingImage")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n📸 Capturing images... Press 'q' to quit early.\n")

    while True:
        ret, img = cam.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        faces = detector.detect_faces(img_rgb)

        for face in faces:
            x, y, w, h = face['box']
            sampleNum += 1
            face_img = img[y:y+h, x:x+w] 
            img_path = os.path.join(save_dir, f"{name}.{Id}.{sampleNum}.jpg")
            cv2.imwrite(img_path, face_img)

            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
            cv2.imshow('Frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"✅ {sampleNum} Images Saved for ID: {Id}, Name: {name}")
    csv_path = os.path.join("FRAS", "StudentDetails", "StudentDetails.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if not file_exists:
            writer.writerow(["Id", "Name"])  
        writer.writerow([Id, name])

    print("📝 Student details saved successfully!")
