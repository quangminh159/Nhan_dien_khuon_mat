import datetime
import os
import time
import cv2
import pandas as pd
from mtcnn import MTCNN

# -------------------------
def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    model_path = os.path.join("FRAS", "TrainingImageLabel", "Trainner.yml")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: File '{model_path}' không tồn tại!")
        return

    recognizer.read(model_path)

    student_csv_path = os.path.join("FRAS", "StudentDetails", "StudentDetails.csv")
    if not os.path.exists(student_csv_path):
        print(f"❌ Lỗi: File '{student_csv_path}' không tồn tại!")
        return

    df = pd.read_csv(student_csv_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    # Initialize MTCNN face detector
    detector = MTCNN()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  
    cam.set(4, 480)  

    while True:
        ret, im = cam.read()
        if not ret:
            print("❌ Lỗi: Không thể truy cập camera!")
            break

        # Convert image to RGB for MTCNN
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                name_row = df.loc[df['Id'] == Id, 'Name']
                if not name_row.empty:
                    name = name_row.values[0]
                else:
                    name = "Unknown"

                confstr = "  {0}%".format(round(100 - conf))
                tt = f"{Id} - {name}"
            else:
                Id = "Unknown"
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

            if (100 - conf) > 65:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                attendance.loc[len(attendance)] = {'Id': Id, 'Name': name, 'Date': date, 'Time': timeStamp}

            if (100 - conf) > 65:
                tt = tt + " [Pass]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                tt = tt + " [Fail]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            color = (0, 255, 0) if (100 - conf) > 67 else (0, 255, 255) if (100 - conf) > 50 else (0, 0, 255)
            cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, color, 1)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

        cv2.imshow('Attendance', im)

        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")

    attendance_folder = os.path.join("FRAS", "Attendance")
    os.makedirs(attendance_folder, exist_ok=True)

    fileName = os.path.join(attendance_folder, f"Attendance_{date}_{Hour}-{Minute}-{Second}.csv")
    attendance.to_csv(fileName, index=False)

    print(f"✅ Attendance Successful! File saved at '{fileName}'")

    cam.release()
    cv2.destroyAllWindows()

