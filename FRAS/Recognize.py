import datetime
import os
import time
import cv2
import pandas as pd

# -------------------------
def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    model_path = os.path.join("FRAS", "TrainingImageLabel", "Trainner.yml")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: File '{model_path}' không tồn tại!")
        return

    recognizer.read(model_path)

    harcascadePath = os.path.join("FRAS", "haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    student_csv_path = os.path.join("FRAS", "StudentDetails", "StudentDetails.csv")
    if not os.path.exists(student_csv_path):
        print(f"❌ Lỗi: File '{student_csv_path}' không tồn tại!")
        return

    df = pd.read_csv(student_csv_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    # Khởi tạo webcam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        if not ret:
            print("❌ Lỗi: Không thể truy cập camera!")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
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
                cv2.putText(im, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)
            else:
                tt = tt + " [Fail]"
                cv2.putText(im, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)

            color = (0, 255, 0) if (100 - conf) > 67 else (0, 255, 255) if (100 - conf) > 50 else (0, 0, 255)
            cv2.putText(im, str(confstr), (x+5, y+h-5), font, 1, color, 1)

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)

        if cv2.waitKey(1) == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")

    attendance_folder = os.path.join("FRAS", "Attendance")
    os.makedirs(attendance_folder, exist_ok=True)  # Đảm bảo thư mục tồn tại

    fileName = os.path.join(attendance_folder, f"Attendance_{date}_{Hour}-{Minute}-{Second}.csv")
    attendance.to_csv(fileName, index=False)
    
    print(f"✅ Attendance Successful! File saved at '{fileName}'")
    
    cam.release()
    cv2.destroyAllWindows()


