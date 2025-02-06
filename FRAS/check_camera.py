import cv2
from mtcnn import MTCNN

def camer():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()

    while True:
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)

        cv2.imshow('Webcam Check', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

