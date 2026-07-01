import cv2
import os

name = input("Enter your name: ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]

        file_path = f"{path}/{count}.jpg"
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
