import cv2
import pickle

import csv
from datetime import datetime
import os

from main.camera_utils import open_camera


def mark_attendance(name):
    file_name = "attendance.csv"

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    already_marked_today = False

    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) >= 3:
                recorded_name = row[0]
                recorded_date = row[1]
                if recorded_name == name and recorded_date == date:
                    already_marked_today = True
                    break

    if not already_marked_today:
        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])

        print(f"{name} attendance recorded")


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

with open("labels.pkl","rb") as f:
    label_map = pickle.load(f) # tutaboost baadaye kutoka training

cam = open_camera()

while True:
    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        name = "Unknown"

        if confidence < 70:
            name = label_map.get(label,"unknown")
            if name != "unknown":
                mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), 2)

    display_frame = cv2.flip(frame, 1)
    cv2.imshow("Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
