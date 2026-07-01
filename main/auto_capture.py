import cv2
import os

from main.camera_utils import open_camera

name = input("Enter name: ")

path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = open_camera()

count = 0

print("Starting face capture... Look at camera")

while True:
    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]

        file_path = f"{path}/{count}.jpg"
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    display_frame = cv2.flip(frame, 1)
    cv2.imshow("Auto Face Capture", display_frame)

    key = cv2.waitKey(1)

    # stop conditions
    if key == ord('q') or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()

print(f"Done! {count} images saved for {name}")
