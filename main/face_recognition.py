import pickle
from datetime import datetime
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


def get_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def mark_attendance(name):
    from .models import Student, Attendance

    try:
        student = Student.objects.get(name=name)
    except Student.DoesNotExist:
        print(f"Student {name} not registered")
        return None

    now = datetime.now()
    current_date = now.date()
    current_time = now.time().replace(microsecond=0)

    attendance, created = Attendance.objects.get_or_create(
        student=student,
        date=current_date,
        defaults={
            'time': current_time,
            'status': 'Present',
        }
    )

    if created:
        print(f"{name} attendance recorded at {current_time}")
    else:
        print(f"{name} already marked today at {attendance.time}")

    return attendance


def capture_faces(name):
    cv2 = get_cv2()
    if cv2 is None:
        print("OpenCV is not installed. Install opencv-python to use face capture.")
        return 0

    if not name:
        print("Name is required for face capture.")
        return 0

    person_path = os.path.join(DATASET_PATH, name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Unable to open camera for capture")
        return 0

    count = 0
    max_images = 20
    print("Starting face capture. Press 'q' to stop when done.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if count >= max_images:
                    break
                count += 1
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                file_path = os.path.join(person_path, f"{count}.jpg")
                cv2.imwrite(file_path, face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Captured: {count}/{max_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name}")
    return count


def run_recognition():
    cv2 = get_cv2()
    if cv2 is None:
        print("OpenCV is not installed. Install opencv-python to use face recognition.")
        return []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Unable to open camera")
        return []

    recognized_results = []
    recognized_names = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)
            name = "Unknown"

            if confidence < 70:
                name = label_map.get(label, "unknown")
                if name != "unknown" and name not in recognized_names:
                    attendance = mark_attendance(name)
                    if attendance:
                        recognized_names.add(name)
                        recognized_results.append({
                            'name': name,
                            'date': attendance.date,
                            'time': attendance.time,
                        })

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return recognized_results


def train_recognizer():
    cv2 = get_cv2()
    if cv2 is None:
        return {"success": False, "message": "OpenCV is not installed. Install opencv-python to train the model."}

    if not os.path.exists(DATASET_PATH):
        return {"success": False, "message": f"Dataset path not found: {DATASET_PATH}"}

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        return {"success": False, "message": "No face images found in dataset."}

    try:
        faces_np = np.array(faces, dtype=np.uint8)
        labels_np = np.array(labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_np, labels_np)
        recognizer.save(TRAINER_PATH)

        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

        return {"success": True, "message": f"Training completed. Model saved to {TRAINER_PATH}."}
    except Exception as e:
        return {"success": False, "message": f"Training failed: {e}"}
