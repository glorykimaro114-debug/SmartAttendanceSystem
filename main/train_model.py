import cv2
import os
import numpy as np

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # resize to consistent size
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# 🔥 IMPORTANT FIX: convert properly (NOT object dtype)
faces = np.array(faces, dtype=np.uint8)
labels = np.array(labels)

print("Faces shape:", faces.shape)
print("Labels shape:", labels.shape)

recognizer.train(faces, labels)

recognizer.save("trainer.yml")

print("Training completed successfully!")
print("Model saved as trainer.yml")
print("Labels:", label_map)
import pickle
with open("labels.pkl", "wb") as f :
    pickle.dump(label_map, f)
