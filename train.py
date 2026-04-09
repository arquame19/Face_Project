import face_recognition
import os
import pickle

dataset_path = "dataset"

known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

# Save encodings
with open("encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Training complete!")