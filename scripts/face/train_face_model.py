import os
import pickle
import face_recognition

known_encodings = []
known_names = []

faces_dir = "ml/data/faces"

for person in os.listdir(faces_dir):

    person_dir = os.path.join(faces_dir, person)

    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):

        image_path = os.path.join(person_dir, image_name)

        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No face found in {image_name}")
            continue

        known_encodings.append(encodings[0])
        known_names.append(person)

        print(f"Processed: {image_name}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("ml/models/face_model.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nFace model saved successfully.")
print(f"Total face samples: {len(known_names)}")
