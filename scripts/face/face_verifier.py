import cv2
import pickle
import numpy as np
import face_recognition


def verify_face():

    with open("ml/models/face_model.pkl", "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return None

    print("\nLooking for a face...")

    while True:

        ret, frame = camera.read()

        if not ret:
            continue

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_frame,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )

        for face_encoding in face_encodings:

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.48:

                name = known_names[best_match_index]

                camera.release()
                cv2.destroyAllWindows()

                return name

        cv2.imshow("Face Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            camera.release()
            cv2.destroyAllWindows()

            return None
