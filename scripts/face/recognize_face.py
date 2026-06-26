import cv2
import pickle
import face_recognition

# Load trained face data
with open("ml/models/face_model.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

camera = cv2.VideoCapture(0)

# Faster camera settings
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Face Recognition...")
print("Press Q to quit")

process_this_frame = True

face_locations = []
face_names = []

while True:

    ret, frame = camera.read()

    if not ret:
        break

    if process_this_frame:

        # Resize for faster processing
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_small,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        face_names = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=0.55
            )

            name = "Unknown"

            if True in matches:
                first_match = matches.index(True)
                name = known_names[first_match]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Draw results
    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()