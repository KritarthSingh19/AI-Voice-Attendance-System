import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime

print("STEP 0: Starting recognition script")

try:
    with open("ml/models/face_model.pkl", "rb") as f:
        data = pickle.load(f)

    print("STEP 1: Face model loaded successfully")

except Exception as e:
    print(f"ERROR LOADING MODEL: {e}")
    exit()

known_encodings = data["encodings"]
known_names = data["names"]

print(f"STEP 2: Loaded {len(known_encodings)} face encodings")

camera = cv2.VideoCapture(0)

print("STEP 3: Camera object created")

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("STEP 4: Camera opened successfully")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_locations = []
face_names = []
process_this_frame = True

print("STEP 5: Starting recognition loop")
print("Press Q to quit")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Failed to read frame from camera")
        break

    if process_this_frame:

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )

        face_names = []

        for face_encoding in face_encodings:

            name = "Unknown"

            if len(known_encodings) > 0:

                face_distances = face_recognition.face_distance(
                    known_encodings,
                    face_encoding
                )

                best_match_index = np.argmin(face_distances)

                best_distance = face_distances[best_match_index]

                if best_distance < 0.50:
                    name = known_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.rectangle(
        frame,
        (20, 20),
        (420, 130),
        (20, 20, 20),
        -1
    )

    cv2.rectangle(
        frame,
        (20, 20),
        (420, 130),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "AI ATTENDANCE VERIFICATION",
        (35, 55),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Time: {current_time}",
        (35, 95),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        verified = name != "Unknown"

        color = (0, 255, 0) if verified else (0, 0, 255)

        corner = 30

        cv2.line(frame, (left, top), (left + corner, top), color, 2)
        cv2.line(frame, (left, top), (left, top + corner), color, 2)

        cv2.line(frame, (right, top), (right - corner, top), color, 2)
        cv2.line(frame, (right, top), (right, top + corner), color, 2)

        cv2.line(frame, (left, bottom), (left + corner, bottom), color, 2)
        cv2.line(frame, (left, bottom), (left, bottom - corner), color, 2)

        cv2.line(frame, (right, bottom), (right - corner, bottom), color, 2)
        cv2.line(frame, (right, bottom), (right, bottom - corner), color, 2)

        status = "VERIFIED" if verified else "UNKNOWN"

        cv2.putText(
            frame,
            status,
            (left, top - 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (left, bottom + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow(
        "AI Attendance Verification Terminal",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Exiting...")
        break

camera.release()
cv2.destroyAllWindows()

print("Recognition script ended")