import cv2
import os
import time

name = input("Enter person's name: ").strip().replace(" ", "_")

save_dir = f"ml/data/faces/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

time.sleep(2)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

count = 0

print("\nPress SPACE to capture image")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("Face Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        count += 1

        file_path = os.path.join(
            save_dir,
            f"face_{count:02d}.jpg"
        )

        cv2.imwrite(file_path, frame)

        print(f"Saved: {file_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
