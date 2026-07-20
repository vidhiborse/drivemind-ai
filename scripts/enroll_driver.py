"""
Driver enrollment script — captures a photo from webcam and saves it as
a registered driver profile image. DeepFace will use this as the reference
for recognition later.

Run this once per driver you want to register.
"""

import sys
sys.path.insert(0, "src")

import cv2
import os

PROFILES_DIR = "data/driver_profiles"

driver_name = input("Enter driver name (no spaces, e.g. 'vidhi'): ").strip()
if not driver_name:
    print("Name cannot be empty.")
    sys.exit(1)

os.makedirs(PROFILES_DIR, exist_ok=True)
save_path = os.path.join(PROFILES_DIR, f"{driver_name}.jpg")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Press SPACE to capture your photo, or 'q' to cancel.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "Press SPACE to capture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Driver Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(save_path, frame)
        print(f"Saved driver profile: {save_path}")
        break
    elif key == ord("q"):
        print("Cancelled.")
        break

cap.release()
cv2.destroyAllWindows()