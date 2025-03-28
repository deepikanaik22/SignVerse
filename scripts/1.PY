import cv2

cap = cv2.VideoCapture(0)  # Try changing 0 to 1 or 2 if you have multiple cameras

if not cap.isOpened():
    print("❌ Error: Could not access webcam. Try changing the index (0, 1, 2).")
else:
    print("✅ Webcam accessed successfully!")

cap.release()
