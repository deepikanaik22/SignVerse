import cv2
import mediapipe as mp
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Define dataset path
DATASET_PATH = "dataset/"

# Ask for gesture name
gesture_name = input("Enter gesture name: ").strip().lower()
gesture_folder = os.path.join(DATASET_PATH, gesture_name)

# Create folder if it does not exist
os.makedirs(gesture_folder, exist_ok=True)
print(f"📂 Saving gesture data in: {gesture_folder}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

sample_count = 0  # Number of samples collected
print("📸 Press 'c' to capture gesture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])  # Store (x, y, z)

            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display text on screen
    cv2.putText(frame, f"Collecting: {gesture_name} ({sample_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Gesture Data Collection", frame)

    # Read keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Press 'c' to capture data
        if result.multi_hand_landmarks:
            np.save(os.path.join(gesture_folder, f"{sample_count}.npy"), np.array(keypoints))
            print(f"✅ Sample {sample_count} saved for '{gesture_name}'.")
            sample_count += 1
        else:
            print("⚠️ No hand detected. Try again.")
    
    if key == ord('q'):  # Press 'q' to quit
        print("🛑 Data collection complete.")
        break

cap.release()
cv2.destroyAllWindows()
