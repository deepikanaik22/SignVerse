import tensorflow as tf
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyttsx3

# Load Model & Label Encoder
model = tf.keras.models.load_model("models/gesture_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

label_map = {i: label for i, label in enumerate(label_encoder)}

# Initialize Mediapipe & Text-to-Speech
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
engine = pyttsx3.init()

# Open Camera
cap = cv2.VideoCapture(0)

print("📷 Camera started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            expected_input_size = model.input_shape[1]  # Get expected input size

            print(f"🔹 Keypoints Length: {len(keypoints)}, Expected: {expected_input_size}")  # Debugging

            if len(keypoints) == expected_input_size:  # Ensure correct shape
                keypoints = np.array(keypoints).reshape(1, -1)
                prediction = model.predict(keypoints)
                gesture_label = np.argmax(prediction)
                gesture_text = label_map.get(gesture_label, "Unknown")

                print(f"🤖 Recognized Gesture: {gesture_text}")
                engine.say(gesture_text)
                engine.runAndWait()

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed.")
