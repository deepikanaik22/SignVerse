import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3

# Load trained model
model = tf.keras.models.load_model("gesture_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize text-to-speech
engine = pyttsx3.init()

# Open camera
cap = cv2.VideoCapture(0)

print("📷 Camera started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # If hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Convert to numpy array and reshape
            keypoints = np.array(keypoints).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(keypoints)
            gesture_label = np.argmax(prediction)
            gesture_text = label_encoder.classes_[gesture_label]

            print(f"🤖 Recognized Gesture: {gesture_text}")
            engine.say(gesture_text)
            engine.runAndWait()

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed.")
