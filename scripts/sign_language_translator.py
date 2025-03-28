import tensorflow as tf
import numpy as np
import pickle
import pyttsx3
import cv2
import mediapipe as mp

# Load Model & Label Encoder
model = tf.keras.models.load_model("C:\\Users\\mouly\\OneDrive\\Desktop\\h\\models\\gesture_model.h5")
with open("C:\\Users\\mouly\\OneDrive\\Desktop\\h\\models\\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Convert label numbers to text
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

engine = pyttsx3.init()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print("📷 Camera started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        print("✅ Hand detected!")  # Debugging line

        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            print(f"🖐 Extracted {len(keypoints)} keypoints.")  # Debugging

            if len(keypoints) == 127:
                keypoints = np.array(keypoints).reshape(1, 127)
                prediction = model.predict(keypoints)
                gesture_label = np.argmax(prediction)
                gesture_text = label_map.get(gesture_label, "Unknown Gesture")

                print(f"🤖 Recognized Gesture: {gesture_text}")

                engine.say(gesture_text)
                engine.runAndWait()

    else:
        print("❌ No hand detected.")  # Debugging

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed.")
