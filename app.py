from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
from flask_socketio import SocketIO, emit
import base64
import pyttsx3

# Initialize Flask & SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Model & Label Encoder
model = tf.keras.models.load_model("models/gesture_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

label_map = {i: label for i, label in enumerate(label_encoder)}

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
engine = pyttsx3.init()
@app.route("/")
def index():
    return render_template("index1.html")

@app.route('/try-now')
def try_now():
    return render_template("index.html")


@socketio.on("video_frame")
def handle_video(data):
    try:
        # Convert base64 image to OpenCV format
        img_data = base64.b64decode(data["frame"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture_text = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                expected_input_size = model.input_shape[1]  # Get expected input size

                if len(keypoints) == expected_input_size:
                    keypoints = np.array(keypoints).reshape(1, -1)
                    prediction = model.predict(keypoints)
                    gesture_label = np.argmax(prediction)
                    gesture_text = label_map.get(gesture_label, "Unknown")

        # Send recognized gesture to frontend
        emit("gesture_result", {"gesture": gesture_text})
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
