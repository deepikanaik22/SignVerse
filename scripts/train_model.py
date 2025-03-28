import numpy as np
import tensorflow as tf
import os
import pickle

# Define dataset path
DATASET_PATH = "dataset/"

# Auto-detect all gesture folders inside the dataset directory
GESTURES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

# Check if dataset is found
if not GESTURES:
    raise ValueError("❌ No gesture data found in the dataset folder!")

print(f"📂 Found Gestures: {GESTURES}")

# Initialize data storage
X, Y = [], []

# Load data dynamically from detected gesture folders
for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATASET_PATH, gesture)

    for file in os.listdir(gesture_path):
        if file.endswith(".npy"):  # Ensure only .npy files are loaded
            data = np.load(os.path.join(gesture_path, file))
            X.append(data)
            Y.append(label)

# Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Save Label Encoder
os.makedirs("models", exist_ok=True)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(GESTURES, f)

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(GESTURES), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, Y, epochs=20)

# Save Model
model.save("models/gesture_model.h5")

print("✅ Model trained and saved successfully!")
