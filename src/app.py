from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import csv
import datetime

# -------------------------
# CONFIG
# -------------------------
dataset_path = "permafrost_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

app = Flask(__name__)
log_file = "logs/predictions_log.csv"
os.makedirs("logs", exist_ok=True)

# -------------------------
# TRAIN MODEL ON STARTUP
# -------------------------
print("⏳ Training model... please wait")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train for fewer epochs to keep it fast
history = model.fit(train_data, validation_data=val_data, epochs=3)
print("✅ Model trained and ready for predictions!")

# -------------------------
# PREDICTION ROUTE
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    img = Image.open(file).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Thawing" if prediction > 0.5 else "Stable"

    # Log the prediction
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([file.filename, float(prediction), result, datetime.datetime.now()])

    return jsonify({"prediction": result, "confidence": float(prediction)})

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
