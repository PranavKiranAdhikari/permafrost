# ✅ Permafrost Thaw Detection using CNN - VS Code Version

# --------------------
# STEP 1: Import Libraries
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# --------------------
# STEP 2: Set Dataset Path
# Folder structure:
# permafrost_dataset/
# ├── thawing/
# └── stable/

dataset_path = "permafrost_dataset"

# --------------------
# STEP 3: Image Preprocessing and Data Loading
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

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

# --------------------
# STEP 4: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --------------------
# STEP 5: Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# --------------------
# STEP 6: Plot Accuracy and Loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --------------------
# STEP 7: Evaluate on Validation Data
val_data.reset()
pred_probs = model.predict(val_data)
pred_classes = (pred_probs > 0.5).astype(int).flatten()
true_classes = val_data.classes

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))

print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=list(val_data.class_indices.keys())))

# --------------------
# STEP 8: Predict on a Single Image (Optional)
img_path = "t1.png"  # Put your test image in the project folder
img = load_img(img_path, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🔴 The image is predicted as: Thawing")
else:
    print("🟢 The image is predicted as: Stable")
