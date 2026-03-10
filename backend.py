import tensorflow as tf
import numpy as np
import cv2
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "digit_model.h5"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

 #MODEL

def create_and_train_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)

    return model


if os.path.exists(MODEL_PATH):
    print("✅ Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🚀 Training new model...")
    model = create_and_train_model()



def preprocess_image_bytes(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# api

@app.get("/")
def health():
    return {"status": "AI backend online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image_bytes(image_bytes)

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return {
        "digit": digit,
        "confidence": round(confidence, 2)
    }
