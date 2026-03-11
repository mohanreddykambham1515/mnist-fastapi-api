from fastapi import FastAPI
from tensorflow import keras
import numpy as np

app = FastAPI()

# Load trained model
model = keras.models.load_model("mnist_model.h5")


@app.get("/")
def home():
    return {"message": "MNIST Digit Prediction API"}


@app.post("/predict")
def predict_digit(pixels: list):

    # Convert input to numpy
    data = np.array(pixels).reshape(1, 784)

    prediction = model.predict(data)

    digit = int(np.argmax(prediction))

    return {"predicted_digit": digit}