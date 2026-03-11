from fastapi import FastAPI
import torch
import torch.nn as nn

app = FastAPI()

model = nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

@app.get("/")
def home():
    return {"message": "MNIST PyTorch API"}

@app.post("/predict")
def predict_digit(pixels: list):

    data = torch.tensor(pixels, dtype=torch.float32).reshape(1,784)

    with torch.no_grad():
        output = model(data)

    digit = torch.argmax(output, dim=1).item()

    return {"digit": digit}