from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict
from model.model import __version__ as model_version

app = FastAPI()

class StepsIn(BaseModel):
    steps:int

class PredictionOut(BaseModel):
    predictions: list

@app.get("/")
def home():
    return {"health_check": "OK"
            ,"model_version": model_version
            }

@app.post("/forecast", response_model=PredictionOut)
def forecast(payload: PredictionOut):
    predictions = predict(payload.predictions)
    return {"predictions": predictions}