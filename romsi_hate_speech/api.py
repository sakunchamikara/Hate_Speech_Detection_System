from fastapi import FastAPI
from pydantic import BaseModel
from romsi_hate_speech.predictor import Predictor

app = FastAPI(title="Romanized Sinhala Hate Speech Detection API")

predictor = Predictor(model_path="sakunchamikara/romsi-hate-speech")

class PredictRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(request: PredictRequest):
    results = predictor.predict(request.texts)
    return {"predictions": results}
