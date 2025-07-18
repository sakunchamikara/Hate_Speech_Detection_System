from fastapi import FastAPI
from pydantic import BaseModel
from romsi_hate_speech import Predictor

# ====== FastAPI metadata ======
app = FastAPI(
    title="Romanized Sinhala Hate Speech Detection API",
    description="Detect hate speech in Romanized Sinhala using a fine-tuned XLM-RoBERTa model hosted on HuggingFace Hub.",
    version="1.0.0"
)

MODEL_NAME = "sakunchamikara/romsi-hate-speech"

predictor = Predictor(model_path=MODEL_NAME)

class TextRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(request: TextRequest):
    results = []
    for text in request.texts:
        label, confidence = predictor.predict(text)
        results.append({
            "text": text,
            "label": "hate" if label == 1 else "non-hate",
            "score": round(confidence, 4)
        })
    return results
