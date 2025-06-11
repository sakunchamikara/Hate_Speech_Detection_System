from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Predictor:
    def __init__(self, model_path="models/trained_xlm_roberta"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][label].item()
        return label, confidence
