from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Predictor:
    def __init__(self, model_path="models/trained_xlm_roberta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            labels = torch.argmax(probs, dim=1)

        results = []
        for text, label, confidence in zip(
            texts, labels.cpu(), probs.max(dim=1).values.cpu()
        ):
            label_str = "hate" if label.item() == 1 else "non-hate"
            results.append({
                "text": text,
                "label": label_str,
                "confidence": round(confidence.item(), 4)
            })
        return results
