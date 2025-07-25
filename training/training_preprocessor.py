import re
from transformers import AutoTokenizer
from training.config import Config
from training.transliterator import transliterate, transliterate_mixed_text

class TrainingPreprocessor:
    def __init__(self, model_name=Config.MODEL_NAME, adhoc=False):
        self.adhoc = adhoc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean(self, text):
        if self.adhoc:
            text = transliterate_mixed_text(text)
        else:
            text = transliterate(text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return self.tokenizer(text["text"], truncation=True, padding="max_length", max_length=128)
