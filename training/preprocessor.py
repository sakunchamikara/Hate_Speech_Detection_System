import re
from transformers import AutoTokenizer
from training.config import Config
from training.transliterator import transliterate, transliterate_mixed_text

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def tokenize_function(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=128)

def clean_and_transliterate(text, adhoc=False):
    if adhoc:
        text = transliterate_mixed_text(text)
    else:
        text = transliterate(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text