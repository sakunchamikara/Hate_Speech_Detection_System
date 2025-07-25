
# Romanized Sinhala Hate Speech Detection

Detect hate speech in **Romanized Sinhala text** using a fine-tuned deep learning model based on XLM-RoBERTa.  
This project includes:  
✅ Training code  
✅ Inference code  
✅ Pip-installable package  
✅ REST API (FastAPI)  
✅ CLI tool  

---

## 📚 About

This project was developed as part of my MSc research to address hate speech detection in Romanized Sinhala, commonly used in Sri Lankan social media.

The model is trained on the SOLD dataset, fine-tuned on XLM-RoBERTa, and exposes predictions through a REST API, a CLI, and as a reusable Python package.

---

## 🚀 Features

- Fine-tuned multilingual transformer for Romanized Sinhala text.
- Inference available via:
  - Python package (`romsi_hate_speech`)
  - CLI command: `romsi-detect`
  - REST API (FastAPI server)
- Training pipeline to reproduce experiments.
- MIT licensed and open source.

---

## 🗂️ Project Structure

```
romanized_hate_speech_detection/
├── romsi_hate_speech/         # Inference package (pip-installable)
│   ├── __init__.py
│   ├── predictor.py
│   ├── api.py
│   ├── cli.py
├── training/                  # Training and evaluation code
│   ├── trainer.py
│   ├── evaluator.py
│   ├── data_loader.py
│   └── ...
├── models/                    # Saved models
├── data/                      # Datasets and preprocessing scripts
├── README.md                  # This file
├── setup.py                   # Packaging metadata
├── requirements.txt
├── .gitignore
├── LICENSE
```

---

## 🔷 Installation

You can install the inference package locally:

```bash
pip install .
```

Or (once published):

```bash
pip install romsi-hate-speech
```

---

## 🧪 Usage

### 🐍 Python
```python
from romsi_hate_speech import Predictor

predictor = Predictor(model_path="sakunchamikara/romsi-hate-speech")
label, confidence = predictor.predict("meka thamai mage msc research project eka")
print(label, confidence)
```

### 💻 CLI
```bash
romsi-detect "meka thamai mage msc research project eka"
```

### 🌐 REST API
Run the API server:
```bash
uvicorn romsi_hate_speech.api:app --reload
```

Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Or POST to `/predict`:
```json
{
  "texts": ["patta horekta yahapalanayen"]
}
```

---

## 📈 Training

To reproduce training:
```bash
python training/model_trainer.py
```

You can configure hyperparameters in `training/config.py`.

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

- Sakun Chamikara
- MSc Research Project, 2025

---

## 🌐 Links

- [HuggingFace Model](https://huggingface.co/sakunchamikara/romsi-hate-speech) (if applicable)
- [PyPI Package](https://pypi.org/project/romsi-hate-speech/) (if applicable)

---
