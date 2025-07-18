import argparse
from romsi_hate_speech import Predictor

def main():
    parser = argparse.ArgumentParser(
        description="Detect Romanized Sinhala hate speech using XLM-RoBERTa"
    )
    parser.add_argument(
        "texts",
        metavar="TEXT",
        type=str,
        nargs="+",
        help="One or more Romanized Sinhala texts to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sakunchamikara/romsi-hate-speech",
        help="HuggingFace model name or local path (default: sakunchamikara/romsi-hate-speech)"
    )

    args = parser.parse_args()

    predictor = Predictor(model_path=args.model)

    for text in args.texts:
        label, confidence = predictor.predict(text)
        label_str = "hate" if label == 1 else "non-hate"
        print(f'"{text}" → {label_str} (confidence: {confidence:.4f})')

if __name__ == "__main__":
    main()
