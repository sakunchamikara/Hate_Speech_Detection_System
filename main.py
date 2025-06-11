import argparse
from training.trainer import get_model, get_training_args, train_model ,save_model
from training.data_loader import load_dataset_pipeline
from training.metrics import compute_metrics
from inference.predictor import Predictor
from training.preprocessor import tokenizer

def run_training():

    train_dataset, eval_dataset = load_dataset_pipeline()

    model = get_model()
    training_args = get_training_args()

    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=compute_metrics
    )

    save_model(trainer=trainer,tokenizer=tokenizer) 


def predict_text(text):
    model_path = "/content/drive/MyDrive/msc_research/Data/models/trained_xlm_roberta"
    predictor = Predictor(model_path)
    label, confidence = predictor.predict(text)
    print(f"Input: {text}")
    print(f"Prediction: {'Hate' if label == 1 else 'Not Hate'} ({confidence:.2f} confidence)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Romanized Sinhala Hate Speech Detection CLI")
    parser.add_argument("--mode", choices=["train", "predict", "predict-batch"], required=True,
                        help="Mode: train, predict, or predict-batch")
    parser.add_argument("--text", type=str, help="Text input for prediction")

    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "predict":
        if not args.text:
            print("Error: --text argument is required in predict mode")
        else:
            predict_text(args.text)
