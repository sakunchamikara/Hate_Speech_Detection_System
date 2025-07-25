import argparse
from training.model_trainer import ModelTrainer
from training.data_loader import DatasetLoader
from training.metrics_evaluator import MetricsEvaluator
from training.training_preprocessor import TrainingPreprocessor
from training.model_evaluator import ModelEvaluator
from romsi_hate_speech.predictor import Predictor

def run_training(adhoc=False):
    loader = DatasetLoader(adhoc=adhoc)
    train_dataset, val_dataset, final_test_dataset = loader.load()

    preprocessor = TrainingPreprocessor(adhoc=adhoc)
    metrics = MetricsEvaluator()
    trainer = ModelTrainer()

    trainer_obj = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics.compute,
        tokenizer=preprocessor.tokenizer
    )

    print("\n[INFO] HuggingFace evaluation on final test set:")
    trainer.evaluate_model(trainer_obj, eval_dataset=final_test_dataset)

    print("\n[INFO] Manual evaluation using ModelEvaluator:")
    model_path = "models/saved_model"
    predictor = Predictor(model_path)
    predictions = predictor.predict([x["text"] for x in final_test_dataset])

    y_pred = [1 if p["label"] == "hate" else 0 for p in predictions]
    y_true = final_test_dataset["label"]

    evaluator = ModelEvaluator()
    evaluator.evaluate_model(y_true, y_pred)

    trainer.save_model(trainer_obj, preprocessor.tokenizer, output_dir="models/saved_model")

def predict_text(text):
    model_path = "models/saved_model"
    predictor = Predictor(model_path)
    result = predictor.predict(text)[0]

    print(f"Input: {result['text']}")
    print(f"Prediction: {result['label'].capitalize()} ({result['confidence']:.2f} confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Romanized Sinhala Hate Speech Detection CLI")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="Mode: train or predict")
    parser.add_argument("--text", type=str, help="Text input for prediction")
    parser.add_argument("--adhoc", action="store_true", help="Enable ad hoc transliteration")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(adhoc=args.adhoc)
    elif args.mode == "predict":
        if not args.text:
            print("Error: --text argument is required in predict mode")
        else:
            predict_text(args.text)
