import argparse
from training.model_trainer import ModelTrainer
from training.data_loader import DatasetLoader
from training.metrics_evaluator import MetricsEvaluator
from training.training_preprocessor import TrainingPreprocessor
from romsi_hate_speech.predictor import Predictor

def run_training(adhoc=False):
    # Load and preprocess data
    loader = DatasetLoader(adhoc=adhoc)
    train_dataset, eval_dataset = loader.load()

    # Initialize training components
    preprocessor = TrainingPreprocessor(adhoc=adhoc)
    metrics = MetricsEvaluator()
    trainer = ModelTrainer()

    # Train model
    trainer_obj = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics.compute,
        tokenizer=preprocessor.tokenizer
    )

    # Evaluate and save
    trainer.evaluate(trainer_obj)
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
