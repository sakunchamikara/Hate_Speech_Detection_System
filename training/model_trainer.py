from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from training.config import Config

class ModelTrainer:
    def __init__(self, model_name=Config.MODEL_NAME, num_labels=Config.NUM_LABELS):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def get_training_args(self, output_dir="results"):
        return TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            report_to="none",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

    def train(self, train_dataset, eval_dataset, compute_metrics, tokenizer=None, output_dir="results"):
        args = self.get_training_args(output_dir)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        trainer.train()
        return trainer

    def evaluate(self, trainer):
        results = trainer.evaluate()
        print("Final evaluation metrics:", results)
        return results

    def save_model(self, trainer, tokenizer, output_dir):
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)