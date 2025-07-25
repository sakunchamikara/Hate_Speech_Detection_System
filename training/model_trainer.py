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
            report_to="none",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=7,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            seed=42,
            warmup_steps=500,
            gradient_accumulation_steps=2,
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