from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from training.config import Config

def get_training_args(output_dir="results"):
    return TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
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

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=Config.NUM_LABELS)

def train_model(model, train_dataset, eval_dataset, training_args, compute_metrics, tokenizer, callbacks=[]):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=callbacks
    )
    trainer.train()
    results = trainer.evaluate()
    print("Final evaluation metrics:", results)
    return trainer

def save_model(trainer,tokenizer,output_dir="models/trained_xlm_roberta"):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)