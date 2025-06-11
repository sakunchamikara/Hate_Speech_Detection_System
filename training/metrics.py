
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }
