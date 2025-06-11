from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_predictions(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["not_hate", "hate"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))