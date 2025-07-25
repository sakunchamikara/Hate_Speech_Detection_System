from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, target_names=None):
        if target_names is None:
            self.target_names = ["not_hate", "hate"]
        else:
            self.target_names = target_names

    def evaluate(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.target_names, output_dict=False)
        matrix = confusion_matrix(y_true, y_pred)

        print("Classification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(matrix)

        return {
            "report": report,
            "confusion_matrix": matrix
        }