from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import numpy as np

def evaluate_model(y_true, preds, problem_type):

    if len(y_true) != len(preds):
        raise ValueError("y_true and preds length mismatch")

    if problem_type == "classification":

        return {
            "Accuracy": accuracy_score(y_true, preds),
            "F1 Score": f1_score(y_true, preds, average="weighted"),
            "Precision": precision_score(y_true, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, preds, average="weighted", zero_division=0)
        }

    else:

        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, preds)),
            "MAE": mean_absolute_error(y_true, preds),
            "R2 Score": r2_score(y_true, preds)
        }