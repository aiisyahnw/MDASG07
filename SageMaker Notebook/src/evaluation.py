"""
ASG 05 – Step 4: Model Evaluation
Evaluates the trained model using classification metrics.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(test_data, pipeline):

    X_test, y_test = test_data

    predictions = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec  = recall_score(y_test, predictions)
    f1   = f1_score(y_test, predictions)

    print("-" * 30)
    print("Evaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}  <-- main metric")
    print("-" * 30)

    return acc, prec, rec, f1


if __name__ == "__main__":
    print("Run evaluation from pipeline")

#F1-score digunakan sebagai metrik utama karena
#mampu menyeimbangkan precision dan recall pada data yang sedikit tidak seimbang.