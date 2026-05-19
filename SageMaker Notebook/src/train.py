"""
ASG 05 – Step 3: Model Training
Trains a Logistic Regression model using preprocessing pipeline
and saves the trained model.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train(train_data, preprocessor):

    X_train, y_train = train_data

    # Combine preprocessing + model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/model.pkl")

    print("Training selesai.")

    return pipeline