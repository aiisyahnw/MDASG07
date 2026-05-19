"""
ASG 05 – Step 2: Data Preprocessing
Reads ingested dataset, performs data cleaning, feature engineering,
and splits the data into training and testing sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer #gabung pipeline numerik & kategorikal
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path


def winsorize_features(X):
    X = pd.DataFrame(X).copy()
    upper_limit = X.quantile(0.95)
    return X.clip(upper=upper_limit, axis=1)


def preprocess():
    os.makedirs("artifacts", exist_ok=True)

    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "ingested" / "spaceship_train.csv"
    df = pd.read_csv(DATA_PATH)

    # Drop irrelevant columns (from EDA)
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin'], errors='ignore')

    # Remove duplicates (from EDA)
    df = df.drop_duplicates()

    # Feature groups
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

    # Split features & target
    X = df[num_cols + cat_cols]
    y = df['Transported'].astype(int)

    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorize', FunctionTransformer(winsorize_features)),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_test, y_test), preprocessor


if __name__ == "__main__":
    preprocess()