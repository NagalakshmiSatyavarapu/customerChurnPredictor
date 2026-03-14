# src/train.py

import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.pipeline import build_pipeline

def train(file_path, target_column):

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():

        pipeline = build_pipeline(X, model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted")

        print(f"{name} → F1 Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, "models/model.pkl")

    print("\n🏆 Best Model:", best_name)
    print("Model saved successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.train <dataset_path> <target_column>")
    else:
        train(sys.argv[1], sys.argv[2])