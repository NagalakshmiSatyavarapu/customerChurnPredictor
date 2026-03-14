# src/predict.py

import sys
import joblib
import pandas as pd

def predict(file_path):

    model = joblib.load("models/model.pkl")

    df = pd.read_csv(file_path)

    predictions = model.predict(df)

    df["Prediction"] = predictions
    df.to_csv("outputs/predictions.csv", index=False)

    print("Predictions saved to outputs/predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <dataset_path>")
    else:
        predict(sys.argv[1])