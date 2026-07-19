"""
Trains a LightGBM gradient-boosted classifier to predict risk_level from
extracted features. This replaces the rule-based Decision Engine's risk
scoring, using real logged data as training labels.

Uses a driver-held-out-style split isn't applicable yet (single driver so
far), so a standard train/test split is used for this first version.
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH = "models/risk_model.pkl"

FEATURE_COLUMNS = [
    "avg_ear", "eye_closed", "mar", "is_yawning", "yawn_count_2min",
    "yaw", "pitch", "looking_away", "object_distraction",
    "seatbelt_missing", "emotion_encoded",
]

LABEL_COLUMN = "risk_level"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n--- Confusion Matrix ---")
    print("Labels order:", sorted(y.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    print("\n--- Feature Importance ---")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importance.sort_values(ascending=False))

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()