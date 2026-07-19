"""
Explainability layer using SHAP — shows WHY the model made a given prediction,
not just what it predicted. This satisfies the "Explainable AI" requirement:
every decision should include reason, confidence, and evidence.
"""

import pandas as pd
import joblib
import shap

MODEL_PATH = "models/risk_model.pkl"
DATA_PATH = "data/processed/training_data.csv"

FEATURE_COLUMNS = [
    "avg_ear", "eye_closed", "mar", "is_yawning", "yawn_count_2min",
    "yaw", "pitch", "looking_away", "object_distraction",
    "seatbelt_missing", "emotion_encoded",
]


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]

    explainer = shap.TreeExplainer(model)

    # Explain one example prediction in detail
    sample = X.iloc[[0]]
    prediction = model.predict(sample)[0]
    shap_values = explainer.shap_values(sample)

    print(f"Sample features:\n{sample.to_string(index=False)}")
    print(f"\nPredicted risk level: {prediction}")
    print("\nSHAP values show how much each feature pushed the prediction")
    print("toward or away from each class. Larger absolute value = more influence.\n")

    classes = model.classes_
    for i, class_name in enumerate(classes):
        print(f"--- Influence on class '{class_name}' ---")
        values = shap_values[0][:, i] if len(shap_values[0].shape) > 1 else shap_values[i][0]
        contributions = pd.Series(values, index=FEATURE_COLUMNS).sort_values(
            key=abs, ascending=False
        )
        print(contributions.head(3))
        print()


if __name__ == "__main__":
    main()