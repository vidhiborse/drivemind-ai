"""
Feature Extraction — converts raw logged decisions (JSON blobs in DB) into
a clean, flat CSV suitable for training an ML model.

This is a one-time (or rerun-as-needed) offline script — not part of the
live pipeline.
"""

import sys
sys.path.insert(0, "src")

import pandas as pd
from drivemind.database.models import init_db, get_session, DecisionLog

EMOTION_ENCODING = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "sad": 4, "surprise": 5, "neutral": 6,
}


def extract_features(raw_state: dict) -> dict:
    eye = raw_state.get("eye_state_detector", {})
    yawn = raw_state.get("yawn_detector", {})
    head_pose = raw_state.get("head_pose_estimator", {})
    distraction = raw_state.get("distraction_detector", {})
    seatbelt = raw_state.get("seatbelt_detector", {})
    emotion = raw_state.get("emotion_detector", {})

    dominant_emotion = emotion.get("dominant_emotion")

    return {
        "avg_ear": eye.get("avg_ear", 0.3),          # default: eyes open
        "eye_closed": 1 if eye.get("eye_state") == "closed" else 0,
        "mar": yawn.get("mar", 0.2),                  # default: mouth normal
        "is_yawning": 1 if yawn.get("is_yawning") else 0,
        "yawn_count_2min": yawn.get("yawn_count_last_2min", 0),
        "yaw": head_pose.get("yaw", 0.0),
        "pitch": head_pose.get("pitch", 0.0),
        "looking_away": 0 if head_pose.get("direction") == "forward" else 1,
        "object_distraction": 1 if distraction.get("distraction_detected") else 0,
        "seatbelt_missing": 0 if seatbelt.get("seatbelt_detected") else 1,
        "emotion_encoded": EMOTION_ENCODING.get(dominant_emotion, 6),  # default: neutral
    }


def main():
    engine = init_db()
    session = get_session(engine)
    rows = session.query(DecisionLog).all()

    records = []
    for row in rows:
        features = extract_features(row.raw_state)
        features["risk_level"] = row.risk_level
        records.append(features)

    df = pd.DataFrame(records)
    output_path = "data/processed/training_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Extracted {len(df)} rows to {output_path}")
    print("\nClass distribution:")
    print(df["risk_level"].value_counts())
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()