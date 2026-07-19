# DriveMind AI

**Intelligent Driver Safety, Risk Prediction & Autonomous Driver Intelligence Platform**

DriveMind AI is a modular, production-oriented AI system that monitors driver behavior and road conditions in real time using computer vision, then fuses those signals into a live, ML-driven risk assessment and intelligent alerting system.

Built as a portfolio-grade demonstration of computer vision, AI systems design, and production software engineering practices — not a single-script demo.

---

## What It Does

DriveMind AI runs a live camera pipeline that:

- Detects driver presence and tracks facial landmarks
- Computes Eye Aspect Ratio (EAR) to detect eye closure / drowsiness
- Computes Mouth Aspect Ratio (MAR) to detect yawning and fatigue buildup
- Estimates head pose (yaw/pitch/roll) to detect distraction (looking away from the road)
- Detects distraction-causing objects (phone, cup, bottle) using YOLOv8
- Detects seatbelt compliance (rule-based heuristic, ML upgrade planned)
- Detects road lane lines and estimates lane-centering from road-facing footage
- Detects vehicles and pedestrians on the road and estimates **monocular Time-To-Collision (TTC)** from bounding-box expansion rate
- Classifies driver emotion (angry, happy, neutral, fear, etc.) using DeepFace
- Fuses all of the above into a live risk decision using a **trained ML classifier (LightGBM)**, with **SHAP-based explainability** for every prediction
- Estimates a rolling **fatigue trend / probability** over the next 10 minutes from recent EAR/MAR/yawn history
- Triggers cooldown-aware alerts (with audio for critical risk) so the driver isn't spammed
- Logs every decision and alert to a database — this same logged data is what trained the ML Risk Engine

## Architecture

Camera Frame
&nbsp;&nbsp;&nbsp;→ Perception Layer (Face, Eyes, Mouth, Head Pose, Emotion, Objects, Seatbelt, Lanes, Vehicles/TTC)
&nbsp;&nbsp;&nbsp;→ Feature Aggregator (merges all module outputs into one state snapshot)
&nbsp;&nbsp;&nbsp;→ Decision Engine — rule-based v1, now upgraded to an **ML-based Risk Engine** (LightGBM, trained on logged decisions) with SHAP explainability
&nbsp;&nbsp;&nbsp;→ Fatigue Predictor (rolling-window trend estimator — LSTM upgrade planned once long continuous trip logs exist)
&nbsp;&nbsp;&nbsp;→ Alert Engine (cooldown-aware alerting, audio for critical risk)
&nbsp;&nbsp;&nbsp;→ Database (SQLite — decisions + alerts persisted per trip; this log is the training data source for the ML models)

Every perception module implements a shared `Detector` interface (see `src/drivemind/common/interfaces.py`), and ML predictors implement a shared `Predictor` interface, so any model can be swapped (e.g., the rule-based Decision Engine → the trained LightGBM Risk Engine, or the trend-based Fatigue Predictor → a future LSTM) without touching the rest of the system.

## Tech Stack

| Layer | Technology |
|---|---|
| Face & landmark detection | MediaPipe (Tasks API — BlazeFace, Face Landmarker) |
| Object & vehicle detection | YOLOv8 (Ultralytics) |
| Emotion recognition | DeepFace |
| Computer vision | OpenCV (Canny edges, Hough Transform for lanes/seatbelt) |
| ML Risk Engine | LightGBM (gradient-boosted classifier) |
| Explainability | SHAP (TreeExplainer) |
| Database / ORM | SQLite + SQLAlchemy (Repository pattern) |
| Config management | YAML-based, environment-layered |
| Language | Python 3.10 |

## Project Status

This is an actively developed, phased build. Current state:

- Done — Phase 0: Project scaffold, config system, core interfaces
- Done — Phase 1: Face Detection, Eye State (EAR), Yawning (MAR), Head Pose, Distraction (YOLO), Seatbelt (heuristic placeholder), Lane Detection, Emotion Detection
- Done — Phase 3 (partial): Vehicle/Pedestrian Detection with monocular Time-To-Collision (TTC) estimation
- Done — Phase 4: Feature Aggregator, rule-based Decision Engine, Alert Engine, database logging
- Done — Phase 5 (partial): **ML-based Risk Engine (LightGBM) trained on logged decision data, with SHAP explainability**; rolling-window Fatigue Predictor (trend-based placeholder for a future LSTM)
- In progress — Face Recognition / driver profiles, Weather Detection, Digital Twin
- Planned — Full LSTM-based Fatigue Predictor (needs long, continuous multi-driver trip logs)
- Planned — FastAPI backend, Streamlit dashboard, MLOps (MLflow/ONNX/TensorRT), Docker deployment, edge optimization

## Known Limitations (Documented Intentionally)

- **Seatbelt detection** is currently a rule-based heuristic (edge + line detection), not a trained model. Seatbelt is not a standard COCO class, so a custom-labeled dataset and training pass is planned for the MLOps phase.
- **Head pose roll angle** has a known axis-convention discrepancy in the current `solvePnP` implementation. Yaw and pitch (used for distraction classification) are verified accurate; roll is not currently used in decision logic.
- **Vehicle tracking for TTC** uses a simplified centroid-distance matcher across frames, not full multi-object tracking (e.g., ByteTrack). This works for the current single-camera test setup but won't handle occlusion or crowded scenes robustly — a planned upgrade.
- **The ML Risk Engine is trained on labels generated by the rule-based Decision Engine itself**, from a single driver's session data. This means it currently reproduces the rule-based logic rather than discovering genuinely new patterns, and the CRITICAL risk class has very few training samples (16, out of ~2100 total) — both are expected at this stage and would require multi-driver, longer-duration data collection to move past.
- **The Fatigue Predictor is a rolling-window trend estimator, not a trained LSTM.** LSTMs need long, continuous sequences (30–60+ minute real trips); the current data is short test clips. This is documented as the planned Phase 7 upgrade path, not a hidden shortcut.

## Running It Locally

Clone and enter the repo:

    git clone https://github.com/vidhiborse/drivemind-ai.git
    cd drivemind-ai

Create and activate a virtual environment (Windows PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements/base.txt

Run the full integrated pipeline demo (webcam required):

    python scripts/run_full_pipeline_demo.py

Individual modules can also be tested standalone via the other scripts in `scripts/` (e.g. `test_face_detector.py`, `test_eye_state.py`, `test_yawn_detector.py`, `test_vehicle_detector.py`, `test_emotion_detector.py`, etc.).

To retrain the ML Risk Engine on freshly logged data:

    python scripts/extract_training_data.py
    python scripts/train_risk_model.py
    python scripts/explain_risk_model.py

## Project Structure

    drivemind-ai/
    ├── configs/              YAML configuration (base + environment overrides)
    ├── src/drivemind/
    │   ├── common/           Shared interfaces (Detector, Predictor), config loader
    │   ├── perception/       Face, eyes, mouth, head_pose, emotion, distraction, seatbelt, road (lanes + vehicles/TTC)
    │   ├── cognition/        Feature Aggregator, Decision Engine, Fatigue Predictor
    │   ├── action/           Alert Engine
    │   └── database/         SQLAlchemy models + repositories
    ├── scripts/               Standalone test scripts, full pipeline demo, ML training/explainability scripts
    ├── models/                Downloaded model weights + trained risk_model.pkl (large binary weights gitignored)
    ├── data/                  Test video files + extracted training CSVs
    ├── docs/                  Architecture documentation
    └── requirements/          Dependency lists

## Why This Project

This project was built to demonstrate production-level AI engineering practices, not just model usage:

- Modular architecture with swappable components (Strategy pattern via `Detector`/`Predictor` interfaces)
- Clean separation between perception, cognition, and action layers
- Repository pattern for database access
- Explicit, documented limitations rather than hidden shortcuts
- A rule-based-first approach to ML systems: the rule-based Decision Engine was built and run first specifically to generate the labeled training data the ML Risk Engine was then trained on — rather than guessing at a model with no data
- Explainability treated as a first-class requirement (SHAP integration), not an afterthought

---

*Built by [vidhiborse](https://github.com/vidhiborse)*