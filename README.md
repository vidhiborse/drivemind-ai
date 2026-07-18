# DriveMind AI

**Intelligent Driver Safety, Risk Prediction & Autonomous Driver Intelligence Platform**

DriveMind AI is a modular, production-oriented AI system that monitors driver behavior and road conditions in real time using computer vision, then fuses those signals into a live risk assessment and intelligent alerting system.

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
- Fuses all of the above into a single, explainable risk decision (LOW / MEDIUM / HIGH / CRITICAL)
- Triggers cooldown-aware alerts (with audio for critical risk) so the driver isn't spammed
- Logs every decision and alert to a database for later analysis and future ML training

## Architecture

Camera Frame
→ Perception Layer (Face, Eyes, Mouth, Head Pose, Objects, Seatbelt, Lanes)
→ Feature Aggregator (merges all module outputs into one state snapshot)
→ Decision Engine (rule-based risk fusion — ML upgrade planned)
→ Alert Engine (cooldown-aware alerting, audio for critical risk)
→ Database (SQLite — decisions + alerts persisted per trip)

Every perception module implements a shared `Detector` interface (see `src/drivemind/common/interfaces.py`), so any model can be swapped (e.g., YOLOv8 → YOLOv11, or a rule-based seatbelt check → a trained classifier) without touching the rest of the system.

## Tech Stack

| Layer | Technology |
|---|---|
| Face & landmark detection | MediaPipe (Tasks API — BlazeFace, Face Landmarker) |
| Object detection | YOLOv8 (Ultralytics) |
| Computer vision | OpenCV (Canny edges, Hough Transform for lanes/seatbelt) |
| Database / ORM | SQLite + SQLAlchemy (Repository pattern) |
| Config management | YAML-based, environment-layered |
| Language | Python 3.10 |

## Project Status

This is an actively developed, phased build. Current state:

- ✅ **Phase 0** — Project scaffold, config system, core interfaces
- ✅ **Phase 1 (partial)** — Face Detection, Eye State (EAR), Yawning (MAR), Head Pose, Distraction (YOLO), Seatbelt (heuristic placeholder), Lane Detection
- ✅ **Phase 4 (partial)** — Feature Aggregator, rule-based Decision Engine, Alert Engine, database logging
- ⏳ Emotion Detection, Face Recognition, Vehicle/Pedestrian Detection, Weather Detection — in progress
- ⏳ ML-based Risk Engine and Fatigue Predictor — planned once sufficient logged trip data is available
- ⏳ FastAPI backend, Streamlit dashboard, MLOps (MLflow/ONNX/TensorRT), Docker deployment, edge optimization — planned

## Known Limitations (Documented Intentionally)

- **Seatbelt detection** is currently a rule-based heuristic (edge + line detection), not a trained model. Seatbelt is not a standard COCO class, so a custom-labeled dataset and training pass is planned for the MLOps phase.
- **Head pose roll angle** has a known axis-convention discrepancy in the current `solvePnP` implementation. Yaw and pitch (used for distraction classification) are verified accurate; roll is not currently used in decision logic.
- **Risk scoring is currently rule-based**, not ML-based. This is intentional — the rule-based system is what generates the labeled training data the ML Risk Engine will later be trained on.

## Running It Locally

```bash
# 1. Clone and enter the repo
git clone https://github.com/vidhiborse/drivemind-ai.git
cd drivemind-ai

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

# 3. Install dependencies
pip install -r requirements/base.txt

# 4. Run the full integrated pipeline demo (webcam required)
python scripts/run_full_pipeline_demo.py
```

Individual modules can also be tested standalone via the other scripts in `scripts/` (e.g. `test_face_detector.py`, `test_eye_state.py`, `test_yawn_detector.py`, etc.).

## Project Structure

drivemind-ai/
├── configs/                  # YAML configuration (base + environment overrides)
├── src/drivemind/
│   ├── common/                # Shared interfaces, config loader
│   ├── perception/             # Face, eyes, mouth, head_pose, distraction, seatbelt, road
│   ├── cognition/               # Feature Aggregator, Decision Engine
│   ├── action/                  # Alert Engine
│   └── database/                 # SQLAlchemy models + repositories
├── scripts/                  # Standalone test scripts + full pipeline demo
├── models/                   # Downloaded model weights (gitignored except small .tflite files)
├── data/                     # Test video/data files
├── docs/                     # Architecture documentation
└── requirements/              # Dependency lists

## Why This Project

This project was built to demonstrate production-level AI engineering practices, not just model usage:

- Modular architecture with swappable components (Strategy pattern via `Detector`/`Predictor` interfaces)
- Clean separation between perception, cognition, and action layers
- Repository pattern for database access
- Explicit, documented limitations rather than hidden shortcuts
- A rule-based-first approach to ML systems — collecting real labeled data before training predictive models, rather than guessing at a model with no data

---

*Built by [vidhiborse](https://github.com/vidhiborse)*