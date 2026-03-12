# Formative 2: User Identity and Product Recommendation System

**Multimodal Data Preprocessing** — A sequential authentication system that verifies users via facial recognition and voice validation before providing personalized product recommendations.

## System Flow

1. **Facial Recognition** → User face must match a known member
2. **Product Recommendation** → Predict product based on merged customer data
3. **Voice Verification** → User voice must match approved voiceprint
4. **Display Product** → Show predicted product only if all steps pass

Access is **denied** if any authentication step fails.

---

## Repository Structure

```
ml-pipeline-formative-2/
├── data/
│   ├── raw/                    # customer_social_profiles.csv, customer_transactions.csv
│   ├── processed/              # merged_dataset.csv, image_features.csv, audio_features.csv
│   ├── images/                 # Per-member images + augmented/
│   │   ├── member1/ ... member4/
│   │   ├── augmented/
│   │   └── unauthorized_face.jpg
│   └── audio/
│       └── samples/            # Audio recordings per member
├── models/
│   ├── product_recommendation_model.pkl
│   ├── facial_recognition_model.pkl
│   ├── voiceprint_model.joblib
│   ├── voice_scaler.joblib
│   └── voice_label_encoder.joblib
├── notebooks/
│   ├── TASK1_DATA_MERGE.ipynb       # Data merge & product model (Kumi)
│   ├── TASK2_IMAGE_PROCESSING.ipynb  # Image pipeline & face model (Josue)
│   └── TASK3_VOICE_PROCESSING.ipynb # Audio pipeline & voice model (Bonaparte)
├── outputs/                    # Plots and visualizations
├── scripts/
│   └── run_system.py           # CLI system simulation (Preye - Task 4)
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
└── formative2-task-breakdown.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Note:** Audio processing uses scipy/numpy (no librosa/numba) for compatibility with Python 3.13 and systems where numba fails to build (e.g. missing OpenMP on macOS).

---

## Running the Notebooks

1. **Task 1 — Data Merge & Product Model**  
   Open `notebooks/TASK1_DATA_MERGE.ipynb` and run all cells. Expects `data/raw/customer_*.csv`.

2. **Task 2 — Image Processing & Face Model**  
   Open `notebooks/TASK2_IMAGE_PROCESSING.ipynb` and run all cells. Expects `data/images/member*/`.

3. **Task 3 — Voice Processing & Voiceprint Model**  
   Open `notebooks/TASK3_VOICE_PROCESSING.ipynb` and run all cells. Expects `data/audio/samples/`.

4. **Task 4 — System Simulation**  
   Run the CLI app: `python scripts/run_system.py` (see [Task 4](#task-4-system-simulation) below).

---

## Task 4: System Simulation

The CLI app simulates the full authentication flow. Run from project root.

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Transaction (Face → Product → Voice → Display)

**Live capture (webcam + microphone):**
```bash
python scripts/run_system.py --demo-full --live
```
Camera and microphone will activate. Press SPACE to capture your face, then speak when prompted for voice verification.

**File-based (pre-recorded image and audio):**
```bash
python scripts/run_system.py --demo-full \
  --face-image data/images/member1/member1_neutral.jpg \
  --voice-audio data/audio/samples/Preye-REC.m4a
```

### Unauthorized Face Demo

```bash
# Live: capture from webcam
python scripts/run_system.py --demo-unauthorized-face --live

# File-based
python scripts/run_system.py --demo-unauthorized-face \
  --face-image data/images/unauthorized_face.jpg
```

### Unauthorized Voice Demo

```bash
# Live: record from microphone
python scripts/run_system.py --demo-unauthorized-voice --live

# File-based: use audio from a non-member
python scripts/run_system.py --demo-unauthorized-voice \
  --voice-audio path/to/non-member-audio.wav
```

### Options

| Flag | Description |
|------|-------------|
| `--demo-full` | Run full transaction (face + product + voice) |
| `--demo-unauthorized-face` | Simulate unauthorized face attempt |
| `--demo-unauthorized-voice` | Simulate unauthorized voice attempt |
| `--live` | Capture from webcam and microphone instead of files |
| `--face-image PATH` | Path to face image (omit with --live) |
| `--voice-audio PATH` | Path to voice audio file (omit with --live) |
| `--record-duration N` | Seconds to record voice when using --live (default: 3) |
| `--verbose` | Show confidence scores and model details |

---

## Team Contributions

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for per-member task breakdown and contributions.
