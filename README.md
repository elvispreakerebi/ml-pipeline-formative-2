# Formative 2: User Identity and Product Recommendation System

**Multimodal Data Preprocessing** — Machine Learning Pipeline

A sequential authentication system that verifies users via facial recognition and voice validation before providing personalized product recommendations. Access is granted only when both face and voice match the same authorized person; the predicted product is displayed only after all verification steps succeed.

---

## Team Member Contributions

| Member | Task | Contribution |
|--------|------|--------------|
| **Kumi** | Task 1 | Data merge, EDA (summary statistics, distributions, correlations, outlier detection), data cleaning (nulls, duplicates), feature engineering, product recommendation model training, `merged_dataset.csv` |
| **Josue** | Task 2 | Image collection (4 members × 3 expressions), augmentation (rotation, flip, grayscale), feature extraction (histogram + HOG), facial recognition model (Random Forest vs Logistic Regression), `image_features.csv` |
| **Bonaparte** | Task 3 | Audio collection, augmentation (pitch shift, time stretch, noise), MFCC/spectral rolloff/energy extraction, voiceprint model with StandardScaler and LabelEncoder, `audio_features.csv` |
| **Elvis Preye Kerebi** | Task 4 | System simulation CLI (`run_system.py`), integration of all three models, M4A/audioread support for voice files, flow logic (face → product → voice → display), face–voice identity matching |

---

## System Flow

The system follows a strict sequential flow:

1. **Facial Recognition** — User face must match a known member (Josue, Bonaparte, Yunis, Elvis Preye Kerebi). On failure → **Access Denied**.
2. **Product Recommendation** — Product model runs using merged customer data. Result is computed but **not displayed yet**.
3. **Voice Verification** — User voice must match an approved voiceprint and the **same person** identified by face. On failure → **Access Denied**.
4. **Display Product** — Predicted product is shown only if all steps pass.

Face and voice must identify the same person (`member_id` comparison). The product is revealed only after both verifications succeed.

---

## System Simulation Video

[![Watch the video](https://img.shields.io/badge/▶️_Watch_System_Simulation_Video-red?style=for-the-badge)](https://drive.google.com/file/d/1oEliQuXiBuuKrI9rphPyr-oONk6UpHLt/view?usp=sharing)

*[formative-2-system-simulation.mov](https://drive.google.com/file/d/1oEliQuXiBuuKrI9rphPyr-oONk6UpHLt/view?usp=sharing)* — Full authentication flow demo (face → product → voice → display)

---

## Repository Structure

```
ml-pipeline-formative-2/
├── data/
│   ├── raw/                          # Source datasets
│   │   ├── customer_transactions.csv  # 150 rows × 6 cols
│   │   └── customer_social_profiles.csv # 155 rows × 5 cols
│   ├── processed/                    # Merged and feature outputs
│   │   ├── merged_dataset.csv        # 61 customers × 23 features
│   │   ├── image_features.csv
│   │   └── audio_features.csv
│   ├── images/
│   │   ├── member1/ … member4/       # Per-member images (neutral, smiling, surprised)
│   │   ├── augmented/
│   │   └── unauthorized_face.jpg
│   └── audio/
│       └── samples/                  # Audio recordings per member (WAV, M4A)
├── models/
│   ├── product_recommendation_model.pkl
│   ├── facial_recognition_model.pkl
│   ├── voiceprint_model.joblib
│   ├── voice_scaler.joblib
│   └── voice_label_encoder.joblib
├── notebooks/
│   ├── TASK1_DATA_MERGE.ipynb        # Data merge & product model (Kumi)
│   ├── TASK2_IMAGE_PROCESSING.ipynb   # Image pipeline & face model (Josue)
│   └── TASK3_VOICE_PROCESSING.ipynb  # Audio pipeline & voice model (Bonaparte)
├── outputs/                          # Plots and visualizations
├── scripts/
│   ├── run_system.py                 # CLI system simulation (Task 4)
│   ├── feature_extractors.py         # Image histogram + HOG extraction
│   └── audio_features.py             # Audio MFCC, rolloff, energy extraction
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd ml-pipeline-formative-2
pip install -r requirements.txt
```

**Requirements:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, opencv-python, Pillow, scipy, soundfile, audioread.

**Note:** Audio processing uses scipy/numpy only (no librosa/numba) for compatibility with Python 3.13 and systems where numba fails to build (e.g. missing OpenMP on macOS).

---

## Running the Notebooks

Run notebooks from project root. Paths in notebooks may assume `data/raw/` or notebook-relative paths.

| Notebook | Purpose | Inputs |
|----------|---------|--------|
| **TASK1_DATA_MERGE.ipynb** | Merge transactions + social profiles, train product model | `data/raw/customer_*.csv` |
| **TASK2_IMAGE_PROCESSING.ipynb** | Extract image features, train face model | `data/images/member*/` |
| **TASK3_VOICE_PROCESSING.ipynb** | Extract audio features, train voiceprint model | `data/audio/samples/` |

---

## Task 4: System Simulation CLI

The CLI app simulates the full authentication flow. Run from project root.

### Full Transaction (Face + Product + Voice → Display)

```bash
python scripts/run_system.py --demo-full \
  --face-image data/images/member4/member4_neutral.jpg \
  --voice-audio data/audio/samples/Preye-REC.m4a
```

### Unauthorized Face Demo

```bash
python scripts/run_system.py --demo-unauthorized-face \
  --face-image data/images/unauthorized_face.jpg
```

### Unauthorized Voice Demo

```bash
python scripts/run_system.py --demo-unauthorized-voice \
  --voice-audio path/to/non-member-audio.wav
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--demo-full` | Run full transaction (face + product + voice) |
| `--demo-unauthorized-face` | Simulate unauthorized face attempt |
| `--demo-unauthorized-voice` | Simulate unauthorized voice attempt |
| `--face-image PATH` | Path to face image |
| `--voice-audio PATH` | Path to voice audio (WAV, M4A, etc.) |
| `--face-threshold N` | Face confidence threshold 0–1 (default: 0.45) |
| `--verbose` | Show confidence scores and model details |

---

## Model Summary

| Model | Accuracy | F1-Score (weighted) | Notes |
|-------|----------|---------------------|-------|
| **Product Recommendation** | 0.385 | 0.354 | RandomForestClassifier; 61 samples, 5 classes |
| **Facial Recognition** | 1.0000 | 1.0000 | Random Forest; histogram + HOG features |
| **Voiceprint** | 0.8750 | 0.8667 | Random Forest + StandardScaler; MFCC, rolloff, energy |

---

## Task 1: Data Merge & Product Model (Kumi)

- **Sources:** 150 transactions, 155 social profiles
- **Cleaning:** 10 missing ratings → median fill; 5 duplicates removed from social
- **Merge:** Inner join on `customer_id` → 61 customers
- **Features:** Transaction aggregations (avg_purchase_amount, total_spent, avg_rating, most_purchased_category), social aggregations (avg_engagement_score, primary_platform, dominant_sentiment), derived (engagement_x_interest, spending_per_transaction, sentiment_encoded, platform one-hot)
- **Target:** `most_purchased_category` — Electronics (18), Books (14), Clothing (14), Sports (8), Groceries (7)
- **Output:** `merged_dataset.csv`, `product_recommendation_model.pkl`

---

## Task 2: Image Processing & Face Model (Josue)

- **Images:** 4 members × 3 expressions = 12 base; augmented (rotation ±15°, flip, grayscale)
- **Features:** Color histogram (96) + HOG (1764) = 1860 per image
- **Model:** Random Forest (best of RF vs Logistic Regression)
- **Output:** `image_features.csv`, `facial_recognition_model.pkl`

---

## Task 3: Voice Processing & Voiceprint Model (Bonaparte)

- **Audio:** WAV/M4A from `data/audio/samples/`; labels from filename (e.g. `Preye-REC.m4a`)
- **Features:** 30 total — MFCCs (26), spectral rolloff (2), RMS energy (2)
- **Model:** RandomForestClassifier + StandardScaler + LabelEncoder
- **Output:** `audio_features.csv`, `voiceprint_model.joblib`, `voice_scaler.joblib`, `voice_label_encoder.joblib`

---

## Task 4: System Integration (Elvis Preye Kerebi)

- **Script:** `scripts/run_system.py`
- **Flow:** Load models → verify face → run product model → verify voice → check face–voice match → display product
- **Audio:** `soundfile` for WAV/FLAC/OGG; `audioread` for M4A/AAC/MP4/MP3
- **Multimodal:** Face and voice must identify the same person (member_id)

---

## Technical Notes

- **Python 3.13:** Audio feature extraction uses scipy/numpy only for compatibility where OpenMP fails.
- **M4A Support:** `audioread` for M4A, AAC, MP4, MP3; `soundfile` for WAV, FLAC, OGG.
- **Face–Voice Matching:** System requires face and voice to identify the same person.

---

## GitHub

**Repository:** [https://github.com/elvispreakerebi/ml-pipeline-formative-2](https://github.com/elvispreakerebi/ml-pipeline-formative-2)
