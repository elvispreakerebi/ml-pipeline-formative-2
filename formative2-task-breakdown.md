# Formative 2: Multimodal Data Preprocessing — Task Breakdown

## Assignment Summary

This assignment builds a **User Identity and Product Recommendation System** with sequential authentication:
1. **Facial Recognition** → 2. **Product Recommendation** → 3. **Voice Verification** → 4. **Display Product**

Access is denied if any authentication step fails. The system uses pre-trained models to verify face and voice before allowing product predictions.

---

## Team Task Allocation (4 Members)

| Member | Primary Task | Dependencies |
|--------|--------------|--------------|
| **Person 1** | Data Merge & Product Recommendation Model | None (starts first) |
| **Person 2** | Image Data Collection, Processing & Facial Recognition Model | None (parallel with P1, P3) |
| **Person 3** | Sound Data Collection, Processing & Voiceprint Model | None (parallel with P1, P2) |
| **Person 4** | System Integration & CLI Demonstration | All outputs from P1, P2, P3 |

---

## Person 1: Data Merge & Product Recommendation Model

**Deliverables:** Merged dataset, EDA notebook section, `merged_dataset.csv`, Product Recommendation Model (trained), evaluation metrics.

### Step 1: Project Setup & Dataset Acquisition
**Commit:** `feat(data): add project structure and raw datasets`

- Create `data/raw/` directory
- Download `customer_social_profiles` and `customer_transactions` datasets
- Add `requirements.txt` with: pandas, numpy, scikit-learn, matplotlib, seaborn
- Create `notebooks/` folder for Jupyter work
- Document dataset sources and column descriptions in a README

### Step 2: Load & Explore customer_social_profiles
**Commit:** `feat(eda): load and inspect customer_social_profiles dataset`

- Load the social profiles CSV
- Print shape, dtypes, first 5 rows
- Identify key columns (e.g., customer_id, demographics, engagement metrics)
- Note any obvious data quality issues
- Add a notebook cell with initial observations

### Step 3: Load & Explore customer_transactions
**Commit:** `feat(eda): load and inspect customer_transactions dataset`

- Load the transactions CSV
- Print shape, dtypes, first 5 rows
- Identify key columns (e.g., customer_id, product purchased, transaction details)
- Note overlap with social profiles (join keys)
- Document which columns will be used for the product prediction target

### Step 4: Exploratory Data Analysis — Summary Statistics & Plots
**Commit:** `feat(eda): add summary statistics and ≥3 labeled visualizations`

- Compute summary statistics (mean, median, std, min, max) for numeric columns
- Create **≥3 labeled plots** as per rubric:
  - Distribution plots (histograms/KDE) for key variables
  - Outlier detection (boxplots or IQR)
  - Correlation heatmap between numeric features
- Add clear titles, axis labels, and brief interpretations
- Document variable types and any skewed distributions

### Step 5: Data Cleaning — Nulls, Duplicates & Types
**Commit:** `fix(data): handle nulls, duplicates, and fix column types`

- Count nulls per column; decide on strategy (drop, impute, or fill)
- Remove or document duplicate rows
- Fix column types (e.g., dates, categorical encoding)
- Validate no data loss beyond justified removals
- Add a `data_cleaning.py` or notebook section documenting each decision

### Step 6: Implement Merge Logic with Justification
**Commit:** `feat(data): implement merge logic with join justification`

- Choose join type (inner/left/outer) and key(s) (e.g., `customer_id`)
- Justify the choice in comments (e.g., "Inner join to keep only customers with both profile and transaction data")
- Perform the merge
- Log row counts before and after merge
- Save intermediate cleaned datasets for reproducibility

### Step 7: Post-Merge Validation
**Commit:** `feat(data): add post-merge validation checks`

- Verify no unexpected row explosion or loss
- Check for new nulls introduced by merge
- Validate referential integrity (e.g., all product IDs exist)
- Add assertions or checks that fail if merge is invalid
- Document validation results in notebook

### Step 8: Feature Engineering for Product Prediction
**Commit:** `feat(data): engineer features for product recommendation`

- Create derived features (e.g., engagement score, recency, frequency)
- Encode categorical variables for ML (OneHot, LabelEncoder)
- Define target variable (product to be predicted)
- Split features vs. target; optionally create train/test split
- Document feature definitions in a `FEATURES.md` or notebook section

### Step 9: Build & Train Product Recommendation Model
**Commit:** `feat(model): implement Product Recommendation Model`

- Use Random Forest, Logistic Regression, or XGBoost (per assignment)
- Train on merged dataset
- Implement proper train/validation split
- Save model artifact (e.g., `models/product_recommendation_model.pkl`)
- Add `scripts/train_product_model.py` or equivalent

### Step 10: Evaluate Product Model & Save Merged Dataset
**Commit:** `feat(model): evaluate product model and save merged dataset`

- Compute **Accuracy, F1-Score, and Loss** (or appropriate metric)
- Add confusion matrix and classification report
- Save final `merged_dataset.csv` (or `merged_dataset_with_features.csv`) to `data/processed/`
- Document evaluation results in notebook
- Update README with dataset location and column descriptions

---

## Person 2: Image Data Collection, Processing & Facial Recognition Model

**Deliverables:** Image collection per member (3 expressions each), `image_features.csv`, Facial Recognition Model, evaluation metrics.

### Step 1: Image Collection Structure & Submission Template
**Commit:** `feat(images): add image collection structure and submission template`

- Create `data/images/` with subfolders per member (e.g., `member1/`, `member2/`, …)
- Add `README.md` with instructions: each member submits 3 images (neutral, smiling, surprised)
- Define naming convention: `member{N}_{expression}.jpg` (e.g., `member1_neutral.jpg`)
- Add placeholder or sample images for testing
- Document expected format (e.g., JPG, min resolution)

### Step 2: Load & Display Sample Images
**Commit:** `feat(images): load and display sample images for each member`

- Write `scripts/load_images.py` or notebook section
- Load at least one image per member
- Display using matplotlib (grid layout)
- Add labels (member name, expression)
- Ensure all 4 members’ images are represented (or document who has submitted)

### Step 3: Implement Rotation Augmentation
**Commit:** `feat(images): add rotation augmentation`

- Use OpenCV or PIL to rotate images (e.g., ±15°, ±30°)
- Apply rotation to sample images and display before/after
- Save augmented versions with naming: `{original_name}_rotated_{angle}.jpg`
- Document parameters (angle range, interpolation method)

### Step 4: Implement Flipping Augmentation
**Commit:** `feat(images): add horizontal/vertical flip augmentation`

- Implement horizontal flip (mirror) and optionally vertical flip
- Apply to sample images; display results
- Save with naming: `{original_name}_flipped.jpg`
- Ensure at least 2 augmentations per image (rotation + flip) for rubric

### Step 5: Implement Grayscale Augmentation
**Commit:** `feat(images): add grayscale augmentation`

- Convert images to grayscale
- Display grayscale versions
- Save as `{original_name}_grayscale.jpg`
- Document use case (e.g., robustness to lighting)

### Step 6: Apply Augmentation Pipeline to All Images
**Commit:** `feat(images): apply full augmentation pipeline per image`

- For each member’s 3 images, apply ≥2 augmentations each (e.g., rotation + flip)
- Create `scripts/augment_images.py` with configurable pipeline
- Save all augmented images to `data/images/augmented/`
- Log count of original vs. augmented images

### Step 7: Extract Histogram Features
**Commit:** `feat(images): extract histogram features from images`

- Compute color histograms (RGB or grayscale) per image
- Flatten into feature vector
- Store in DataFrame with columns: `image_id`, `member_id`, `expression`, `hist_*`
- Document histogram parameters (bins, range)

### Step 8: Extract Embedding or Additional Features
**Commit:** `feat(images): extract embedding or additional image features`

- Extract features (e.g., embeddings from a pre-trained CNN, or HOG, or SIFT)
- Alternatively, use a simple embedding (e.g., flattened small resized image)
- Add columns to feature DataFrame
- Ensure features are numeric and suitable for ML

### Step 9: Save image_features.csv
**Commit:** `feat(images): save image features to image_features.csv`

- Combine histogram + embedding features into final DataFrame
- Include metadata: `member_id`, `expression`, `image_path`, `is_augmented`
- Save to `data/processed/image_features.csv`
- Validate file loads correctly and has expected columns

### Step 10: Build & Evaluate Facial Recognition Model
**Commit:** `feat(model): implement and evaluate Facial Recognition Model`

- Use Random Forest, Logistic Regression, or XGBoost
- Train to predict `member_id` (or binary: authorized vs. unauthorized)
- Evaluate with **Accuracy, F1-Score, Loss**
- Save model to `models/facial_recognition_model.pkl`
- Document how "unauthorized" will be simulated (unknown face = different member or synthetic)

---

## Person 3: Sound Data Collection, Processing & Voiceprint Model

**Deliverables:** Audio samples per member (2 phrases each), `audio_features.csv`, Voiceprint Verification Model, evaluation metrics.

### Step 1: Audio Collection Structure & Recording Instructions
**Commit:** `feat(audio): add audio collection structure and recording instructions`

- Create `data/audio/` with subfolders per member
- Document required phrases: "Yes, approve" and "Confirm transaction"
- Define naming: `member{N}_{phrase_slug}.wav` (e.g., `member1_yes_approve.wav`)
- Add instructions (sample rate, format, duration, quiet environment)
- Create placeholder or use sample recordings for testing

### Step 2: Load & Display Waveforms
**Commit:** `feat(audio): load and display waveforms for each member`

- Use `librosa` or `scipy` to load audio files
- Plot waveform (amplitude vs. time) for each member’s samples
- Use subplots with clear labels (member, phrase)
- Save plot to `outputs/waveforms.png` or display in notebook

### Step 3: Display Spectrograms
**Commit:** `feat(audio): display spectrograms for each member`

- Compute and plot spectrograms (time vs. frequency)
- Use `librosa.display.specshow` or matplotlib
- Add colorbar and labels
- Interpret: identify voice characteristics, noise, silence
- Save to `outputs/spectrograms.png`

### Step 4: Implement Pitch Shift Augmentation
**Commit:** `feat(audio): add pitch shift augmentation`

- Use `librosa.effects.pitch_shift` or similar
- Apply to sample (e.g., ±2 semitones)
- Display waveform/spectrogram before and after
- Save augmented file with `_pitch_shifted` suffix

### Step 5: Implement Time Stretch Augmentation
**Commit:** `feat(audio): add time stretch augmentation`

- Use `librosa.effects.time_stretch`
- Apply rate change (e.g., 0.9x, 1.1x)
- Display and save with `_time_stretched` suffix
- Ensure ≥2 augmentations per sample (pitch + time stretch)

### Step 6: Add Background Noise Augmentation
**Commit:** `feat(audio): add background noise augmentation`

- Mix audio with low-level noise (Gaussian or recorded room noise)
- Control SNR (e.g., 20 dB)
- Save with `_noisy` suffix
- Document noise source and parameters

### Step 7: Apply Augmentation Pipeline to All Audio
**Commit:** `feat(audio): apply full augmentation pipeline per sample`

- For each member’s 2 phrases, apply ≥2 augmentations each
- Create `scripts/augment_audio.py`
- Save to `data/audio/augmented/`
- Log counts

### Step 8: Extract MFCC Features
**Commit:** `feat(audio): extract MFCC features`

- Compute MFCCs using `librosa.feature.mfcc`
- Use standard params (n_mfcc=13, n_fft=2048, hop_length=512)
- Aggregate (mean, std) across time to get fixed-length vector
- Store in DataFrame with `member_id`, `phrase`, `mfcc_*`

### Step 9: Extract Spectral Roll-off & Energy
**Commit:** `feat(audio): extract spectral roll-off and energy features`

- Compute spectral roll-off: `librosa.feature.spectral_rolloff`
- Compute RMS energy: `librosa.feature.rms`
- Aggregate (mean, std) per sample
- Add columns to feature DataFrame
- Per rubric: MFCCs + roll-off/energy required

### Step 10: Save audio_features.csv
**Commit:** `feat(audio): save audio features to audio_features.csv`

- Combine MFCC, roll-off, energy into final DataFrame
- Include metadata: `member_id`, `phrase`, `file_path`, `is_augmented`
- Save to `data/processed/audio_features.csv`
- Validate structure and column types

### Step 11: Build & Evaluate Voiceprint Verification Model
**Commit:** `feat(model): implement and evaluate Voiceprint Verification Model`

- Train model to predict `member_id` or binary (authorized/unauthorized)
- Use Random Forest, Logistic Regression, or XGBoost
- Evaluate with **Accuracy, F1-Score, Loss**
- Save to `models/voiceprint_model.pkl`
- Document how "unauthorized" voice will be simulated

---

## Person 4: System Integration & CLI Demonstration

**Deliverables:** Integrated CLI app, full transaction simulation, unauthorized attempt simulation, polished interaction.

### Step 1: CLI App Structure
**Commit:** `feat(cli): add CLI app structure and argument parsing`

- Create `scripts/run_system.py` or `app/cli.py`
- Use `argparse` or `click` for CLI
- Define commands: `--demo-full`, `--demo-unauthorized-face`, `--demo-unauthorized-voice`
- Add `--help` and clear usage message
- Create `requirements.txt` entry for CLI dependencies

### Step 2: Load Face Model
**Commit:** `feat(cli): integrate facial recognition model loading`

- Load `models/facial_recognition_model.pkl`
- Add function to predict from image path or image features
- Handle missing model file with clear error
- Test with sample image path

### Step 3: Load Voice Model
**Commit:** `feat(cli): integrate voiceprint model loading`

- Load `models/voiceprint_model.pkl`
- Add function to predict from audio path or audio features
- Handle missing model file
- Test with sample audio path

### Step 4: Load Product Model
**Commit:** `feat(cli): integrate product recommendation model loading`

- Load `models/product_recommendation_model.pkl`
- Add function to predict product from customer features
- Define interface: input = customer vector, output = predicted product
- Test with sample customer data

### Step 5: Implement Face Recognition Step
**Commit:** `feat(cli): implement face recognition as first gate`

- Accept image path as input
- Extract features (reuse Person 2’s feature extraction)
- Run face model prediction
- Return `authorized` or `denied` with clear message
- Print "Access Denied" if face fails (per flowchart)

### Step 6: Implement Product Prediction Step
**Commit:** `feat(cli): implement product prediction after face auth`

- Only run if face recognition succeeds
- Accept customer features (from merged dataset or mock)
- Call product model
- Return predicted product name/ID
- Do not display yet—wait for voice confirmation

### Step 7: Implement Voice Verification Step
**Commit:** `feat(cli): implement voice verification as second gate`

- Accept audio path as input
- Extract features (reuse Person 3’s extraction)
- Run voiceprint model
- Return `authorized` or `denied`
- Print "Access Denied" if voice fails
- Only then allow display of product

### Step 8: Implement Full Transaction Flow
**Commit:** `feat(cli): wire full transaction flow`

- Sequence: Face → Product → Voice → Display
- Input: face image path, customer features, voice audio path
- Flow matches flowchart exactly
- On success: print "Predicted Product: {product}"
- On any failure: print "Access Denied" and exit
- Add clear step-by-step console output

### Step 9: Unauthorized Face Attempt Simulation
**Commit:** `feat(cli): add unauthorized face attempt demo`

- Use an image of an **unknown person** (not in training set)
- Run through flow; face model should deny
- Print "Access Denied" at face step
- Document in README: "Use `--demo-unauthorized-face` with path to unknown face image"

### Step 10: Unauthorized Voice Attempt Simulation
**Commit:** `feat(cli): add unauthorized voice attempt demo`

- Use audio from an **unknown person** (not in training set)
- Run through flow; if face passes, voice should deny
- Print "Access Denied" at voice step
- Document: "Use `--demo-unauthorized-voice` with path to unknown voice sample"

### Step 11: Polish CLI & Error Handling
**Commit:** `fix(cli): polish interaction and error handling`

- Add friendly prompts and formatted output
- Handle file-not-found, invalid paths
- Add `--verbose` for debugging
- Ensure smooth interaction per rubric ("smooth interaction")

### Step 12: Documentation & Run Instructions
**Commit:** `docs: add run instructions and contribution summary`

- Update main README with:
  - How to run full transaction
  - How to run unauthorized demos
  - Prerequisites and setup
- Add `CONTRIBUTIONS.md` with each member’s tasks and commits
- Ensure all deliverables are listed (datasets, scripts, notebook, report)

---

## Rubric Checklist (Quick Reference)

| Criterion | Key Requirements |
|-----------|------------------|
| **EDA Quality** | Summary stats, variable types, ≥3 labeled plots (dist, outliers, correlations) |
| **Data Cleaning & Merge** | Nulls/duplicates handled, join logic justified, post-merge checks |
| **Image Quantity** | Each member: 3 expressions (neutral, smile, surprised) |
| **Image Augmentation & Features** | ≥2 augmentations per image; embeddings/histograms in `image_features.csv` |
| **Audio Quality & Viz** | 2 phrases per member; waveform and spectrogram plotted |
| **Audio Augmentation & Features** | ≥2 augmentations; MFCCs + roll-off/energy in `audio_features.csv` |
| **Model Implementation** | All 3 models (face, voice, product) implemented and functional |
| **Evaluation** | Accuracy, F1, Loss per model; multimodal logic explained |
| **System Simulation** | Full transaction + unauthorized demo working in CLI |
| **Submission Quality** | Report, notebook, code clean and well-documented |

---

## Repository Structure (Target)

```
project/
├── data/
│   ├── raw/           # customer_social_profiles, customer_transactions
│   ├── processed/     # merged_dataset.csv, image_features.csv, audio_features.csv
│   ├── images/       # Per-member images + augmented/
│   └── audio/        # Per-member audio + augmented/
├── models/
│   ├── facial_recognition_model.pkl
│   ├── voiceprint_model.pkl
│   └── product_recommendation_model.pkl
├── scripts/
│   ├── train_product_model.py
│   ├── augment_images.py
│   ├── augment_audio.py
│   └── run_system.py   # CLI app
├── notebooks/
│   └── formative2_analysis.ipynb
├── outputs/           # Plots, screenshots
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
└── formative2-task-breakdown.md  # This file
```

---

## Suggested Git Workflow

1. **Main branch:** `main` — only merged, working code
2. **Feature branches:** `person1/data-merge`, `person2/images`, `person3/audio`, `person4/cli`
3. **Commit frequency:** Aim for 10+ commits per person; each step above = 1 commit
4. **Pull before push:** Always pull latest before pushing to avoid conflicts
5. **PRs:** Optional but recommended for Person 4’s integration work

---

## Timeline Suggestion

| Week | Person 1 | Person 2 | Person 3 | Person 4 |
|------|----------|----------|----------|----------|
| 1 | Steps 1–5 (EDA, cleaning) | Steps 1–4 (collect, augment) | Steps 1–4 (collect, augment) | Step 1 (CLI structure) |
| 2 | Steps 6–10 (merge, model) | Steps 5–9 (features, CSV) | Steps 5–10 (features, CSV) | Steps 2–4 (load models) |
| 3 | — | Step 10 (face model) | Step 11 (voice model) | Steps 5–12 (integration, demos) |

Person 4 can start CLI structure early and integrate incrementally as models become available.
