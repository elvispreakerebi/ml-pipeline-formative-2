# Task 2: Image Data Collection, Processing & Facial Recognition Model

## Overview
This task collects facial images from all group members, applies image augmentations, extracts visual features, and trains a facial recognition model to identify authorized users.

## Group Members
| Member ID | Name      | Expressions Collected            |
|-----------|-----------|----------------------------------|
| 1         | Josue     | Neutral, Smiling, Surprised      |
| 2         | Bonaparte | Neutral (serious), Smiling, Surprised (amazed) |
| 3         | Yunis     | Neutral (serious), Smiling, Surprised |
| 4         | Preye     | Neutral, Smiling, Surprised      |

## Pipeline

### 1. Image Collection
- 3 facial images per member (neutral, smiling, surprised) = **12 original images**
- Naming convention: `member{N}_{expression}.jpg`
- Stored in `data/images/member{N}/`

### 2. Image Display
- All 12 images displayed in a 4x3 grid with member names and expressions
- Saved to `outputs/member_images_grid.png`

### 3. Image Augmentation
Three augmentations applied per image:
- **Rotation (+15 degrees)** — simulates head tilt
- **Horizontal Flip** — simulates camera angle variation
- **Grayscale** — simulates lighting variation

This produces **36 augmented images** + 12 originals = **48 total images**.
Augmented images saved to `data/images/augmented/`.

### 4. Feature Extraction
Two feature types extracted from each image:
- **Color Histograms** — RGB histograms with 32 bins per channel (96 features)
- **HOG (Histogram of Oriented Gradients)** — edge/texture patterns from 64x64 grayscale (1,764 features)

Total: **1,860 features per image**

### 5. Output: image_features.csv
- Saved to `data/processed/image_features.csv`
- Shape: 48 rows x 1,865 columns (5 metadata + 96 histogram + 1,764 HOG)
- Zero missing values

### 6. Facial Recognition Model
Two models trained to classify faces by member ID (4-class classification):

| Model               | Accuracy | F1-Score | Log Loss |
|---------------------|----------|----------|----------|
| Random Forest       | 1.0000   | 1.0000   | 0.2232   |
| Logistic Regression | 1.0000   | 1.0000   | 0.0957   |

- Train/test split: 80/20 (stratified)
- Best model (Random Forest) saved to `models/facial_recognition_model.pkl`

### 7. Unauthorized Access Demo
- A real face image of a person **not in the group** is tested against the model
- The model uses a confidence threshold (0.7) to detect unauthorized users
- If max prediction confidence is below the threshold, access is denied

## File Structure
```
data/
├── images/
│   ├── member1/          # Josue's facial images
│   ├── member2/          # Bonaparte's facial images
│   ├── member3/          # Yunis's facial images
│   ├── member4/          # Preye's facial images
│   ├── augmented/        # All augmented images (36 files)
│   └── unauthorized_face.jpg
├── processed/
│   └── image_features.csv
models/
│   └── facial_recognition_model.pkl
notebooks/
│   └── TASK2_IMAGE_PROCESSING.ipynb
outputs/
│   ├── member_images_grid.png
│   ├── augmentation_demo.png
│   ├── face_model_confusion_matrix.png
│   └── unauthorized_attempt.png
```

## How to Run
1. Ensure all dependencies are installed: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `Pillow`, `scikit-learn`
2. Open `notebooks/TASK2_IMAGE_PROCESSING.ipynb`
3. Run all cells from top to bottom
4. The notebook automatically sets the working directory to the project root
