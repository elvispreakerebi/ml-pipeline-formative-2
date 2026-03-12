# Voice Recognition & Voiceprint Verification Model

**Formative 2 – Sound Data Collection and Processing + Voiceprint Verification**

This project loads local voice samples, extracts acoustic features, and trains a **voiceprint verification** classifier to identify speakers from audio. All processing is done in a Jupyter notebook using local data (no cloud or Google Drive).

## Features

- **Audio loading** from `VoiceModel/audio_samples/` (or `audio_samples/` if run from project root)
- **Label extraction** from filenames: first segment before `-` (e.g. `Bonapatre-REC.wav` → label `Bonaparte`)
- **Visualizations**: waveforms and spectrograms for sample files
- **Augmentations**: pitch shift, time stretch, and background noise (one original + 3 augmented clips per file)
- **Feature extraction**: MFCCs (mean/std), spectral roll-off (mean/std), and energy (RMS mean/std), stored in a pandas DataFrame and saved as CSV
- **Voiceprint model**: Random Forest classifier; evaluation with Accuracy, F1-Score, and Log Loss
- **Saved artifacts**: trained model, scaler, and label encoder for use in a CLI or other pipeline

## Project structure

```
VoiceModel/
├── README.md
├── voice_recognition.ipynb   # Main notebook: load → augment → features → train
├── audio_samples/            # Put your .wav (or .mp3, .flac, .ogg, .m4a) files here
├── audio_features.csv        # Generated: extracted features (one row per clip)
├── voiceprint_model.joblib   # Generated: trained Random Forest
├── voice_scaler.joblib       # Generated: StandardScaler fit on training data
├── voice_label_encoder.joblib  # Generated: label → integer mapping
└── myenv/                    # Optional virtual environment
```

## Setup

1. **Python**: 3.9+ recommended.

2. **Dependencies** (install in your environment or run the notebook’s first cell):

   ```bash
   pip install librosa soundfile pandas matplotlib seaborn scikit-learn joblib
   ```

3. **Audio data**: Place audio files in `VoiceModel/audio_samples/`. Use names like `SpeakerName-REC.wav` so the part before the first `-` becomes the speaker label.

## Usage

1. Open `voice_recognition.ipynb` in Jupyter (or VS Code / Cursor).
2. Run all cells in order:
   - Install deps (if needed) → imports & config → load dataset
   - Visualize waveforms/spectrograms → apply augmentations
   - Extract features (DataFrame → `audio_features.csv`) → train Random Forest → save model artifacts

3. Outputs:
   - **audio_features.csv**: one row per clip (original + augmented), columns = `label`, `source`, and feature columns (e.g. `mfcc_mean_0` … `energy_std`).
   - **VoiceModel/** (or cwd): `voiceprint_model.joblib`, `voice_scaler.joblib`, `voice_label_encoder.joblib` for inference elsewhere.

## Feature pipeline (dataframes)

Features are computed with `extract_audio_features(y, sr)`, which returns a **dict** of scalar features. The notebook builds a list of dicts (one per `(y, sr, label, source)`) and creates the feature table with:

```python
feature_rows = [
    {"label": label, "source": source_name, **extract_audio_features(y, sr)}
    for y, sr, label, source_name in all_audio_for_features
]
df_audio = pd.DataFrame(feature_rows)
```

So the pipeline is fully dataframe-oriented: one row per clip, named columns, then `df_audio.to_csv(...)` and `df_audio[feature_cols]` for training.

## Metrics

The voiceprint model is evaluated on a train/test split with:

- **Accuracy**
- **F1-Score** (macro)
- **Log Loss** (probabilistic)

Results and optional confusion matrix/classification report are printed in the notebook.

## License

Use according to your course or project terms.
