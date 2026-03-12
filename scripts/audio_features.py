"""Audio feature extraction for voiceprint model — matches TASK3 pipeline."""

import numpy as np


def extract_audio_features(y, sr, n_mfcc=13):
    """Extract MFCCs, spectral roll-off, and energy. Returns flat vector."""
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    return np.concatenate([mfcc_mean, mfcc_std, [rolloff_mean, rolloff_std], [energy_mean, energy_std]])
