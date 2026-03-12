"""Audio feature extraction for voiceprint model — matches TASK3 pipeline.
Uses scipy/numpy only (no librosa) for compatibility with Python 3.13 and systems without OpenMP.
"""

import numpy as np
from scipy.fft import dct
from scipy.signal import stft


def _hz_to_mel(hz):
    return 1127.01048 * np.log(1 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel):
    return 700 * (np.exp(mel / 1127.01048) - 1)


def _mel_filterbank(sr, n_fft=2048, n_mels=26, fmin=0, fmax=None):
    """Create mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        filterbank[i, left:center] = (np.arange(left, center) - left) / (center - left)
        filterbank[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filterbank


def _compute_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=26):
    """Compute MFCCs from audio signal."""
    # Pad for alignment
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    f, t, Zxx = stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(Zxx) ** 2
    filterbank = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = filterbank @ power
    mel_spec = np.where(mel_spec > 0, mel_spec, 1e-10)
    log_mel = np.log(mel_spec)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfcc


def _spectral_rolloff(y, sr, n_fft=2048, hop_length=512, roll_percent=0.85):
    """Compute spectral roll-off frequency."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    f, t, Zxx = stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)
    threshold = roll_percent * np.sum(magnitude, axis=0, keepdims=True)
    cumsum = np.cumsum(magnitude, axis=0)
    rolloff_bins = np.argmax(cumsum >= threshold, axis=0)
    rolloff_hz = f[rolloff_bins]
    return rolloff_hz


def _rms_energy(y, frame_length=2048, hop_length=512):
    """Compute RMS energy per frame."""
    if len(y) < frame_length:
        return np.array([np.sqrt(np.mean(y**2))])
    n_frames = 1 + (len(y) - frame_length) // hop_length
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame**2))
    return rms


def extract_audio_features(y, sr, n_mfcc=13):
    """Extract MFCCs, spectral roll-off, and energy. Returns flat vector.
    Compatible with librosa output format for voice model compatibility.
    """
    mfcc = _compute_mfcc(y, sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    rolloff = _spectral_rolloff(y, sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))
    rms = _rms_energy(y)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    return np.concatenate([mfcc_mean, mfcc_std, [rolloff_mean, rolloff_std], [energy_mean, energy_std]])
