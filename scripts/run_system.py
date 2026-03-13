#!/usr/bin/env python3
"""
Formative 2 — System Simulation CLI

Simulates the User Identity and Product Recommendation flow:
1. Face recognition (must match known member)
2. Product recommendation (based on merged customer data)
3. Voice verification (must match approved voiceprint)
4. Display predicted product only if all steps pass

Usage:
  python scripts/run_system.py --demo-full --face-image path --voice-audio path
  python scripts/run_system.py --demo-unauthorized-face --face-image path
  python scripts/run_system.py --demo-unauthorized-voice --voice-audio path

Run from project root. Uses files from data/images/ and data/audio/.
"""

import argparse
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Trying to unpickle estimator", category=UserWarning)

try:
    import cv2
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("Missing dependencies. Installing opencv-python soundfile...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "opencv-python", "soundfile"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print("Installed. Re-running...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except (subprocess.CalledProcessError, OSError):
        print("Auto-install failed. Run manually:")
        print("  python -m pip install opencv-python soundfile")
        print(f"  ({e})")
        sys.exit(1)

# Ensure project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(PROJECT_ROOT)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
FACE_MODEL_PATH = MODELS_DIR / "facial_recognition_model.pkl"
VOICE_MODEL_PATH = MODELS_DIR / "voiceprint_model.joblib"
VOICE_SCALER_PATH = MODELS_DIR / "voice_scaler.joblib"
VOICE_LE_PATH = MODELS_DIR / "voice_label_encoder.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_recommendation_model.pkl"
MERGED_CSV_PATH = DATA_DIR / "processed" / "merged_dataset.csv"

DEFAULT_FACE_THRESHOLD = 0.45
MEMBER_NAMES = {1: "Josue", 2: "Bonaparte", 3: "Yunis", 4: "Elvis Preye Kerebi"}
VOICE_DISPLAY_NAMES = {"Preye": "Elvis Preye Kerebi", "Josue": "Josue", "Bonaparte": "Bonaparte", "Yunis": "Yunis"}
VOICE_LABEL_TO_MEMBER = {"Preye": 4, "Josue": 1, "Bonaparte": 2, "Yunis": 3}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Formative 2 — User Identity and Product Recommendation System Simulation"
    )
    parser.add_argument("--demo-full", action="store_true", help="Run full transaction")
    parser.add_argument("--demo-unauthorized-face", action="store_true", help="Unauthorized face demo")
    parser.add_argument("--demo-unauthorized-voice", action="store_true", help="Unauthorized voice demo")
    parser.add_argument("--face-image", type=str, help="Path to face image (e.g. data/images/member4/member4_neutral.jpg)")
    parser.add_argument("--voice-audio", type=str, help="Path to voice audio (e.g. data/audio/samples/Preye-REC.m4a)")
    parser.add_argument("--face-threshold", type=float, default=DEFAULT_FACE_THRESHOLD, help=f"Face confidence threshold (default: {DEFAULT_FACE_THRESHOLD})")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser, parser.parse_args()


def load_face_model():
    """Load facial recognition model bundle."""
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Face model not found: {FACE_MODEL_PATH}")
    with open(FACE_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_voice_model():
    """Load voiceprint model, scaler, and label encoder."""
    if not VOICE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Voice model not found: {VOICE_MODEL_PATH}")
    model = joblib.load(VOICE_MODEL_PATH)
    scaler = joblib.load(VOICE_SCALER_PATH) if VOICE_SCALER_PATH.exists() else None
    le = joblib.load(VOICE_LE_PATH) if VOICE_LE_PATH.exists() else None
    return {"model": model, "scaler": scaler, "label_encoder": le}


def load_product_model():
    """Load product recommendation model."""
    if not PRODUCT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Product model not found: {PRODUCT_MODEL_PATH}")
    return joblib.load(PRODUCT_MODEL_PATH)


def extract_image_features(img):
    """Extract histogram + HOG from BGR image."""
    from feature_extractors import extract_image_features as _extract
    return _extract(img)


def extract_audio_features_from_file(audio_path):
    """Load audio and extract features for voice model.
    Uses soundfile for WAV/FLAC/OGG; audioread for M4A and other formats.
    """
    from audio_features import extract_audio_features

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix = path.suffix.lower()
    if suffix in (".m4a", ".aac", ".mp4", ".mp3"):
        # soundfile/libsndfile does not support these; use audioread
        import audioread
        with audioread.audio_open(str(path)) as f:
            sr = f.samplerate
            chunks = []
            for buf in f:
                chunks.append(buf)
            raw = b"".join(chunks)
            # audioread typically decodes to 16-bit signed PCM
            y = np.frombuffer(raw, dtype=np.int16)
            if f.channels > 1:
                y = y.reshape(-1, f.channels).mean(axis=1)
            y = y.astype(np.float32) / 32768.0
    else:
        import soundfile as sf
        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

    feat = extract_audio_features(y, sr)
    return feat.reshape(1, -1)


def verify_face(face_path, face_bundle, threshold=0.45, verbose=False):
    """Verify face from image path. Returns (authorized: bool, member_name: str, member_id: int)."""
    path = Path(face_path)
    if not path.exists():
        raise FileNotFoundError(f"Face image not found: {face_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {face_path}")
    features = extract_image_features(img)
    model = face_bundle["model"]
    proba = model.predict_proba(features)[0]
    confidence = proba.max()
    pred_idx = np.argmax(proba)
    authorized_members = face_bundle.get("authorized_members", [1, 2, 3, 4])
    member_id = authorized_members[pred_idx] if pred_idx < len(authorized_members) else pred_idx + 1
    member_name = MEMBER_NAMES.get(member_id, f"Member{member_id}")
    authorized = confidence >= threshold
    if verbose or not authorized:
        print(f"  Face confidence: {confidence:.4f} (threshold: {threshold})")
    return authorized, member_name, member_id


def predict_product(product_model):
    """Predict product using first customer profile from merged dataset."""
    if not MERGED_CSV_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_CSV_PATH}")
    df = pd.read_csv(MERGED_CSV_PATH)
    if df.empty:
        return None
    row = df.iloc[0]
    expected = getattr(product_model, "feature_names_in_", None)
    if expected is not None:
        values = []
        for col in expected:
            if col in row.index:
                val = row[col]
            elif col in df.columns:
                val = row[col]
            else:
                val = 0
            values.append(1 if val is True else (0 if val is False else float(val)))
        X = np.array(values, dtype=np.float64).reshape(1, -1)
    else:
        feature_cols = [
            "avg_purchase_amount", "total_spent", "transaction_count", "avg_rating",
            "avg_engagement_score", "avg_purchase_interest", "engagement_x_interest",
            "spending_per_transaction", "sentiment_encoded"
        ] + [c for c in df.columns if c.startswith("platform_")]
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df.iloc[[0]][feature_cols].values.astype(np.float64)
    pred = product_model.predict(X)[0]
    return str(pred)


def verify_voice(audio_path, voice_bundle, verbose=False):
    """Verify voice from audio path. Returns (authorized: bool, display_name: str, member_id: int or None)."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    feat = extract_audio_features_from_file(path)
    model = voice_bundle["model"]
    scaler = voice_bundle.get("scaler")
    le = voice_bundle.get("label_encoder")
    if scaler is not None:
        feat = scaler.transform(feat)
    pred = model.predict(feat)[0]
    if le is not None:
        speaker_label = le.inverse_transform([pred])[0]
    else:
        speaker_label = str(pred)
    display_name = VOICE_DISPLAY_NAMES.get(speaker_label, speaker_label)
    member_id = VOICE_LABEL_TO_MEMBER.get(speaker_label)
    authorized = member_id is not None
    if verbose:
        print(f"  Voice predicted: {display_name}")
    return authorized, display_name, member_id


def run_full_transaction(args):
    """Execute full flow: face → product → voice → display."""
    print("\n--- Step 1: Facial Recognition ---")
    face_path = args.face_image
    if not face_path:
        print("Access Denied: --face-image required")
        return 1
    face_bundle = load_face_model()
    try:
        auth, face_name, face_id = verify_face(face_path, face_bundle, args.face_threshold, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if not auth:
        print("Access Denied: Face not recognized")
        return 1
    print(f"  ✓ Face recognized: {face_name}")

    print("\n--- Step 2: Product Recommendation ---")
    product_model = load_product_model()
    product = predict_product(product_model)
    if product is None:
        print("Access Denied: Merged dataset empty")
        return 1
    print("  ✓ Product recommendation ready")

    print("\n--- Step 3: Voice Verification ---")
    voice_path = args.voice_audio
    if not voice_path:
        print("Access Denied: --voice-audio required")
        return 1
    voice_bundle = load_voice_model()
    try:
        auth, voice_name, voice_id = verify_voice(voice_path, voice_bundle, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if not auth:
        print("Access Denied: Voice not recognized (unauthorized speaker)")
        return 1
    if face_id != voice_id:
        print("Access Denied: Face and voice do not match the same person")
        return 1
    print(f"  ✓ Voice verified: {voice_name}")

    print("\n" + "=" * 60)
    print(f"Predicted Product: {product}")
    print("=" * 60)
    return 0


def run_unauthorized_face_demo(args):
    """Demo: unauthorized face attempt."""
    print("\n--- Unauthorized Face Attempt ---")
    face_path = args.face_image or str(IMAGES_DIR / "unauthorized_face.jpg")
    if not Path(face_path).exists():
        print(f"Error: Face image not found: {face_path}")
        return 1
    face_bundle = load_face_model()
    try:
        auth, name, _ = verify_face(face_path, face_bundle, args.face_threshold, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if auth:
        print(f"  (Unexpected) Face recognized: {name}")
    else:
        print("Access Denied: Face not recognized (unauthorized user detected)")
    return 0


def run_unauthorized_voice_demo(args):
    """Demo: unauthorized voice attempt."""
    print("\n--- Unauthorized Voice Attempt ---")
    if not args.voice_audio:
        print("Error: --voice-audio required")
        return 1
    voice_bundle = load_voice_model()
    try:
        auth, voice_name, _ = verify_voice(args.voice_audio, voice_bundle, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if auth:
        print(f"  (Unexpected) Voice verified: {voice_name}")
    else:
        print("Access Denied: Voice not recognized (unauthorized speaker)")
    return 0


def main():
    parser, args = parse_args()

    if not any([args.demo_full, args.demo_unauthorized_face, args.demo_unauthorized_voice]):
        parser.print_help()
        print("\nError: Specify at least one demo mode (--demo-full, --demo-unauthorized-face, --demo-unauthorized-voice)")
        sys.exit(1)

    if args.demo_full and (not args.face_image or not args.voice_audio):
        parser.print_help()
        print("\nError: --demo-full requires --face-image and --voice-audio")
        sys.exit(1)

    print("=" * 60)
    print("Formative 2 — System Simulation")
    print("=" * 60)

    exit_code = 0
    if args.demo_full:
        exit_code = run_full_transaction(args)
    elif args.demo_unauthorized_face:
        exit_code = run_unauthorized_face_demo(args)
    elif args.demo_unauthorized_voice:
        exit_code = run_unauthorized_voice_demo(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
