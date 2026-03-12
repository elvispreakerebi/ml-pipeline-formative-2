#!/usr/bin/env python3
"""
Formative 2 — System Simulation CLI

Simulates the User Identity and Product Recommendation flow:
1. Face recognition (must match known member)
2. Product recommendation (based on merged customer data)
3. Voice verification (must match approved voiceprint)
4. Display predicted product only if all steps pass

Usage:
  python scripts/run_system.py --demo-full --face-image path --voice-audio path --customer-id N
  python scripts/run_system.py --demo-unauthorized-face --face-image path
  python scripts/run_system.py --demo-unauthorized-voice --voice-audio path

Run from project root.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

# Ensure project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(PROJECT_ROOT)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FACE_MODEL_PATH = MODELS_DIR / "facial_recognition_model.pkl"
VOICE_MODEL_PATH = MODELS_DIR / "voiceprint_model.joblib"
VOICE_SCALER_PATH = MODELS_DIR / "voice_scaler.joblib"
VOICE_LE_PATH = MODELS_DIR / "voice_label_encoder.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_recommendation_model.pkl"
MERGED_CSV_PATH = DATA_DIR / "processed" / "merged_dataset.csv"

CONFIDENCE_THRESHOLD = 0.7
MEMBER_NAMES = {1: "Josue", 2: "Bonaparte", 3: "Yunis", 4: "Preye"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Formative 2 — User Identity and Product Recommendation System Simulation"
    )
    parser.add_argument("--demo-full", action="store_true", help="Run full transaction")
    parser.add_argument("--demo-unauthorized-face", action="store_true", help="Unauthorized face demo")
    parser.add_argument("--demo-unauthorized-voice", action="store_true", help="Unauthorized voice demo")
    parser.add_argument("--face-image", type=str, help="Path to face image")
    parser.add_argument("--voice-audio", type=str, help="Path to voice audio file")
    parser.add_argument("--customer-id", type=int, default=1, help="Customer ID for product prediction")
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
    """Load audio and extract features for voice model."""
    import librosa
    from audio_features import extract_audio_features
    y, sr = librosa.load(str(audio_path), sr=None)
    feat = extract_audio_features(y, sr)
    return feat.reshape(1, -1)


def verify_face(face_image_path, face_bundle, verbose=False):
    """Verify face. Returns (authorized: bool, member_name: str or None)."""
    if not Path(face_image_path).exists():
        raise FileNotFoundError(f"Face image not found: {face_image_path}")
    img = cv2.imread(str(face_image_path))
    if img is None:
        raise ValueError(f"Could not load image: {face_image_path}")
    features = extract_image_features(img)
    model = face_bundle["model"]
    proba = model.predict_proba(features)[0]
    confidence = proba.max()
    pred_idx = np.argmax(proba)
    authorized_members = face_bundle.get("authorized_members", [1, 2, 3, 4])
    member_id = authorized_members[pred_idx] if pred_idx < len(authorized_members) else pred_idx + 1
    member_name = MEMBER_NAMES.get(member_id, f"Member{member_id}")
    authorized = confidence >= CONFIDENCE_THRESHOLD
    if verbose:
        print(f"  Face confidence: {confidence:.4f} (threshold: {CONFIDENCE_THRESHOLD})")
    return authorized, member_name


def predict_product(customer_id, product_model):
    """Predict product for customer. Returns product name or None."""
    if not MERGED_CSV_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_CSV_PATH}")
    df = pd.read_csv(MERGED_CSV_PATH)
    feature_cols = [
        "avg_purchase_amount", "total_spent", "transaction_count", "avg_rating",
        "avg_engagement_score", "avg_purchase_interest", "engagement_x_interest",
        "spending_per_transaction", "sentiment_encoded"
    ] + [c for c in df.columns if c.startswith("platform_")]
    feature_cols = [c for c in feature_cols if c in df.columns]
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None
    X = row[feature_cols].values
    pred = product_model.predict(X)[0]
    return str(pred)


def verify_voice(audio_path, voice_bundle, verbose=False):
    """Verify voice. Returns (authorized: bool, speaker_name: str or None)."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    feat = extract_audio_features_from_file(audio_path)
    model = voice_bundle["model"]
    scaler = voice_bundle.get("scaler")
    le = voice_bundle.get("label_encoder")
    if scaler is not None:
        feat = scaler.transform(feat)
    pred = model.predict(feat)[0]
    if le is not None:
        speaker_name = le.inverse_transform([pred])[0]
    else:
        speaker_name = str(pred)
    authorized = speaker_name in list(MEMBER_NAMES.values())
    if verbose:
        print(f"  Voice predicted: {speaker_name}")
    return authorized, speaker_name


def run_full_transaction(args):
    """Execute full flow: face → product → voice → display."""
    print("\n--- Step 1: Facial Recognition ---")
    if not args.face_image:
        print("Access Denied: --face-image required")
        return 1
    face_bundle = load_face_model()
    try:
        auth, name = verify_face(args.face_image, face_bundle, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if not auth:
        print("Access Denied: Face not recognized (confidence below threshold)")
        return 1
    print(f"  ✓ Face recognized: {name}")

    print("\n--- Step 2: Product Recommendation ---")
    product_model = load_product_model()
    product = predict_product(args.customer_id, product_model)
    if product is None:
        print("Access Denied: Customer not found in dataset")
        return 1
    print(f"  ✓ Predicted product: {product}")

    print("\n--- Step 3: Voice Verification ---")
    if not args.voice_audio:
        print("Access Denied: --voice-audio required")
        return 1
    voice_bundle = load_voice_model()
    try:
        auth, speaker = verify_voice(args.voice_audio, voice_bundle, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if not auth:
        print("Access Denied: Voice not recognized (unauthorized speaker)")
        return 1
    print(f"  ✓ Voice verified: {speaker}")

    print("\n" + "=" * 60)
    print(f"Predicted Product: {product}")
    print("=" * 60)
    return 0


def run_unauthorized_face_demo(args):
    """Demo: unauthorized face attempt."""
    print("\n--- Unauthorized Face Attempt ---")
    face_path = args.face_image or str(DATA_DIR / "images" / "unauthorized_face.jpg")
    if not Path(face_path).exists():
        print(f"Error: Face image not found: {face_path}")
        return 1
    face_bundle = load_face_model()
    try:
        auth, name = verify_face(face_path, face_bundle, args.verbose)
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
        print("Error: --voice-audio required for unauthorized voice demo")
        return 1
    voice_bundle = load_voice_model()
    try:
        auth, speaker = verify_voice(args.voice_audio, voice_bundle, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if auth:
        print(f"  (Unexpected) Voice verified: {speaker}")
    else:
        print("Access Denied: Voice not recognized (unauthorized speaker)")
    return 0


def main():
    parser, args = parse_args()

    if not any([args.demo_full, args.demo_unauthorized_face, args.demo_unauthorized_voice]):
        parser.print_help()
        print("\nError: Specify at least one demo mode")
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
