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
  python scripts/run_system.py --demo-full --live          # Capture from webcam + mic
  python scripts/run_system.py --demo-unauthorized-face --face-image path
  python scripts/run_system.py --demo-unauthorized-voice --voice-audio path

Run from project root. Use --live for webcam and microphone capture.
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
    print("Missing dependencies. Installing opencv-python soundfile sounddevice...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "opencv-python", "soundfile", "sounddevice"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print("Installed. Re-running...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except (subprocess.CalledProcessError, OSError):
        print("Auto-install failed. Run manually:")
        print("  python -m pip install opencv-python soundfile sounddevice")
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
FACE_MODEL_PATH = MODELS_DIR / "facial_recognition_model.pkl"
VOICE_MODEL_PATH = MODELS_DIR / "voiceprint_model.joblib"
VOICE_SCALER_PATH = MODELS_DIR / "voice_scaler.joblib"
VOICE_LE_PATH = MODELS_DIR / "voice_label_encoder.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_recommendation_model.pkl"
MERGED_CSV_PATH = DATA_DIR / "processed" / "merged_dataset.csv"

DEFAULT_FACE_THRESHOLD = 0.45  # Lower for webcam (lighting/angle differ from training)
MEMBER_NAMES = {1: "Josue", 2: "Bonaparte", 3: "Yunis", 4: "Preye"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Formative 2 — User Identity and Product Recommendation System Simulation"
    )
    parser.add_argument("--demo-full", action="store_true", help="Run full transaction")
    parser.add_argument("--demo-unauthorized-face", action="store_true", help="Unauthorized face demo")
    parser.add_argument("--demo-unauthorized-voice", action="store_true", help="Unauthorized voice demo")
    parser.add_argument("--face-image", type=str, help="Path to face image (omit with --live)")
    parser.add_argument("--voice-audio", type=str, help="Path to voice audio file (omit with --live)")
    parser.add_argument("--live", action="store_true", help="Capture from webcam and microphone")
    parser.add_argument("--record-duration", type=float, default=3.0, help="Seconds to record voice when using --live (default: 3)")
    parser.add_argument("--face-threshold", type=float, default=DEFAULT_FACE_THRESHOLD, help=f"Face confidence threshold 0–1 (default: {DEFAULT_FACE_THRESHOLD}, lower = more lenient)")
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
    import soundfile as sf
    from audio_features import extract_audio_features
    y, sr = sf.read(str(audio_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    feat = extract_audio_features(y, sr)
    return feat.reshape(1, -1)


def extract_audio_features_from_raw(y, sr):
    """Extract features from raw audio (y, sr) for voice model."""
    from audio_features import extract_audio_features
    feat = extract_audio_features(y, sr)
    return feat.reshape(1, -1)


def capture_face_from_webcam():
    """Capture a single frame from the webcam. Press SPACE to capture, Q to quit.
    Returns BGR image (numpy array) or None if cancelled."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is it connected and not in use by another app?")
    print("  Webcam active. Press SPACE to capture, Q to quit.")
    frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read from webcam")
            display = frame.copy()
            cv2.putText(display, "SPACE = capture | Q = quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q") or key == 27:
                frame = None
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frame


def record_voice_from_microphone(duration_sec=3.0, sample_rate=22050):
    """Record audio from the default microphone. Returns (y, sr) or None on error."""
    import sounddevice as sd
    print(f"  Recording for {duration_sec:.1f} seconds... Speak now.")
    try:
        recording = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return recording.flatten(), sample_rate
    except Exception as e:
        raise RuntimeError(f"Microphone recording failed: {e}") from e


def verify_face(face_input, face_bundle, threshold=0.45, verbose=False):
    """Verify face. face_input: path (str/Path) or BGR image (ndarray). Returns (authorized: bool, member_name: str)."""
    if isinstance(face_input, np.ndarray):
        img = face_input
    else:
        path = Path(face_input)
        if not path.exists():
            raise FileNotFoundError(f"Face image not found: {face_input}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {face_input}")
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
    return authorized, member_name


def predict_product(product_model):
    """Predict product using first customer profile from merged dataset.
    Face/voice members are separate from Task 1's customer data; we use a sample profile for the demo.
    Uses model's feature_names_in_ to match exact training feature order (handles duplicates).
    """
    if not MERGED_CSV_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_CSV_PATH}")
    df = pd.read_csv(MERGED_CSV_PATH)
    if df.empty:
        return None
    row = df.iloc[0]
    # Use model's expected feature order (Task 1 may have saved duplicates)
    expected = getattr(product_model, "feature_names_in_", None)
    if expected is not None:
        values = []
        for col in expected:
            if col in row.index:
                val = row[col]
            elif col in df.columns:
                val = row[col]
            else:
                val = 0  # missing column, fill with 0
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


def verify_voice(audio_input, voice_bundle, verbose=False):
    """Verify voice. audio_input: path (str/Path) or (y, sr) tuple. Returns (authorized: bool, speaker_name: str)."""
    if isinstance(audio_input, tuple):
        y, sr = audio_input
        feat = extract_audio_features_from_raw(y, sr)
    else:
        path = Path(audio_input)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        feat = extract_audio_features_from_file(path)
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
    face_input = None
    if args.live:
        try:
            face_input = capture_face_from_webcam()
        except Exception as e:
            print(f"Access Denied: {e}")
            return 1
        if face_input is None:
            print("Access Denied: Capture cancelled")
            return 1
    else:
        if not args.face_image:
            print("Access Denied: --face-image required (or use --live)")
            return 1
        face_input = args.face_image
    face_bundle = load_face_model()
    try:
        auth, name = verify_face(face_input, face_bundle, args.face_threshold, args.verbose)
    except Exception as e:
        print(f"Access Denied: {e}")
        return 1
    if not auth:
        print("Access Denied: Face not recognized (try --face-threshold 0.3 for webcam)")
        return 1
    print(f"  ✓ Face recognized: {name}")

    print("\n--- Step 2: Product Recommendation ---")
    product_model = load_product_model()
    product = predict_product(product_model)
    if product is None:
        print("Access Denied: Merged dataset empty")
        return 1
    print(f"  ✓ Predicted product: {product}")

    print("\n--- Step 3: Voice Verification ---")
    voice_input = None
    if args.live:
        try:
            voice_input = record_voice_from_microphone(args.record_duration)
        except Exception as e:
            print(f"Access Denied: {e}")
            return 1
    else:
        if not args.voice_audio:
            print("Access Denied: --voice-audio required (or use --live)")
            return 1
        voice_input = args.voice_audio
    voice_bundle = load_voice_model()
    try:
        auth, speaker = verify_voice(voice_input, voice_bundle, args.verbose)
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
    face_input = None
    if args.live:
        try:
            face_input = capture_face_from_webcam()
        except Exception as e:
            print(f"Error: {e}")
            return 1
        if face_input is None:
            print("Capture cancelled")
            return 1
    else:
        face_path = args.face_image or str(DATA_DIR / "images" / "unauthorized_face.jpg")
        if not Path(face_path).exists():
            print(f"Error: Face image not found: {face_path}")
            return 1
        face_input = face_path
    face_bundle = load_face_model()
    try:
        auth, name = verify_face(face_input, face_bundle, args.face_threshold, args.verbose)
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
    voice_input = None
    if args.live:
        try:
            voice_input = record_voice_from_microphone(args.record_duration)
        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        if not args.voice_audio:
            print("Error: --voice-audio required (or use --live)")
            return 1
        voice_input = args.voice_audio
    voice_bundle = load_voice_model()
    try:
        auth, speaker = verify_voice(voice_input, voice_bundle, args.verbose)
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
        print("\nError: Specify at least one demo mode (--demo-full, --demo-unauthorized-face, --demo-unauthorized-voice)")
        sys.exit(1)

    if args.demo_full and not args.live and (not args.face_image or not args.voice_audio):
        parser.print_help()
        print("\nError: --demo-full requires --face-image and --voice-audio, or use --live")
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
