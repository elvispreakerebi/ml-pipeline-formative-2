"""Feature extraction for face and voice models — matches TASK2 and TASK3 pipelines."""

import numpy as np
import cv2


def extract_color_histogram(img, bins=32):
    """Extract flattened RGB color histogram features (96 features)."""
    features = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features


def extract_hog_features(img, target_size=(64, 64)):
    """Extract HOG features (1764 features)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    win_size = target_size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    n_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    features = hog.compute(resized)
    return features.flatten()


def extract_image_features(img):
    """Extract histogram + HOG features from BGR image. Returns shape (1, 1860)."""
    hist = extract_color_histogram(img)
    hog = extract_hog_features(img)
    return np.array(hist + list(hog)).reshape(1, -1)
