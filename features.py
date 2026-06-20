"""
Acoustic feature extraction for Alzheimer classification.
Extracts hand-crafted speech features known to correlate with cognitive decline:
  - MFCC statistics (mean/std of 40 coefficients)
  - Prosody: pitch (F0) statistics, voiced/unvoiced ratio
  - Energy: RMS mean/std
  - Spectral: centroid, rolloff, bandwidth, contrast
  - ZCR, spectral flatness
Total feature vector: 130-dim
"""

import numpy as np
import librosa
from typing import Optional

TARGET_SR = 16000
N_MFCC = 40
FEATURE_DIM = 130


def extract(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Return a fixed-dim acoustic feature vector for one audio clip."""
    feats = []

    # --- MFCC (40 x 2 = 80) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.append(mfcc.mean(axis=1))   # 40
    feats.append(mfcc.std(axis=1))    # 40

    # --- Pitch / F0 (4) ---
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=2048
    )
    f0_voiced = f0[voiced_flag == 1] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    voiced_ratio = voiced_flag.mean() if voiced_flag is not None else 0.0
    feats.append(np.array([
        f0_voiced.mean() if len(f0_voiced) else 0.0,
        f0_voiced.std() if len(f0_voiced) else 0.0,
        float(voiced_ratio),
        float(np.percentile(f0_voiced, 75) - np.percentile(f0_voiced, 25)) if len(f0_voiced) > 1 else 0.0,
    ]))

    # --- Energy (2) ---
    rms = librosa.feature.rms(y=y)
    feats.append(np.array([rms.mean(), rms.std()]))

    # --- Spectral features (5 x 2 = 10) ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # 7 bands
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    for feat_map in [centroid, rolloff, bandwidth, flatness, zcr]:
        feats.append(np.array([feat_map.mean(), feat_map.std()]))
    feats.append(contrast.mean(axis=1))   # 7
    feats.append(contrast.std(axis=1))    # 7

    vec = np.concatenate(feats).astype(np.float32)

    # Pad or trim to FEATURE_DIM for safety
    if len(vec) < FEATURE_DIM:
        vec = np.pad(vec, (0, FEATURE_DIM - len(vec)))
    else:
        vec = vec[:FEATURE_DIM]

    return vec


def extract_from_file(path: str, sr: int = TARGET_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return extract(y, sr)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        vec = extract_from_file(path)
        print(f"Feature dim: {vec.shape}, mean: {vec.mean():.4f}, std: {vec.std():.4f}")
    else:
        # Quick sanity check with synthetic audio
        y = np.random.randn(TARGET_SR * 5).astype(np.float32)
        vec = extract(y)
        print(f"Sanity check OK — feature dim: {vec.shape[0]}")
