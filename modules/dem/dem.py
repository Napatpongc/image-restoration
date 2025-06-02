# modules/dem/dem.py

import sys
import cv2
import numpy as np
from pathlib import Path
import math

# ───────── constants & helper ───────────────────────────────────────────────────
# Calibration factor: convert MAD estimate to Gaussian σ
CALIBRATION_FACTOR = math.sqrt(math.pi / 2) * 1.5  # ≈1.2533

# ───────── helper ───────────────────────────────────────────────────
def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_noise(gray: np.ndarray) -> float:
    """Return calibrated noise sigma by Median Absolute Deviation (MAD)."""
    H = cv2.medianBlur(gray, 3)
    mad = np.mean(np.abs(gray.astype(np.float32) - H))
    return mad * CALIBRATION_FACTOR

# ───────── main ──────────────────────────────────────────────────────
def main(img_path: str):
    if not Path(img_path).is_file():
        print("clean None")
        return

    img       = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w      = gray.shape

    # 1) noise detection (MAD)
    sigma_est = estimate_noise(gray)
    if sigma_est > 12:
        print(f"noise {sigma_est:.1f}")
        return

    # 2) blur detection (Variance of Laplacian)
    lap_var = variance_of_laplacian(gray)
    if lap_var < 100:
        print("blur None")
        return

    # 3) low-res detection
    #    - ถ้า min(width, height) < 720 → low-res
    #    - หรือภาพเบลอมาก (lap_var < 50) แต่เราตรวจ blur ที่ <100 ไปแล้ว
    if min(h, w) < 721 or lap_var < 1200:
        print("hr None")
        return

    # 4) clean (no noise, no blur, no low-res)
    print("clean None")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: dem.py <image_path>")
    main(sys.argv[1])
