"""
DEM – hand-crafted detector:
  • ตัดสินว่ารูปควรเข้า deblur, denoise, หรือ high-resolution (sr)
  • ถ้าเป็น denoise จะคืนค่า sigma (ระดับ noise) กลับไปด้วย
พิมพ์ออกทาง stdout รูปแบบ  <kind> <sigma_or_None>
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import math

# ───────── constants & helper ───────────────────────────────────────────────────
# Calibration factor: convert MAD estimate to Gaussian σ (E|X| for Gaussian noise = σ * sqrt(2/π))
CALIBRATION_FACTOR = math.sqrt(math.pi/2)*1.5  # ≈1.2533

# ───────── helper ───────────────────────────────────────────────────
def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_noise(gray):
    """Return calibrated noise sigma by Median Absolute Deviation (MAD)."""
    H = cv2.medianBlur(gray, 3)
    mad = np.mean(np.abs(gray.astype(np.float32) - H))
    return mad * CALIBRATION_FACTOR

# ───────── main ──────────────────────────────────────────────────────
def main(img_path: str):
    if not Path(img_path).is_file():
        print("hr None")
        return

    img        = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) blur detection
    lap_var    = variance_of_laplacian(gray)
    is_blur    = lap_var < 100    # threshold ตามต้องการ

    # 2) noise detection ด้วย sigma ที่ calibrated แล้ว
    sigma_est  = estimate_noise(gray)
    is_noise   = sigma_est > 12   # threshold σ ≈12

    # 3) low-res detection (SR)
    h, w       = gray.shape
    is_low_res = min(h, w) < 720  # ถ้าฝั่งใดเล็กกว่า 720 px

    if is_blur:
        print("blur None")
    elif is_noise:
        print(f"noise {sigma_est:.1f}")
    else:
        print("hr None")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: dem.py <image_path>")
    main(sys.argv[1])
