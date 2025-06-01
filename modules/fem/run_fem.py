# modules/fem/run_fem.py

import os
import shutil
import uuid
import subprocess
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# ─── PATH SETUP ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEM_ROOT     = PROJECT_ROOT / 'modules' / 'fem'
MODEL_DIR    = FEM_ROOT / 'models'
RESULT_DIR   = FEM_ROOT / 'result_images'
UPLOAD_DIR   = PROJECT_ROOT / 'static' / 'uploads'

# Ensure folders exist
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def clear_results():
    """ลบไฟล์เก่าใน result_images ก่อนรันใหม่"""
    for f in RESULT_DIR.iterdir():
        if f.is_file():
            f.unlink()

def is_color_distorted(img_path: Path, threshold: float = 0.1) -> bool:
    """
    ตรวจ color distortion แบบ Gray-world assumption:
    |Rmean-Gmean|/Gmean หรือ |Bmean-Gmean|/Gmean > threshold → เพี้ยน
    """
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img).astype(np.float32)
    # ถ้า grayscale ข้ามเลย
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    means = arr.reshape(-1, 3).mean(axis=0)
    r, g, b = means
    return (abs(r-g)/g > threshold) or (abs(b-g)/g > threshold)

def apply_fem(input_fp: str) -> str:
    """
    1) ล้างโฟลเดอร์ result_images
    2) ตรวจ distortion
    3) ถ้าเพี้ยน → เรียก demo_single_image.py (--task awb) เพื่อสร้าง <stem>_AWB.png
    4) คัดลอก AWB ผลลัพธ์ไป static/uploads/ ด้วยชื่อใหม่มี uuid
    5) คืนพาธไฟล์สุดท้าย
    """
    inp = Path(input_fp)
    clear_results()

    # 2) ตรวจสีเพี้ยน
    if not is_color_distorted(inp):
        return str(inp)

    # 3) รัน AWB
    cmd = [
        sys.executable,
        str(FEM_ROOT / 'demo_single_image.py'),
        '--model_dir', str(MODEL_DIR),
        '--input',      str(inp),
        '--output_dir', str(RESULT_DIR),
        '--task',       'awb',
        '--save'
    ]
    subprocess.check_call(cmd, cwd=str(FEM_ROOT))

    # 4) หาไฟล์ AWB
    stem = inp.stem
    awb_files = list(RESULT_DIR.glob(f"{stem}_AWB.*"))
    if not awb_files:
        raise FileNotFoundError(f"ไม่พบผลลัพธ์ AWB ใน {RESULT_DIR}")
    awb_fp = awb_files[0]

    # คัดลอกไป static/uploads พร้อม uuid
    ext        = awb_fp.suffix
    final_name = f"{stem}_awb_{uuid.uuid4().hex[:6]}{ext}"
    dest       = UPLOAD_DIR / final_name
    shutil.copy(awb_fp, dest)

    return str(dest)
