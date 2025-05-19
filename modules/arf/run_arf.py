import sys, subprocess, uuid, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # image_restoration/
UPLOAD_DIR   = PROJECT_ROOT / 'static' / 'uploads'

def _run_deblurgan(in_path: Path) -> Path:
    out_path = in_path.with_name(f"{in_path.stem}_deblur_{uuid.uuid4().hex[:6]}.jpg")
    script   = PROJECT_ROOT / 'modules' / 'arf' / 'deblurganv2' / 'predict.py'
    subprocess.check_call([sys.executable, str(script), str(in_path), str(out_path)])
    return out_path

def _run_ffdnet(in_path: Path, sigma: str) -> Path:
    """
    เรียก FFDNet จาก venv Python3.6
    """
    # โฟลเดอร์ modules/arf/ffdnet
    base_dir   = Path(__file__).resolve().parent / 'ffdnet'
    # Python interpreter ใน venv
    python_exe = base_dir / 'venv' / 'Scripts' / 'python.exe'
    # สคริปต์ที่ปรับแล้ว
    script     = base_dir / 'test_ffdnet_ipol.py'

    # สร้างชื่อไฟล์ผลลัพธ์
    out_name   = f"{in_path.stem}_denoise_{uuid.uuid4().hex[:6]}.png"
    out_path   = in_path.with_name(out_name)

    cmd = [
        str(python_exe),
        str(script),
        '--input',        str(in_path),
        '--noise_sigma',  sigma,
        '--add_noise',    'False',
        '--no_gpu',
        '--output',       str(out_path)
    ]

    # รันใน cwd ของ ffdnet เพื่อให้หา models/ และ utils.py เจอ
    subprocess.check_call(cmd, cwd=str(base_dir))
    return out_path

def apply_arf(in_path: str, kind: str, sigma: str | None = None) -> str:
    """
    kind: 'blur' | 'noise' | 'hr' | 'blur+noise' …
    """
    # ถ้าเจอ blur
    if 'blur' in kind:
        in_p = Path(in_path)
        return str(_run_deblurgan(in_p))

    # ถ้าเจอ noise
    if 'noise' in kind:
        in_p = Path(in_path)
        return str(_run_ffdnet(in_p, sigma))

    # กรณีอื่น fallback
    return in_path
