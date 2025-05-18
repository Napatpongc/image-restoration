import sys, subprocess, uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # image_restoration/
UPLOAD_DIR   = PROJECT_ROOT / 'static' / 'uploads'

def _run_deblurgan(in_path: Path) -> Path:
    out_path = in_path.with_name(f"{in_path.stem}_deblur_{uuid.uuid4().hex[:6]}.jpg")
    script   = PROJECT_ROOT / 'modules' / 'arf' / 'deblurganv2' / 'predict.py'
    subprocess.check_call([sys.executable, str(script), str(in_path), str(out_path)])
    return out_path

def apply_arf(in_path: str, kind: str, sigma: str | None = None) -> str:
    """
    kind: 'blur' | 'noise' | 'hr' | 'blur+noise' …
    ตอนนี้รองรับเฉพาะ blur → DeblurGAN-v2
    """
    if 'blur' in kind:
        return str(_run_deblurgan(Path(in_path)))
    return in_path          # ชนิดอื่น—ยังไม่ทำ
