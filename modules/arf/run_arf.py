# modules/arf/run_arf.py

import sys
import subprocess
import uuid
import shutil
from pathlib import Path
from ..fem.run_fem import apply_fem

# ─── PATH SETUP ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR   = PROJECT_ROOT / 'static' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def run_dem(img_path: Path):
    """
    รัน dem.py เพื่อตรวจหาว่าในภาพมี noise, blur, หรือ low-resolution
    คืนค่า (kind, sigma) เช่น ('noise', 15.2) หรือ ('blur', None)
    """
    dem_script = PROJECT_ROOT / 'modules' / 'dem' / 'dem.py'
    out = subprocess.check_output(
        [sys.executable, str(dem_script), str(img_path)],
        text=True
    ).strip().split()
    kind  = out[0]
    sigma = None if out[1] == 'None' else float(out[1])
    return kind, sigma

def _run_ffdnet(in_path: Path, sigma: float) -> str:
    """
    Wrapper ให้ FFDNet เขียนผลลัพธ์ตรงไปที่ static/uploads
    """
    base_dir = PROJECT_ROOT / 'modules' / 'arf' / 'ffdnet'
    script   = base_dir / 'test_ffdnet_ipol.py'
    out_name = f"{in_path.stem}_denoise_{uuid.uuid4().hex[:6]}.png"
    out_path = UPLOAD_DIR / out_name

    cmd = [
        sys.executable, str(script),
        '--input',       str(in_path),
        '--noise_sigma', str(sigma),
        '--add_noise',   'False',
        '--output',      str(out_path)
    ]
    subprocess.check_call(cmd, cwd=str(base_dir))
    return str(out_path)

def _run_deblurgan(in_path: Path) -> str:
    """
    Wrapper ให้ DeblurGAN-v2 เขียนผลลัพธ์ตรงไปที่ static/uploads
    """
    script   = PROJECT_ROOT / 'modules' / 'arf' / 'deblurganv2' / 'predict.py'
    out_name = f"{in_path.stem}_deblur_{uuid.uuid4().hex[:6]}.jpg"
    out_path = UPLOAD_DIR / out_name

    subprocess.check_call([
        sys.executable,
        str(script),
        str(in_path),
        str(out_path)
    ])
    return str(out_path)

def _run_edsr(in_path: Path) -> str:
    """
    Wrapper ให้ EDSR เขียนผลลัพธ์ไปที่ static/uploads
    """
    edsr_dir   = PROJECT_ROOT / 'modules' / 'arf' / 'edsr'
    script     = edsr_dir / 'main.py'
    demo_name  = 'Demo'
    demo_dir   = edsr_dir / 'data' / demo_name

    # เคลียร์ data/Demo ก่อนเสมอ
    if demo_dir.exists():
        for f in demo_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        demo_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(in_path, demo_dir / in_path.name)

    scale      = '3'  # เปลี่ยนเป็น '4' ถ้าต้องการ x4
    model_file = PROJECT_ROOT / 'modules' / 'arf' / 'experiment' / 'model' / f'EDSR_x{scale}.pt'

    cmd = [
        sys.executable, str(script),
        '--data_test', demo_name,
        '--scale',     scale,
        '--model',     'EDSR',
        '--n_resblocks', '32',
        '--n_feats',     '256',
        '--res_scale',   '0.1',
        '--pre_train',   str(model_file),
        '--test_only',
        '--save_results',
        '--save_dir',   str(UPLOAD_DIR),
        '--n_threads',   '1',
    ]
    subprocess.check_call(cmd, cwd=str(edsr_dir))

    # หาไฟล์ <stem>_x{scale}_SR.* ใน static/uploads
    result_dir = UPLOAD_DIR
    sr_files   = list(result_dir.glob(f"{in_path.stem}_x{scale}_SR.*"))
    if not sr_files:
        sr_files = list(result_dir.glob(in_path.name))
    if not sr_files:
        raise FileNotFoundError(f"ไม่พบผลลัพธ์ EDSR ใน {result_dir}")

    out_file = sr_files[0]
    return str(out_file)

def apply_arf(input_path: str, kind: str = None, sigma: float = None) -> str:
    """
    Pipeline อัตโนมัติ:
      1) DEM
      2) ถ้าเจอ noise → FFDNet → DEM
      3) ถ้าเจอ blur  → DeblurGAN-v2 → DEM
      4) ถ้าเจอ low-res → EDSR
      หลังจากขั้นตอนสุดท้าย (denoise/deblur/SR) ให้ส่งภาพเข้า FEM ตรวจสี
    """
    p = Path(input_path)

    # 1) ถ้าไม่มี kind มาจาก front-end ให้ตรวจ DEM รอบแรก
    if kind is None:
        kind, sigma = run_dem(p)

    # 2) กรณีแรก DEM บอกว่าเป็น noise
    if kind == 'noise':
        p = Path(_run_ffdnet(p, sigma))      # รัน FFDNet → ได้ภาพ denoise
        kind, sigma = run_dem(p)             # ตรวจ DEM รอบสอง
        if kind == 'noise':
            # ถ้ายังเป็น noise จบ pipeline → ส่งเข้า FEM แล้ว return
            p = Path(apply_fem(str(p)))
            return str(p)
        # ถ้าเปลี่ยนเป็น blur หรือ hr (noise หายไป) ให้ดำเนินต่อ

    # 3) กรณี DEM บอกว่าเป็น blur
    if kind == 'blur':
        p = Path(_run_deblurgan(p))          # รัน DeblurGAN-v2 → ได้ภาพ deblur
        kind, sigma = run_dem(p)             # ตรวจ DEM รอบสอง
        if kind == 'blur':
            # ถ้ายังเป็น blur จบ pipeline → ส่งเข้า FEM แล้ว return
            p = Path(apply_fem(str(p)))
            return str(p)
        # ถ้าเปลี่ยนเป็น noise หรือ hr (blur หายไป) ให้ดำเนินต่อ

    # 4) กรณี DEM บอกว่าเป็น low-resolution (hr)
    if kind == 'hr':
        p = Path(_run_edsr(p))               # รัน EDSR → ได้ภาพ super-resolved
        p = Path(apply_fem(str(p)))          # ส่งเข้า FEM แล้ว return

    return str(p)
