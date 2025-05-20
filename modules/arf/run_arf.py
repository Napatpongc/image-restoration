# modules/arf/run_arf.py

import sys
import subprocess
import uuid
import shutil
from pathlib import Path

# ─── PATH SETUP ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <project_root>/image_restoration
UPLOAD_DIR   = PROJECT_ROOT / 'static' / 'uploads'


def _run_deblurgan(in_path: Path) -> Path:
    """
    Wrapper สำหรับ DeblurGAN-v2
    """
    out_path = in_path.with_name(f"{in_path.stem}_deblur_{uuid.uuid4().hex[:6]}.jpg")
    script   = PROJECT_ROOT / 'modules' / 'arf' / 'deblurganv2' / 'predict.py'
    subprocess.check_call([sys.executable, str(script), str(in_path), str(out_path)])
    return out_path


def _run_ffdnet(in_path: Path, sigma: str) -> Path:
    """
    Wrapper สำหรับ FFDNet
    """
    base_dir   = PROJECT_ROOT / 'modules' / 'arf' / 'ffdnet'
    python_exe = base_dir / 'venv' / 'Scripts' / 'python.exe'
    script     = base_dir / 'test_ffdnet_ipol.py'

    out_name = f"{in_path.stem}_denoise_{uuid.uuid4().hex[:6]}.png"
    out_path = in_path.with_name(out_name)

    cmd = [
        str(python_exe), str(script),
        '--input',       str(in_path),
        '--noise_sigma', sigma,
        '--add_noise',   'False',
        '--no_gpu',
        '--output',      str(out_path)
    ]
    subprocess.check_call(cmd, cwd=str(base_dir))
    return out_path


def _run_edsr(in_path: Path) -> Path:
    """
    Wrapper สำหรับ EDSR Super-Resolution (×4):
      1) copy input → edsr/data/Demo
      2) รัน main.py ใน venv Python3.6
      3) รวบรวมผลลัพธ์จาก modules/arf/experiment/test/results-Demo
      4) copy กลับมา static/uploads
    """
    # Paths
    edsr_dir   = PROJECT_ROOT / 'modules' / 'arf' / 'edsr'
    python_exe = edsr_dir / 'venv' / 'Scripts' / 'python.exe'
    script     = edsr_dir / 'main.py'
    model_file = PROJECT_ROOT / 'modules' / 'arf' / 'experiment' / 'model' / 'EDSR_x4.pt'

    # 1) เตรียม edsr/data/Demo และ copy input
    demo_name = 'Demo'
    demo_dir  = edsr_dir / 'data' / demo_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_in   = demo_dir / in_path.name
    shutil.copy(in_path, demo_in)

    # 2) สั่งรัน main.py ของ EDSR
    cmd = [
        str(python_exe), str(script),
        '--data_test',   demo_name,
        '--scale',       '4',
        '--model',       'EDSR',
        '--n_resblocks', '32',
        '--n_feats',     '256',
        '--res_scale',   '0.1',
        '--pre_train',   str(model_file),
        '--test_only',
        '--save_results',
        '--cpu',
        '--n_threads',   '1'
    ]
    subprocess.check_call(cmd, cwd=str(edsr_dir))

    # 3) รวบรวมผลลัพธ์จาก modules/arf/experiment/test/results-Demo
    result_dir = PROJECT_ROOT / 'modules' / 'arf' / 'experiment' / 'test' / f'results-{demo_name}'
    # output filename pattern: <stem>_x4_SR.png
    sr_files   = list(result_dir.glob(f"{in_path.stem}_x4_SR.*"))
    if not sr_files:
        raise FileNotFoundError(f"ไม่พบผลลัพธ์ SR ใน {result_dir}")
    out_file = sr_files[0]

    # 4) copy ผลลัพธ์ไป static/uploads
    final_name = f"{in_path.stem}_sr4_{uuid.uuid4().hex[:6]}.png"
    dest       = UPLOAD_DIR / final_name
    shutil.copy(out_file, dest)
    return dest


def apply_arf(in_path: str, kind: str, sigma: str = None) -> str:
    """
    Dispatcher สำหรับ DeblurGAN-v2, FFDNet, EDSR
    (kind: 'blur' | 'noise' | 'hr')
    """
    p = Path(in_path)
    if 'blur' in kind:
        return str(_run_deblurgan(p))
    if 'noise' in kind:
        return str(_run_ffdnet(p, sigma))
    if 'hr' in kind:
        return str(_run_edsr(p))
    return in_path
