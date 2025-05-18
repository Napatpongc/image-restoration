#!/usr/bin/env python3
"""
predict.py — Single‐image DeblurGAN-v2 inference for image_restoration project

Usage:
    python predict.py <input_image> <output_image>

Expects:
    - ไฟล์ weight ชื่อ fpn_inception.h5 ในโฟลเดอร์เดียวกับสคริปต์
      (modules/arf/deblurganv2/fpn_inception.h5)
    - ไฟล์ config.yaml ที่ระบุชื่อโมเดล (config/model: fpn_inception)
      อยู่ที่ modules/arf/deblurganv2/config/config.yaml
    - โฟลเดอร์ modules/arf/deblurganv2/aug.py และ models/network.py
      บน PYTHONPATH เพื่อเรียก `get_normalize` กับ `get_generator`
"""
import sys
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch

# ─── PATH SETUP ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))     # ให้ import aug.py กับ models.* ได้

from aug import get_normalize
from models.networks import get_generator

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: predict.py <input_image> <output_image>")

    inp_path  = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    # ─── โหลด config ────────────────────────────────────────────────
    cfg_path = ROOT / 'config' / 'config.yaml'
    if not cfg_path.exists():
        sys.exit(f"Error: config not found: {cfg_path}")
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get('model', 'fpn_inception')

    # ─── สร้างโมเดล + โหลด weight ───────────────────────────────────
    model = get_generator(model_name)                      # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    weight_fp = ROOT / 'fpn_inception.h5'
    if not weight_fp.exists():
        sys.exit(f"Error: weight not found: {weight_fp}")
    state = torch.load(str(weight_fp), map_location=device)
    # บางไฟล์ weight เก็บใต้ key 'model'
    state_dict = state.get('model', state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.train(True)  # DeblurGAN-v2 ต้องใช้ train mode

    # ─── เตรียม normalize fn ────────────────────────────────────────
    normalize = get_normalize()

    # ─── โหลดรูป → BGR→RGB ───────────────────────────────────────────
    img_bgr = cv2.imread(str(inp_path))
    if img_bgr is None:
        sys.exit(f"Error: cannot read image {inp_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ─── normalize → (–1..1) แล้ว pad ให้เป็น multiple of 32 ─────────
    img_norm, _ = normalize(img_rgb, img_rgb)
    h, w, _     = img_norm.shape
    block       = 32
    new_h       = ((h + block - 1) // block) * block
    new_w       = ((w + block - 1) // block) * block
    pad_h, pad_w = new_h - h, new_w - w
    img_pad = np.pad(
        img_norm,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='reflect'
    )

    # ─── H×W×C → 1×C×H×W tensor บน device ───────────────────────────
    x = torch.from_numpy(img_pad.transpose(2, 0, 1)[None]).float().to(device)

    # ─── inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()

    # ─── denorm และ crop กลับสัดส่วนเดิม ─────────────────────────────
    out_rgb = ((pred.transpose(1, 2, 0) + 1.0) / 2.0 * 255.0)
    out_rgb = np.clip(out_rgb, 0, 255).astype('uint8')[:h, :w, :]
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    # ─── เขียนไฟล์และพิมพ์ path ให้ orchestrator อ่านต่อ ───────────
    cv2.imwrite(str(out_path), out_bgr)
    print(out_path)  # orchestrator อ่าน stdout บรรทัดนี้

if __name__ == '__main__':
    main()
