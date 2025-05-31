import torch
import os
import time
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from tqdm import tqdm

# กำหนด seed เพื่อความ reproducibility
torch.manual_seed(args.seed)


def run_inference_only():
    """
    สคริปต์สำหรับรัน Inference เพียวๆ:
    - โหลดโมเดลครั้งเดียว
    - วิ่งบน GPU (หรือ CPU ถ้าไม่มี)
    - แสดง progress bar ด้วย tqdm
    """
    # 1) เตรียม device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 2) โหลด checkpoint & โมเดล
    ckp = utility.checkpoint(args)
    net = model.Model(args, ckp).to(device)
    if args.precision == 'half':
        net.half()
    net.eval()

    # 3) เตรียม DataLoader สำหรับ test
    loader_test = data.Data(args).loader_test

    # 4) ถ้าเซฟผลลัพธ์ เปิด background saving
    if args.save_results:
        ckp.begin_background()

    # 5) คำนวณจำนวนภาพทั้งหมด
    total_images = sum(len(d) for d in loader_test)
    if total_images == 0:
        print("No images to process.")
        return

    # 6) รัน inference พร้อม progress bar
    start_time = time.time()
    torch.set_grad_enabled(False)
    pbar = tqdm(total=total_images, desc='EDSR Inference', ncols=80)

    for idx_data, d in enumerate(loader_test):
        scale_idx = 0
        d.dataset.set_scale(scale_idx)
        for lr, hr, filename in d:
            # ย้าย tensor ไป device และ half ถ้าตั้งค่ามา
            if args.precision == 'half':
                lr = lr.half()
            lr = lr.to(device)

            # forward เพียงครั้งเดียว
            sr = net(lr, scale_idx)
            sr = utility.quantize(sr, args.rgb_range)

            # บันทึกผลลัพธ์
            if args.save_results:
                ckp.save_results(d, filename[0], [sr], args.scale[scale_idx])

            pbar.update(1)
    pbar.close()
    torch.set_grad_enabled(True)

    # 7) สรุปเวลา
    elapsed = time.time() - start_time
    print(f"=== Inference done in {elapsed:.2f}s ===")
    if args.save_results:
        ckp.end_background()


def main():
    # ถ้าใช้ --test_only ให้วิ่ง inference-only พร้อม progress bar
    if args.test_only:
        run_inference_only()
        return

    # กรณี training/validation ปกติ
    ckp    = utility.checkpoint(args)
    loader = data.Data(args)
    trainer = Trainer(args, loader, model.Model(args, ckp), loss.Loss(args, ckp), ckp)

    while not trainer.terminate():
        trainer.train()
        trainer.test()
    ckp.done()


if __name__ == '__main__':
    main()
