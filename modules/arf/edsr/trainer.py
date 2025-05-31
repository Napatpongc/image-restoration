import math
from decimal import Decimal
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import utility

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args       = args
        self.scale      = args.scale
        self.ckp        = ckp
        self.loader_train = loader.loader_train
        self.loader_test  = loader.loader_test
        self.model      = my_model
        self.loss       = my_loss
        self.optimizer  = utility.make_optimizer(args, self.model)

        # กำหนด device หนึ่งครั้ง
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not args.cpu
            else 'cpu'
        )
        self.model.to(self.device)

        # เตรียม state ถ้า resume
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(f'[Epoch {epoch}]\tLearning rate: {Decimal(lr):.2e}')
        self.loss.start_log()
        self.model.train()

        # TEMP: ถ้าใช้ scale เดียว
        self.loader_train.dataset.set_scale(0)

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            # ย้ายข้อมูลไป device / half
            lr = lr.to(self.device).half() if self.args.precision=='half' else lr.to(self.device)
            hr = hr.to(self.device).half() if self.args.precision=='half' else hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                t_data, t_model = 0.0, 0.0  # ถ้าต้องการวัดเวลาให้ใส่ timer เพิ่ม
                self.ckp.write_log(
                    f'[{(batch+1)*self.args.batch_size}/{len(self.loader_train.dataset)}]\t'
                    f'{self.loss.display_loss(batch)}'
                )

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        """
        รัน validation/test ในโหมด training loop ดั้งเดิม
        """
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr = lr.to(self.device).half() if self.args.precision=='half' else lr.to(self.device)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    # คำนวณ PSNR ถ้าอยากเก็บ log
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr.to(self.device), scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], [sr], scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    f'[{d.dataset.name} x{scale}]\t'
                    f'PSNR: {self.ckp.log[-1, idx_data, idx_scale]:.3f} '
                    f'(Best: {best[0][idx_data, idx_scale]:.3f} @epoch {best[1][idx_data, idx_scale]+1})'
                )

        self.ckp.write_log(f'Total validation time: {utility.timer().toc():.2f}s\n')
        torch.set_grad_enabled(True)

    def terminate(self):
        if self.args.test_only:
            # ถ้าเรียกจาก main.py โหมด test_only จะไม่เข้าถึงตรงนี้
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
