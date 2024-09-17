import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_step=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr
                + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
