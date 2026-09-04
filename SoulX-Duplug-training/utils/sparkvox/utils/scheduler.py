import math
import torch

from torch import optim
from torch.optim.lr_scheduler import LambdaLR


class WarmupLR(LambdaLR):
    """Linear warmup and then inverse square root decay."""

    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        warmup_step: int = 0,
        down_step: float = 5e4,
        max_lr: float = 1e-4,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
        **kwargs
    ):
        self.warmup_step = warmup_step
        self.down_step = down_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.alpha = (self.max_lr - 1e-5) / self.warmup_step**2
        super(WarmupLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        init_lr = 1e-5
        s1, s2 = self.warmup_step, self.warmup_step + self.down_step
        if step < s1:
            return init_lr + self.alpha * step**2
        elif s1 <= step < s2:
            return (self.max_lr - self.min_lr) / (s1 - s2) * step + (
                self.min_lr * s1 - self.max_lr * s2
            ) / (s1 - s2)
        else:
            return self.min_lr


class WarmupAnnealSteps(LambdaLR):
    """
    Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(
        self,
        optimizer,
        warmup_step,
        anneal_steps,
        anneal_rate,
        final_lr=None,
        last_epoch=-1,
    ):
        self.warmup_step = warmup_step
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.decay_factor = warmup_step**0.5
        self.final_lr = final_lr
        super(WarmupAnnealSteps, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        step += 1
        lr = (
            min(float(step) ** -0.5, self.warmup_step**-1.5 * step)
            * self.warmup_step**0.5
        )

        for s in self.anneal_steps:
            if step > s:
                lr = lr * self.anneal_rate
        if self.final_lr is not None:
            lr = max(self.final_lr, lr)
        # actually, the lr ratio
        return lr


class WarmupCosineLRSchedule(LambdaLR):
    """LLaMA setting:
    optimizer:
        AdamW:
            lr: 3e-4
            betas: [0.9, 0.95]
            eps: 1e-9
            weight_decay: 0.1
            amsgrad: False

    gradient_clip_val: 1.0

    lr schedule:
        peak_lr = 3e-4
        warmup_steps = 2000
        final_lr_ratio = 0.1

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 2000,
        num_training_steps: int = 400000,
        num_cycles: float = 0.5,
        final_lr_ratio: float = 0.1,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.final_lr_ratio = final_lr_ratio

        super(WarmupCosineLRSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step: int):
        """lr_lambda (function or list): A function which computes a multiplicative factor given an
        integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
        """
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))

        progress = float(step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )

        return max(
            self.final_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )


class ExponentialLRScheduler(LambdaLR):
    """Exponential LR scheduler with a minimum learning rate cap."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.gamma = gamma
        self.min_lr = min_lr
        # Capture initial learning rates
        self.init_lr = optimizer.param_groups[0]["lr"]

        super(ExponentialLRScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch, verbose=verbose
        )

    def lr_lambda(self, step: int):
        return max(self.init_lr * self.gamma**step, self.min_lr) / self.init_lr
