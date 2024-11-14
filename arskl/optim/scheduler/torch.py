import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, \
    MultiStepLR

from ...registry import SCHEDULER


@SCHEDULER.register_module()
class CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, T_max, **kwargs):
        super(CosineAnnealingLR, self).__init__(T_max=T_max, **kwargs)


@SCHEDULER.register_module()
class CosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, **kwargs):
        super(CosineAnnealingWarmRestarts, self).__init__(**kwargs)


@SCHEDULER.register_module()
class OneCycleLR(OneCycleLR):
    def __init__(self, **kwargs):
        super(OneCycleLR, self).__init__(**kwargs)


@SCHEDULER.register_module()
class ReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super(ReduceLROnPlateau, self).__init__(**kwargs)


@SCHEDULER.register_module()
class MultiStepLR(MultiStepLR):
    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(**kwargs)

@SCHEDULER.register_module()
class LinearCosineScheduler:
    def __init__(self, optimizer, initial_lr, max_lr, final_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear increase during warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay after warmup
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_in_decay = self.current_epoch - self.warmup_epochs
            lr = self.final_lr + (self.max_lr - self.final_lr) * 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'initial_lr': self.initial_lr,
            'max_lr': self.max_lr,
            'final_lr': self.final_lr,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.initial_lr = state_dict['initial_lr']
        self.max_lr = state_dict['max_lr']
        self.final_lr = state_dict['final_lr']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']