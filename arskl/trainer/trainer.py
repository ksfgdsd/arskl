from lightning import Trainer
from lightning.pytorch.callbacks import RichModelSummary, LearningRateMonitor, RichProgressBar, \
    ModelCheckpoint

from ..registry import TRAINER

profiler = None


@TRAINER.register_module()
class Trainer(Trainer):
    def __init__(self, ckpt_cfg, **kwargs):
        super().__init__(callbacks=[
            RichModelSummary(),
            LearningRateMonitor(),
            RichProgressBar(),
            ModelCheckpoint(**ckpt_cfg),
        ], **kwargs, profiler=profiler)
