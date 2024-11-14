import numpy as np
import torch
from lightning import LightningModule
from torchmetrics.functional import accuracy

from ..model import build_model
from ..optim.optim import build_optim
from ..optim.scheduler import build_scheduler
from ..registry import LEARNER


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@LEARNER.register_module()
class Learner_Skl(LightningModule):

    def __init__(self, model_cfg, optim_cfg, scheduler_cfg, hyper_cfg):
        super().__init__()
        self.lr = optim_cfg['lr']
        self.model = build_model(model_cfg)
        if hyper_cfg['compile']:
            self.model = torch.compile(self.model)
        self.model.init_weights()
        self.num_classes = hyper_cfg['num_classes']
        self.save_hyperparameters(hyper_cfg)
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.check_val_every_n_epoch = hyper_cfg['check_val_every_n_epoch']
        self.base_loss = FocalLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch['vectors'], batch['label']
        y = torch.squeeze(y)
        x, y_a, y_b, lam = mixup_data(x, y, alpha=0.6)

        y_hat = self.model(x)
        loss = mixup_criterion(self.base_loss, y_hat, y_a, y_b, lam)
        y = y_a if lam > 0.5 else y_b
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        metrics = {"acc": acc, "loss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['vectors'], batch['label']
        y_hat = self.model(x)
        loss = self.base_loss(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes, top_k=1)
        acc5 = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes, top_k=5)
        metrics = {"vacc": acc, "vacc5": acc5, "vloss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)


    def test_step(self, batch, batch_idx):
        x, y = batch['vectors'], batch['label']
        y_hat = self.model(x)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        acc5 = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes, top_k=5)
        metrics = {"acc": acc, "acc5": acc5}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        self.optim_cfg['params'] = self.parameters()
        optim = build_optim(self.optim_cfg)
        self.scheduler_cfg['optimizer'] = optim
        scheduler = build_scheduler(self.scheduler_cfg)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "vacc",
            "strict": True,
            "name": None,
        }
        optim_dict = {'optimizer': optim, 'lr_scheduler': lr_scheduler_config}
        return optim_dict

    def lr_scheduler_step(self, scheduler, optimizer_idx, metrics=None):
        scheduler.step()
