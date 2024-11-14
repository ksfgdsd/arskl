import argparse

from lightning.pytorch import seed_everything
from mmcv import Config

from arskl.dataset.builder import build_dataset
from arskl.learner.builder import build_learner
from arskl.trainer.builder import build_trainer

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
seed_everything(seed=2024)
if cfg.var['verbose']:
    print(cfg.pretty_text)
learner = build_learner(cfg.learner)
data = build_dataset(cfg.dataset)
trainer = build_trainer(cfg.trainer)
if cfg.test['is_test']:
    trainer.test(learner, data.test_dataloader(), cfg.test['ckpt_path'])
elif cfg.reload['is_reload']:
    trainer.fit(learner, data, ckpt_path=cfg.reload['ckpt_path'])
else:
    trainer.fit(learner, data)
