from typing import List

import torch
import torch.nn.functional as F
import torch.utils.data
from einops import rearrange, reduce
from einops.layers.torch import Reduce, Rearrange
from torch import nn
from torchvision.ops.misc import SqueezeExcitation

from ..registry import MODEL


def delete_keys_from_dict(d, keys_to_delete):
    d_copy = d.copy()
    for key in keys_to_delete:
        if key in d_copy:
            del d_copy[key]

    return d_copy


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndimension() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor
        return x


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ta = nn.Sequential(
            Reduce('b c t v -> b 1 t', 'mean'),
            nn.Conv1d(1, 1, kernel_size=1, bias=False),
            Rearrange('b 1 t -> b t 1 1'),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        x1 = rearrange(x, 'b c t v->b t c v')
        out = torch.multiply(x1, self.ta(x))
        out = rearrange(out, 'b t c v->b c t v')
        return out


class ChannelTemporalAttention(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ta = TemporalAttention()
        self.se = SqueezeExcitation(planes, int(planes // 4), nn.SiLU)

    def forward(self, x):
        x = self.se(x) + self.ta(x)
        return x


class ChannelTemporalAttention_2(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ta = nn.Sequential(
            Reduce('b c t v -> b 1 t', 'mean'),
            nn.Conv1d(1, 1, kernel_size=1, bias=False),
            Rearrange('b 1 t -> b 1 t 1'),
            nn.Hardsigmoid(),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, int(planes // 4), kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(int(planes // 4), planes, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * (self.se(x) + self.ta(x))
        return x


class BottleNeckBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            exp_ratio: int = 4,
            inflate=True,
            drop_p: float = .0,
    ):
        super().__init__()
        planes = _make_divisible(in_features * exp_ratio, divisor=17)
        self.block = nn.Sequential(
            nn.Conv2d(in_features, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),

            nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=(0, 1), groups=planes, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                      bias=False) if inflate else nn.Identity(),
            nn.BatchNorm2d(planes) if inflate else nn.Identity(),

            nn.Hardswish(True),
            ChannelTemporalAttention_2(planes),

            nn.Conv2d(planes, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()

    def forward(self, x):
        res = x
        x = self.block(x)
        x = self.drop_path(x)
        x = x + res
        return x


class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            stride: int,
    ):
        super().__init__()
        kernel_size = tuple((2 * s - 1) for s in stride)
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_features,
                      bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SparseStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, stride=1, kernel_size=(1, 7)):
        padding = tuple((k - 1) // 2 for k in kernel_size)
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_features),
        )


class SparseStage(nn.Sequential):
    def __init__(
            self, in_features: int, out_features: int, depth: int, stride, exp_ratio: int, inflate=True, **kwargs
    ):
        super().__init__(
            # downsample is done here
            DownsampleBlock(in_features, out_features, stride),
            *[
                BottleNeckBlock(out_features, out_features, inflate=inflate, exp_ratio=exp_ratio, **kwargs)
                for _ in range(depth - 1)
            ],
        )


class SparseEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stem_features: int,
            depths: List[int],
            widths: List[int],
            inflates=[0, 1, 1],
            temporal_strides: List[int] = [1, 1, 2],
            spatial_strides: List[int] = [2, 2, 2],
            drop_p: float = .0
    ):
        super().__init__()
        self.stem = SparseStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))

        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                SparseStage(stem_features, widths[0], depths[0], inflate=inflates[0],
                            stride=(temporal_strides[0], spatial_strides[0]),
                            drop_p=drop_probs[0]),
                *[
                    SparseStage(in_features, out_features, depth, stride=(
                        temporal_stride, spatial_stride),
                                inflate=inflate, drop_p=drop_p)
                    for (in_features, out_features), depth, temporal_stride, spatial_stride, inflate, drop_p
                    in zip(
                        in_out_widths, depths[1:], temporal_strides[1:], spatial_strides[1:],
                        inflates[1:], drop_probs[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


@MODEL.register_module()
class PoseConv2D(nn.Module):

    def __init__(self, thw, num_classes=60, in_chans=17, load_path=None, mismatch=[],
                 freeze_fraction=0.0, is_test=False):
        super().__init__()
        init_chans = int(in_chans * 3)
        self.stem = SparseStem(in_chans, init_chans)
        self.stage1 = SparseStage(init_chans, init_chans * 2, 4, (1, 2), inflate=False, drop_p=0.2, exp_ratio=2)
        self.stage2 = SparseStage(init_chans * 2, init_chans * 4, 6, (1, 4), inflate=True, drop_p=0.2, exp_ratio=2)
        self.stage3 = SparseStage(init_chans * 4, init_chans * 4 * 2, 3, (2, 4), inflate=True, drop_p=0.2, exp_ratio=2)
        self.head = nn.Sequential(
            Reduce('b c t v->b c', 'mean'),
            nn.Dropout(),
            nn.Linear(init_chans * 4 * 2, num_classes)
        )
        self.load_path = load_path
        self.mismatch = mismatch
        self.freeze_fraction = freeze_fraction
        self.is_test = is_test

    def forward(self, x):
        B, S, C, T, V = x.shape
        x = rearrange(x, 'b s c t v->(b s) c t v')
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        x = F.softmax(x, dim=1) if self.is_test else x
        x = reduce(x, '(b s) c->b c', 'mean', s=S)
        return x

    def init_weights(self):
        if self.load_path is not None:
            self.load_pretrain()
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def load_pretrain(self):
        state_dict = torch.load(self.load_path)
        state_dict = state_dict['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '', 1)
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        state_dict = delete_keys_from_dict(state_dict, self.mismatch)
        self.load_state_dict(state_dict=state_dict, strict=True)
        param_keys_list = list(state_dict)
        num_to_freeze = int(len(param_keys_list) * self.freeze_fraction)
        keys_to_freeze = set(param_keys_list[:num_to_freeze])
        print(f"Freezing {num_to_freeze} parameters({self.freeze_fraction * 100}%):")
        for name, param in self.named_parameters():
            if name in keys_to_freeze:
                param.requires_grad = False
                print(f"- {name}")
