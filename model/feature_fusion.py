import torch
from torch import nn
from tool.config import Configuration


class FeatureFusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super(FeatureFusion, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        pass
