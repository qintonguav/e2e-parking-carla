import torch
from torch import nn
from tool.config import Configuration


class SegmentationHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegmentationHead, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        pass
