import torch
from torch import nn
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlPredict, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        pass

    def predict(self, x):
        pass
