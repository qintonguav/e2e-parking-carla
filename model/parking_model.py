import torch
from torch import nn
from tool.config import Configuration


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ParkingModel, self).__init__()
        self.cfg = cfg

    def forward(self, data):
        pass

    def predict(self, data):
        pass
