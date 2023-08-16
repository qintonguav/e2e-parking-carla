from torch import nn
from tool.config import Configuration
import torch.nn.functional as F


class DepthLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(DepthLoss, self).__init__()
        self.cfg = cfg

    def forward(self, pred, data):
        pass
