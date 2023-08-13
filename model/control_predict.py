import torch
from torch import nn
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlPredict, self).__init__()
        self.cfg = cfg

        self.layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=self.cfg.tf_de_layers)

    def forward(self, x):
        pass

    def predict(self, x):
        pass
