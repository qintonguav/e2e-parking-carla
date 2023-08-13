import torch
from torch import nn
from model.cam_encoder import CamEncoder
from tool.config import Configuration


class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super(BevModel, self).__init__()
        self.cfg = cfg
        self.frustum = self.create_frustum()
        self.cam_encode = CamEncoder()

    def calc_bev_params(self):
        pass

    def create_frustum(self):
        pass

    def get_geometry(self, x, intrins, extrins):
        pass

    def encoder_froward(self, x):
        pass

    def proj_bev_feature(self, geom_feats, x):
        pass

    def calc_bev_feature(self, x, intrins, extrins):
        geom = self.get_geometry(x, intrins, extrins)
        x = self.encoder_froward(x)
        bev_feature, pred_depth = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, x, intrins, extrins):
        bev_feature, pred_depth = self.calc_bev_feature(x, intrins, extrins)
        return bev_feature, pred_depth

