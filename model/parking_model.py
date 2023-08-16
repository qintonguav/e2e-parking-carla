import torch
from torch import nn
from tool.config import Configuration
from model.bev_model import BevModel, BevEncoder, FeatureFusion, ControlPredict, SegmentationHead
from tool.geometry import add_target_bev


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ParkingModel, self).__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        self.segmentation_head = SegmentationHead(self.cfg)

    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)

        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        bev_feature, bev_target = add_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion)

        pred_segmentation = self.segmentation_head(fuse_feature)

        return fuse_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        return pred_control, pred_segmentation, pred_depth

    def predict(self, data):
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        for i in range(3):
            pred_control = self.control_predict(fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        return pred_multi_controls, pred_segmentation, pred_depth, bev_target
