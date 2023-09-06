import torch
import torch.nn.functional as F

from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from tool.config import Configuration


class DepthLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        self.d_bound = self.cfg.d_bound
        self.down_sample_factor = self.cfg.bev_down_sample
        self.depth_channels = int((self.cfg.d_bound[1] - self.cfg.d_bound[0]) / self.cfg.d_bound[2])

    def forward(self, depth_preds, depth_labels):
        depth_labels = self.get_down_sampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

    def get_down_sampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.down_sample_factor, self.down_sample_factor,
                                   W // self.down_sample_factor, self.down_sample_factor, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.down_sample_factor * self.down_sample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.down_sample_factor, W // self.down_sample_factor)

        gt_depths = (gt_depths - (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2]
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()
