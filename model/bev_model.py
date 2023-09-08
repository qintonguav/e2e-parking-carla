import torch

from torch import nn
from model.cam_encoder import CamEncoder
from tool.config import Configuration
from tool.geometry import VoxelsSumming, calculate_birds_eye_view_parameters


class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        bev_res, bev_start_pos, bev_dim = calculate_birds_eye_view_parameters(self.cfg.bev_x_bound,
                                                                              self.cfg.bev_y_bound,
                                                                              self.cfg.bev_z_bound)
        self.bev_res = nn.Parameter(bev_res, requires_grad=False)
        self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False)
        self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)

        self.down_sample = self.cfg.bev_down_sample

        self.frustum = self.create_frustum()
        self.depth_channel, _, _, _ = self.frustum.shape
        self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)

    def create_frustum(self):
        h, w = self.cfg.final_dim
        down_sample_h, down_sample_w = h // self.down_sample, w // self.down_sample

        depth_grid = torch.arange(*self.cfg.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)
        depth_slice = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=torch.float)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinsics, extrinsics):
        extrinsics = torch.inverse(extrinsics).cuda()
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        b, n, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine_transform = rotation.matmul(torch.inverse(intrinsics)).cuda()
        points = combine_transform.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(b, n, 1, 1, 1, 3)

        return points

    def encoder_forward(self, images):
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)
        x, depth = self.cam_encoder(images)

        depth_prob = depth.softmax(dim=1)
        if self.cfg.use_depth_distribution:
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth_prob

    def proj_bev_feature(self, geom, image_feature):
        batch, n, d, h, w, c = image_feature.shape
        output = torch.zeros((batch, c, self.bev_dim[0], self.bev_dim[1]),
                             dtype=torch.float, device=image_feature.device)
        N = n * d * h * w
        for b in range(batch):
            image_feature_b = image_feature[b]
            geom_b = geom[b]

            x_b = image_feature_b.reshape(N, c)

            geom_b = ((geom_b - (self.bev_start_pos - self.bev_res / 2.0)) / self.bev_res)
            geom_b = geom_b.view(N, 3).long()

            mask = ((geom_b[:, 0] >= 0) & (geom_b[:, 0] < self.bev_dim[0])
                    & (geom_b[:, 1] >= 0) & (geom_b[:, 1] < self.bev_dim[1])
                    & (geom_b[:, 2] >= 0) & (geom_b[:, 2] < self.bev_dim[2]))
            x_b = x_b[mask]
            geom_b = geom_b[mask]

            ranks = ((geom_b[:, 0] * (self.bev_dim[1] * self.bev_dim[2])
                     + geom_b[:, 1] * self.bev_dim[2]) + geom_b[:, 2])
            sorts = ranks.argsort()
            x_b, geom_b, ranks = x_b[sorts], geom_b[sorts], ranks[sorts]

            x_b, geom_b = VoxelsSumming.apply(x_b, geom_b, ranks)

            bev_feature = torch.zeros((self.bev_dim[2], self.bev_dim[0], self.bev_dim[1], c),
                                      device=image_feature_b.device)
            bev_feature[geom_b[:, 2], geom_b[:, 0], geom_b[:, 1]] = x_b
            tmp_bev_feature = bev_feature.permute((0, 3, 1, 2)).squeeze(0)
            output[b] = tmp_bev_feature

        return output

    def calc_bev_feature(self, images, intrinsics, extrinsics):
        geom = self.get_geometry(intrinsics, extrinsics)
        x, pred_depth = self.encoder_forward(images)
        bev_feature = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, images, intrinsics, extrinsics):
        bev_feature, pred_depth = self.calc_bev_feature(images, intrinsics, extrinsics)
        return bev_feature.squeeze(1), pred_depth

