import torch
from torch import nn
from model.cam_encoder import CamEncoder
from tool.config import Configuration
from tool.voxel_summing import VoxelSumming


class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super(BevModel, self).__init__()

        self.cfg = cfg

        self.downsample = 16

        dx, bx, nx = self.calc_bev_params(self.cfg['bev_x_bound'], self.cfg['bev_y_bound'], self.cfg['bev_z_bound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.frustum = self.create_frustum()
        self.cam_encoder = CamEncoder()

    def calc_bev_params(self, bev_x_bound, bev_y_bound, bev_z_bound):
        dx = torch.Tensor([row[2] for row in [bev_x_bound, bev_y_bound, bev_z_bound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [bev_x_bound, bev_y_bound, bev_z_bound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [bev_x_bound, bev_y_bound, bev_z_bound]])
        return dx, bx, nx

    def create_frustum(self):
        """
        Todo
        """
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, x, intrins, extrins):
        """
        Todo
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def encoder_froward(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.cam_encoder(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def proj_bev_feature(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        x, geom_feats = VoxelSumming.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def calc_bev_feature(self, x, intrins, extrins):
        geom = self.get_geometry(x, intrins, extrins)
        x = self.encoder_froward(x)
        bev_feature, pred_depth = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, x, intrins, extrins):
        bev_feature, pred_depth = self.calc_bev_feature(x, intrins, extrins)
        return bev_feature, pred_depth

