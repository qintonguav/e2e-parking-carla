import torch


def add_target_bev(cfg, bev_feature, target_point):
    b, c, h, w = bev_feature.shape
    bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(cfg.device, non_blocking=True)

    x_pixel = (h / 2 + target_point[:, 0] / cfg.bev_x_bound[2]).unsqueeze(0).T.int()
    y_pixel = (w / 2 + target_point[:, 1] / cfg.bev_y_bound[2]).unsqueeze(0).T.int()
    target_point = torch.cat([x_pixel, y_pixel], dim=1)

    noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
    target_point += noise

    for batch in range(b):
        bev_target_batch = bev_target[batch][0]
        target_point_batch = target_point[batch]
        bev_target_batch[target_point_batch[0] - 4:target_point[0] + 4,
                         target_point_batch[1] - 4:target_point[1] + 4] = 1.0

    bev_feature = torch.cat([bev_feature, bev_target], dim=1)
    return bev_feature
