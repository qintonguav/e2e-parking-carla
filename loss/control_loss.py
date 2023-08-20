import torch
from torch import nn
from tool.config import Configuration
import numpy as np


class ControlLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlLoss, self).__init__()
        self.cfg = cfg
        self.ignore_idx = self.cfg.ignore_idx
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

    def forward(self, pred, data):
        pred_control = pred.reshape(-1, pred.shape[-1])
        gt_control = data['gt_control'][:, 1:]
        control_loss = self.ce_loss(pred_control, gt_control)
        return control_loss


class ControlValLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlValLoss, self).__init__()
        self.cfg = cfg
        self.ignore_idx = self.cfg.ignore_idx

        self.valid_token = self.cfg.token_nums - 4
        self.half_token = float(self.valid_token / 2)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        self.l1_loss = nn.SmoothL1Loss()

    def detokenize_acc(self, acc_token):
        if acc_token > self.half_token:
            acc = acc_token / self.half_token - 1
        else:
            acc = -(acc_token / self.half_token - 1)
        return acc

    def detokenize_steer(self, steer_token):
        steer = (steer_token / self.half_token) - 1
        return steer

    def forward(self, pred, data):
        pred_control = pred[:, :-2, :]
        acc_token = pred_control[:, 0::3, :]
        steer_token = pred_control[:, 1::3, :]
        reverse_token = pred_control[:, 2::3, :]

        pred_acc_token = acc_token.softmax(dim=-1)
        pred_acc_token = pred_acc_token.argmax(dim=-1)
        pred_acc_token = pred_acc_token.reshape(-1).to_list()
        pred_acc = [self.detokenize_acc(x) for x in pred_acc_token]
        pred_acc = torch.from_numpy(np.array(pred_acc).astype(np.float32)).cuda()
        gt_acc = data['gt_acc'].reshape(-1)
        acc_val_loss = self.l1_loss(pred_acc, gt_acc)

        pred_steer_token = steer_token.softmax(dim=-1)
        pred_steer_token = pred_steer_token.argmax(dim=-1)
        pred_steer_token = pred_steer_token.reshape(-1).to_list()
        pred_steer = [self.detokenize_steer(x) for x in pred_steer_token]
        pred_steer = torch.from_numpy(np.array(pred_steer).astype(np.float32)).cuda()
        gt_steer = data['gt_steer'].reshape(-1)
        steer_val_loss = self.l1_loss(pred_steer, gt_steer)

        acc_steer_val_loss = (acc_val_loss + steer_val_loss)

        pred_reverse_token = reverse_token.reshape(-1, reverse_token.shape[-1])
        pred_reverse_token = pred_reverse_token.softmax(dim=-1)
        p_no_reverse = torch.sum(pred_reverse_token[:, :101])
        p_reverse = torch.sum(pred_reverse_token[:, 101:])
        pred_reverse = torch.cat([p_no_reverse, p_reverse]).T

        gt_reverse = data['gt_reverse'].reshape(-1)
        reverse_val_loss = self.ce_loss(pred_reverse, gt_reverse)

        return acc_steer_val_loss, reverse_val_loss
