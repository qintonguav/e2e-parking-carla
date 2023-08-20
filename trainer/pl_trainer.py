import torch
import torch.nn as nn
import pytorch_lightning as pl
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss


class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModule, self).__init__()
        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        self.segmentation_loss_func = SegmentationLoss(self.cfg)

        self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = None

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_control, pred_segmentation, pred_depth = self.parking_model(batch)

        control_loss = self.control_loss_func(pred_control, batch)
        loss_dict.update({
            "control_loss": control_loss
        })

        segmentation_loss = self.segmentation_loss_func(pred_control, batch['segmentation'])
        loss_dict.update({
            "segmentation_loss": segmentation_loss
        })

        depth_loss = self.depth_loss_func(pred_control, batch['depth'])
        loss_dict.update({
            "depth_loss": depth_loss
        })

        train_loss = sum(loss_dict.values())
        loss_dict.update({
            "train_loss": train_loss
        })

        self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation')
        self.log_depth(pred_depth, batch['depth'], 'depth')

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_control, pred_segmentation, pred_depth = self.parking_model(batch)

        acc_steer_val_loss, reverse_val_loss = self.control_val_loss_func(pred_control, batch)
        val_loss_dict.update({
            "acc_steer_val_loss": acc_steer_val_loss,
            "reverse_val_loss": reverse_val_loss
        })

        segmentation_val_loss = self.segmentation_loss_func(pred_control, batch['segmentation'])
        val_loss_dict.update({
            "segmentation_val_loss": segmentation_val_loss
        })

        depth_val_loss = self.depth_loss_func(pred_control, batch['depth'])
        val_loss_dict.update({
            "depth_val_loss": depth_val_loss
        })

        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })

        self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation_val')
        self.log_depth(pred_depth, batch['depth'], 'depth_val')

        return val_loss

    def configure_optimizers(self):
        pass

    def log_segmentation(self, pred_segmentation, gt_segmentation, name):
        pass

    def log_depth(self, pred_depth, gt_depth, name):
        pass
