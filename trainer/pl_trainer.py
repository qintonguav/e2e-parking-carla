import torch
import pytorch_lightning as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, ModelSummary
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from model.parking_model import ParkingModel


def setup_callbacks(cfg):
    callbacks = []

    ckpt_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=3,
                                    mode='min',
                                    filename='E2EParking-{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    callbacks.append(ckpt_callback)

    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)

    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    return callbacks


class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        self.segmentation_loss_func = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.seg_vehicle_weights)
        )

        self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = ParkingModel(self.cfg)

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_control, pred_segmentation, pred_depth = self.parking_model(batch)

        control_loss = self.control_loss_func(pred_control, batch)
        loss_dict.update({
            "control_loss": control_loss
        })

        segmentation_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        loss_dict.update({
            "segmentation_loss": segmentation_loss
        })

        depth_loss = self.depth_loss_func(pred_depth, batch['depth'])
        loss_dict.update({
            "depth_loss": depth_loss
        })

        train_loss = sum(loss_dict.values())
        loss_dict.update({
            "train_loss": train_loss
        })

        self.log_dict(loss_dict)
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

        segmentation_val_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        val_loss_dict.update({
            "segmentation_val_loss": segmentation_val_loss
        })

        depth_val_loss = self.depth_loss_func(pred_depth, batch['depth'])
        val_loss_dict.update({
            "depth_val_loss": depth_val_loss
        })

        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })

        self.log_dict(val_loss_dict)
        self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation_val')
        self.log_depth(pred_depth, batch['depth'], 'depth_val')

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.learning_rate,
                                     weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def log_segmentation(self, pred_segmentation, gt_segmentation, name):
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].set_title("GT Seg")
        # ax[1].set_title("Pred Seg")
        #
        # pred_segmentation = pred_segmentation[0]
        # pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        # pred_segmentation = pred_segmentation.detach().cpu().numpy()
        # pred_segmentation[pred_segmentation == 1] = 128
        # pred_segmentation[pred_segmentation == 2] = 255
        # pred_seg_img = pred_segmentation[0, :, :][::-1]
        #
        # gt_segmentation = gt_segmentation[0]
        # gt_segmentation[gt_segmentation == 1] = 128
        # gt_segmentation[gt_segmentation == 2] = 255
        # gt_segmentation = gt_segmentation.detach().cpu().numpy()
        # gt_seg_img = gt_segmentation[0, :, :][::-1]
        #
        # norm = mpl.colors.Normalize(vmin=0, vmax=255)
        # ax[0].imshow(gt_seg_img, norm=norm)
        # ax[1].imshow(pred_seg_img, norm=norm)
        #
        # tensorboard = self.logger.experiment
        # tensorboard.add_figure(figure=fig, tag=name)
        # plt.close(fig)
        pass

    def log_depth(self, pred_depth, gt_depth, name):
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].set_title("GT Depth")
        # ax[1].set_title("Pred Depth")

        # pred_depth = pred_depth[1]
        # pred_depth = torch.softmax(pred_depth, dim=0)
        # pred_depth = pred_depth.detach().cpu().numpy()
        # pred_depth = pred_depth * self.cfg.d_bound[2] + self.cfg.d_bound[0]
        #
        # gt_depth = gt_depth[0][1]
        # gt_depth = gt_depth.detach().cpu().numpy()
        #
        # ax[0].imshow(gt_depth)
        # ax[1].imshow(pred_depth)
        #
        # tensorboard = self.logger.experiment
        # tensorboard.add_figure(figure=fig, tag=name)
        # plt.close(fig)
        pass
