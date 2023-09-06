import os
import torch

from datetime import datetime


class Configuration:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = None
    log_dir = None
    checkpoint_dir = None
    log_every_n_steps = None
    check_val_every_n_epoch = None

    epochs = None
    learning_rate = None
    weight_decay = None
    batch_size = None

    training_map = None
    validation_map = None
    future_frame_nums = None
    hist_frame_nums = None
    token_nums = None
    image_crop = None

    bev_encoder_in_channel = None
    bev_encoder_out_channel = None

    bev_x_bound = None
    bev_y_bound = None
    bev_z_bound = None
    d_bound = None
    final_dim = None
    bev_down_sample = None
    use_depth_distribution = None
    backbone = None

    seg_classes = None
    seg_vehicle_weights = None

    tf_en_dim = None
    tf_en_heads = None
    tf_en_layers = None
    tf_en_dropout = None
    tf_en_bev_length = None
    tf_en_motion_length = None

    tf_de_dim = None
    tf_de_heads = None
    tf_de_layers = None
    tf_de_dropout = None
    tf_de_tgt_dim = None


def get_cfg(cfg_yaml: dict):
    today = datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)

    config = cfg_yaml['parking_model']
    cfg = Configuration()

    cfg.data_dir = config['data_dir']
    cfg.log_dir = os.path.join(config['log_dir'], exp_name)
    cfg.checkpoint_dir = os.path.join(config['checkpoint_dir'], exp_name)
    cfg.log_every_n_steps = config['log_every_n_steps']
    cfg.check_val_every_n_epoch = config['check_val_every_n_epoch']

    cfg.epochs = config['epochs']
    cfg.learning_rate = config['learning_rate']
    cfg.weight_decay = config['weight_decay']
    cfg.batch_size = config['batch_size']

    cfg.training_map = config['training_map']
    cfg.validation_map = config['validation_map']
    cfg.future_frame_nums = config['future_frame_nums']
    cfg.hist_frame_nums = config['hist_frame_nums']
    cfg.token_nums = config['token_nums']
    cfg.image_crop = config['image_crop']

    cfg.bev_encoder_in_channel = config['bev_encoder_in_channel']
    cfg.bev_encoder_out_channel = config['bev_encoder_out_channel']

    cfg.bev_x_bound = config['bev_x_bound']
    cfg.bev_y_bound = config['bev_y_bound']
    cfg.bev_z_bound = config['bev_z_bound']
    cfg.d_bound = config['d_bound']
    cfg.final_dim = config['final_dim']
    cfg.bev_down_sample = config['bev_down_sample']
    cfg.use_depth_distribution = config['use_depth_distribution']
    cfg.backbone = config["backbone"]

    cfg.seg_classes = config['seg_classes']
    cfg.seg_vehicle_weights = config['seg_vehicle_weights']

    cfg.tf_en_dim = config['tf_en_dim']
    cfg.tf_en_heads = config['tf_en_heads']
    cfg.tf_en_layers = config['tf_en_layers']
    cfg.tf_en_dropout = config['tf_en_dropout']
    cfg.tf_en_bev_length = config['tf_en_bev_length']
    cfg.tf_en_motion_length = config['tf_en_motion_length']

    cfg.tf_de_dim = config['tf_de_dim']
    cfg.tf_de_heads = config['tf_de_heads']
    cfg.tf_de_layers = config['tf_de_layers']
    cfg.tf_de_dropout = config['tf_de_dropout']
    cfg.tf_de_tgt_dim = config['tf_de_tgt_dim']

    return cfg
