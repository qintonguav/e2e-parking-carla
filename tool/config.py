from datetime import datetime
import os


class Configuration:
    data_dir = None
    log_dir = None
    ckpt_dir = None
    log_every_n_step = None
    check_val_every_n_train = None

    learning_rate = None
    device = None

    bev_encoder_in_channel = None
    bev_encoder_out_channel = None

    bev_x_bound = None
    bev_y_bound = None
    bev_z_bound = None
    d_bound = None
    final_dim = None
    bev_down_sample = None
    use_depth_distribution = None

    seg_classes = None
    seg_vehicle_weights = None
    ignore_index = None

    token_nums = None
    bos_idx = None
    eos_idx = None
    pad_idx = None

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


def get_cfg(cfg_yaml: dict):
    today = datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)

    config = cfg_yaml['parking_model']
    cfg = Configuration()

    cfg.data_dir = config['data_dir']
    cfg.log_dir = os.path.join(config['log_dir'], exp_name)
    cfg.ckpt_dir = os.path.join(config['ckpt_dir'], exp_name)
    cfg.log_every_n_step = config['log_every_n_step']
    cfg.check_val_every_n_train = config['check_val_every_n_train']

    cfg.learning_rate = config['learning_rate']
    cfg.device = config['device']

    cfg.bev_encoder_in_channel = config['bev_encoder_in_channel']
    cfg.bev_encoder_out_channel = config['bev_encoder_out_channel']

    cfg.bev_x_bound = config['bev_x_bound']
    cfg.bev_y_bound = config['bev_y_bound']
    cfg.bev_z_bound = config['bev_z_bound']
    cfg.d_bound = config['d_bound']
    cfg.final_dim = config['final_dim']
    cfg.bev_down_sample = config['bev_down_sample']
    cfg.use_depth_distribution = config['use_depth_distribution']

    cfg.seg_classes = config['seg_classes']
    cfg.seg_vehicle_weights = config['seg_vehicle_weights']
    cfg.ignore_index = config['ignore_index']

    cfg.token_nums = config['token_nums']
    cfg.bos_idx = config['bos_idx']
    cfg.eos_idx = config['eos_idx']
    cfg.pad_idx = config['pad_idx']

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

    return cfg



