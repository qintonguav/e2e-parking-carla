import json
import math
import random
import logging
import sys
import pathlib
from datetime import datetime
import numpy as np
import cv2

from carla_data_generator import parking_position
from carla_data_generator.world import World


class NetworkEvaluator:
    def __init__(self, carla_world, args):
        self._seed = args.random_seed
        self._init_seed = args.random_seed
        random.seed(args.random_seed)

        self._world = World(carla_world, args)

        # only eva odd slot: 1, 3, 5, ..., 15
        # 2-1, index 15 + 1 = 16
        self._parking_goal_index = 16
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator = parking_position.EgoPosTown04()

        now = datetime.now()
        result_dir = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self._eva_result_path = pathlib.Path(args.save_path) / result_dir
        self._eva_result_path.mkdir(parents=True, exist_ok=False)

        self._render_bev = args.render_bev

        self._eva_epochs = args.eva_epochs
        self._slot_nums = args.slot_nums
        self._parking_nums = args.parking_nums
        self._eva_epoch_idx = 0
        self._slot_idx = 0
        self._parking_idx = 0

        # parking goal diff
        self._position_diff = 10000         # init with large value
        self._orientation_diff = 10000
        self._position_x_diff = 10000
        self._position_y_diff = 10000
        self._max_position_x_diff = 1.0     # meter
        self._max_orientation_diff = 10.0   # degree

        # 1s = 30HZ for our carla setting
        second = 30

        # metric frames
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0
        self._num_frames_in_goal_needed = 2 * second      # 2s
        self._num_frames_nearby_goal_needed = 2 * second  # 2s
        self._num_frames_outside_needed = 20 * second     # 20s
        self._num_frames_total_needed = 30 * second       # 30s

        # metric for 1 slot
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0
        self._position_error = []
        self._orientation_error = []
        self._inference_time = []
        self._parking_time = []

        # metric rate for 16 slot
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_inference_time = []
        self._average_parking_time = []

        # In paper: NTR = NTSR + NTFR, TR = TR + OR
        self._metric_names = {
            "target_success_rate": "TSR",
            "target_fail_rate": "TFR",
            "no_target_success_rate": "NTSR",
            "no_target_fail_rate": "NTFR",
            "collision_rate": "CR",
            "outbound_rate": "OR",
            "timeout_rate": "TR",
            "average_position_error": "APE",
            "average_orientation_error": "AOE",
            "average_inference_time": "AIT",
            "average_parking_time": "APT",
        }

        self._ego_transform = None
        self._eva_parking_goal = None

        # for parking_agent init pos
        self._agent_need_init = None

        self.init()
        self.start_eva_epoch()

    def init(self):
        pass

    def tick(self):
        pass

    def start_eva_epoch(self):
        pass

    def start_next_parking(self):
        pass

    def start_next_slot(self):
        pass

    def is_complete_slot(self):
        pass

    def is_complete_epoch(self):
        pass

    def clear_metric_num(self):
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0
        self._position_error = []
        self._orientation_error = []
        self._inference_time = []
        self._parking_time = []

    def clear_metric_frame(self):
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0

    def clear_metric_rate(self):
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_inference_time = []
        self._average_parking_time = []

    def soft_destroy(self):
        self._parking_idx = 0
        self.clear_metric_num()
        self.clear_metric_frame()
        self._world.soft_destroy()

    def eva_check_goal(self):
        pass

    def check_success_slot(self):
        pass

    def check_fail_slot(self):
        pass

    def save_csv(self):
        pass

    def save_epoch_metric(self):
        pass

    def save_average_metric(self):
        pass

