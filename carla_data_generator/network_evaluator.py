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
        self._eva_task_nums = args.eva_task_nums
        self._eva_parking_nums = args.eva_parking_nums
        self._eva_epoch_idx = 0
        self._eva_task_idx = 0
        self._eva_parking_idx = 0

        # parking goal diff, init with max float value
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = sys.float_info.max

        self._goal_reach_x_diff = 1.0              # meter
        self._goal_reach_y_diff = 0.6              # meter
        self._goal_reach_orientation_diff = 10.0   # degree

        # 1s = 30HZ for our carla setting
        second = 30

        # metric frames
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_nearby_no_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0
        self._num_frames_in_goal_needed = 2 * second         # 2s
        self._num_frames_nearby_goal_needed = 2 * second     # 2s
        self._num_frames_nearby_no_goal_needed = 2 * second  # 2s
        self._num_frames_outside_needed = 20 * second        # 20s
        self._num_frames_total_needed = 30 * second          # 30s

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
        self._agent_need_init = None
        self._start_time = None

        self.init()
        self.start_eva_epoch()

    def init(self):
        self._ego_transform = self._ego_transform_generator.get_init_ego_transform()
        self._world.init_ego_vehicle(self._ego_transform)
        self._world.init_sensors()
        self._world.next_weather()

    def tick(self, clock):
        # update diff to world
        self._world.distance_diff_to_goal = self._distance_diff_to_goal
        self._world.rotation_diff_to_goal = self._orientation_diff_to_goal
        self._world.x_diff_to_goal = self._x_diff_to_goal
        self._world.y_diff_to_goal = self._y_diff_to_goal

        self._num_frames_total += 1

        # detect collision
        is_collision = self._world.tick(clock, self._parking_goal_index)
        if is_collision:
            self._collision_nums += 1
            self.start_next_parking()
            return

        # detect timeout
        if self._num_frames_total >= self._num_frames_total_needed:
            self._num_frames_total += 1
            self.start_next_parking()
            return

        # detect out of bound
        ego_loc = self._world.player.get_location()
        if self.is_out_of_bound(ego_loc):
            self._outbound_nums += 1
            self.start_next_parking()
            return

        self.eva_check_goal()

    def start_eva_epoch(self):
        self._start_time = datetime.now()
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
        self._eva_parking_idx = 0
        self.clear_metric_num()
        self.clear_metric_frame()
        self._world.soft_destroy()

    def eva_check_goal(self):
        # get ego current state
        player = self._world.player
        t = player.get_transform()
        v = player.get_velocity()
        c = player.get_control()

        # find the closest goal
        closest_goal = [0.0, 0.0]
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = sys.float_info.max  # Todo: yaw
        for parking_goal in self._world.all_parking_goals:
            if t.distance(parking_goal) < self._distance_diff_to_goal:
                self._distance_diff_to_goal = t.distance(parking_goal)
                self._x_diff_to_goal = abs(t.x - parking_goal.x)
                self._y_diff_to_goal = abs(t.y - parking_goal.y)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y

        # check stop
        is_stop = (c.throttle == 0.0) and (v < 1e-3) and c.reverse
        if not is_stop:
            self._num_frames_in_goal = 0
            self._num_frames_nearby_goal = 0
            self._num_frames_nearby_no_goal = 0
            return

        # check success parking
        if self.check_success_slot(closest_goal, t):
            self.start_next_parking()
            return

        # check fail parking
        if self.check_fail_slot(closest_goal, t):
            self.start_next_parking()
            return

    def check_success_slot(self, closest_goal, ego_transform):
        x_in_slot = (abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff)
        y_in_slot = (abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff)
        r_in_slot = (self._orientation_diff_to_goal <= self._goal_reach_orientation_diff)

        if x_in_slot and y_in_slot and r_in_slot:
            self._num_frames_in_goal += 1

        if self._num_frames_in_goal >= self._num_frames_in_goal_needed:
            if (self._parking_goal.x == closest_goal[0]) and (self._parking_goal.y == closest_goal[1]):
                self._target_success_nums += 1
            else:
                self._no_target_success_nums += 1

    def check_fail_slot(self, closest_goal, ego_transform):
        x_in_rough_slot = \
            (self._goal_reach_x_diff < abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff * 2)
        y_in_rough_slot = \
            (self._goal_reach_y_diff < abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff * 2)
        r_in_rough_slot = \
            (self._goal_reach_orientation_diff < self._orientation_diff_to_goal <= self._goal_reach_orientation_diff * 2)

        dist_in_rough_slot = x_in_rough_slot and y_in_rough_slot
        if dist_in_rough_slot or r_in_rough_slot:
            if (self._parking_goal.x == closest_goal[0]) and (self._parking_goal.y == closest_goal[1]):
                self._num_frames_nearby_goal += 1
            else:
                self._num_frames_nearby_no_goal += 1

        if self._num_frames_nearby_goal >= self._num_frames_nearby_goal_needed:
            self._target_fail_nums += 1
            return

        if self._num_frames_nearby_no_goal >= self._num_frames_nearby_no_goal_needed:
            self._no_target_fail_nums += 1
            return

    def save_csv(self):
        pass

    def save_epoch_metric(self):
        pass

    def save_average_metric(self):
        pass

    def is_out_of_bound(self, ego_loc):
        x_out_bound = ((ego_loc.x < parking_position.town04_scope['x_min']) or
                       (ego_loc.x > parking_position.town04_scope['x_max']))
        y_out_bound = ((ego_loc.y < parking_position.town04_scope['y_min']) or
                       (ego_loc.y > parking_position.town04_scope['y_max']))
        return x_out_bound or y_out_bound
