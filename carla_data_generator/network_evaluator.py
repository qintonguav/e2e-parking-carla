import math
import random
import logging
import sys
import pathlib

import carla
import numpy as np
from datetime import datetime

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
        result_dir = '_'.join(map(lambda x: '%02d' % x,
                                  (now.year, now.month, now.day, now.hour, now.minute, now.second)))
        self._eva_result_path = pathlib.Path(args.eva_result_path) / result_dir
        self._eva_result_path.mkdir(parents=True, exist_ok=False)

        self._render_bev = args.show_eva_imgs

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
        self._frames_per_second = 30

        # metric frames
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_nearby_no_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0
        self._num_frames_in_goal_needed = 2 * self._frames_per_second         # 2s
        self._num_frames_nearby_goal_needed = 2 * self._frames_per_second     # 2s
        self._num_frames_nearby_no_goal_needed = 2 * self._frames_per_second  # 2s
        self._num_frames_outside_needed = 20 * self._frames_per_second        # 20s
        self._num_frames_total_needed = 30 * self._frames_per_second          # 30s

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

        self._epoch_metric_info = {}
        self._epoch_avg_metric_info = {}

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
        self.start_eva_epoch(self._eva_epoch_idx)

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

    def start_eva_epoch(self, eva_epoch_idx):
        if eva_epoch_idx >= self._eva_parking_nums:
            self.save_epoch_avg_metric_csv()
            exit(0)

        self.soft_destroy()

        self.clear_metric_rate()

        self._seed = self._init_seed

        self._parking_goal_index = 16
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y,
                                                        self._eva_parking_nums,
                                                        self._eva_parking_idx)
        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform(self._eva_parking_nums,
                                                                                  self._eva_parking_idx)
        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]
        self._world.player.set_transform(self._ego_transform)
        self._world.restart(self._seed, self._parking_goal_index, self._ego_transform)

        self._eva_task_idx = 0
        self._epoch_metric_info = {}

    def start_next_parking(self):
        self._start_time = datetime.now()

        self._agent_need_init = True

        self._eva_parking_idx += 1
        if self.is_complete_slot(self._eva_parking_idx):
            self.save_slot_metric()
            self.start_next_slot()
            return

        self.clear_metric_frame()

        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform(self._eva_parking_nums,
                                                                                  self._eva_parking_idx)
        self._world.player.set_transform(self._ego_transform)
        self._world.player.apply_control(carla.VehicleControl())

    def is_complete_slot(self, eva_parking_idx):
        if eva_parking_idx >= self._eva_parking_nums:
            # log
            return True
        return False

    def start_next_slot(self):
        self._eva_task_idx += 1
        if self.is_complete_epoch(self._eva_task_idx):
            # log
            self.save_epoch_metric_csv()
            self._eva_epoch_idx += 1
            self.start_eva_epoch(self._eva_epoch_idx)
            return

        if self._parking_goal_index < 48:
            self._parking_goal_index += 2
        else:
            self._parking_goal_index = 16

        self.soft_destroy()

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform = self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y,
                                                                              self._eva_parking_nums,
                                                                              self._eva_parking_idx)
        self._world.player.set_transform(self._ego_transform)

        self._seed += 1
        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]
        self._world.restart(self._seed, self._parking_goal_index, self._ego_transform)

    def is_complete_epoch(self, eva_task_idx):
        if eva_task_idx >= self._eva_task_nums:
            # log
            return True
        return False

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

    def destroy(self):
        self._world.destroy()

    def eva_check_goal(self):
        # get ego current state
        player = self._world.player
        t = player.get_transform()
        v = player.get_velocity()
        c = player.get_control()
        speed = (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # find the closest goal
        closest_goal = [0.0, 0.0]
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = min(t.yaw, abs(180 - t.yaw))
        for parking_goal in self._world.all_parking_goals:
            if t.distance(parking_goal) < self._distance_diff_to_goal:
                self._distance_diff_to_goal = t.distance(parking_goal)
                self._x_diff_to_goal = abs(t.x - parking_goal.x)
                self._y_diff_to_goal = abs(t.y - parking_goal.y)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y

        # check stop
        is_stop = (c.throttle == 0.0) and (speed < 1e-3) and c.reverse
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
                self._position_error.append(self._distance_diff_to_goal)
                self._orientation_error.append(self._orientation_diff_to_goal)
                self._parking_time.append((datetime.now() - self._start_time) / self._frames_per_second)
            else:
                self._no_target_success_nums += 1
            return True

        return False

    def check_fail_slot(self, closest_goal, ego_transform):
        x_in_rough_slot = \
            (self._goal_reach_x_diff < abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff * 2)
        y_in_rough_slot = \
            (self._goal_reach_y_diff < abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff * 2)
        r_in_rough_slot = \
            (self._goal_reach_orientation_diff <
             self._orientation_diff_to_goal <=
             self._goal_reach_orientation_diff * 2)

        dist_in_rough_slot = x_in_rough_slot and y_in_rough_slot
        if dist_in_rough_slot or r_in_rough_slot:
            if (self._parking_goal.x == closest_goal[0]) and (self._parking_goal.y == closest_goal[1]):
                self._num_frames_nearby_goal += 1
            else:
                self._num_frames_nearby_no_goal += 1

        if self._num_frames_nearby_goal >= self._num_frames_nearby_goal_needed:
            self._target_fail_nums += 1
            return True

        if self._num_frames_nearby_no_goal >= self._num_frames_nearby_no_goal_needed:
            self._no_target_fail_nums += 1
            return True

        return False

    def save_slot_metric(self):
        TSR = (self._target_success_nums / self._eva_parking_nums) * 100.0
        TFR = (self._target_fail_nums / self._eva_parking_nums) * 100.0
        NTSR = (self._no_target_success_nums / self._eva_parking_nums) * 100.0
        NTFR = (self._no_target_fail_nums / self._eva_parking_nums) * 100.0
        CR = (self._collision_nums / self._eva_parking_nums) * 100.0
        OR = (self._outbound_nums / self._eva_parking_nums) * 100.0
        TR = (self._timeout_nums / self._eva_parking_nums) * 100.0
        APE = np.mean(self._position_error)
        AOE = np.mean(self._orientation_error)
        AIT = np.mean(self._inference_time)
        APT = np.mean(self._parking_time)

        slot_id = parking_position.task_idx[self._eva_task_idx]
        self._epoch_metric_info[slot_id] = {
            'target_success_rate': TSR,
            'target_fail_rate': TFR,
            'no_target_success_rate': NTSR,
            'no_target_fail_rate': NTFR,
            'collision_rate': CR,
            'outbound_rate': OR,
            'timeout_rate': TR,
            'average_position_error': APE,
            'average_orientation_error': AOE,
            'average_inference_time': AIT,
            'average_parking_time': APT,
        }

        self._target_success_rate.append(TSR)
        self._target_fail_rate.append(TFR)
        self._no_target_success_rate.append(NTSR)
        self._no_target_fail_rate.append(NTFR)
        self._collision_rate.append(CR)
        self._outbound_rate.append(OR)
        self._timeout_rate.append(TR)
        self._average_position_error.append(APE)
        self._average_orientation_error.append(AOE)
        self._average_inference_time.append(AIT)
        self._average_parking_time.append(APT)

    def save_epoch_metric_csv(self):
        self._epoch_metric_info['Avg'] = {
            'target_success_rate': np.mean(self._target_success_rate),
            'target_fail_rate': np.mean(self._target_fail_rate),
            'no_target_success_rate': np.mean(self._no_target_success_rate),
            'no_target_fail_rate': np.mean(self._no_target_fail_rate),
            'collision_rate': np.mean(self._collision_rate),
            'outbound_rate': np.mean(self._outbound_rate),
            'timeout_rate': np.mean(self._timeout_rate),
            'average_position_error': np.mean(self._average_position_error),
            'average_orientation_error': np.mean(self._average_orientation_error),
            'average_inference_time': np.mean(self._average_inference_time),
            'average_parking_time': np.mean(self._average_parking_time),
        }

    def save_csv(self):
        pass

    def save_epoch_avg_metric_csv(self):
        pass

    def is_out_of_bound(self, ego_loc):
        x_out_bound = ((ego_loc.x < parking_position.town04_bound['x_min']) or
                       (ego_loc.x > parking_position.town04_bound['x_max']))
        y_out_bound = ((ego_loc.y < parking_position.town04_bound['y_min']) or
                       (ego_loc.y > parking_position.town04_bound['y_max']))
        return x_out_bound or y_out_bound

    def world_tick(self):
        self._world.world_tick()

    def render(self, display):
        self._world.render(display)

    @property
    def world(self):
        return self._world

    @property
    def agent_need_init(self):
        return self._agent_need_init

    @agent_need_init.setter
    def agent_need_init(self, need_init):
        self._agent_need_init = need_init

    @property
    def inference_time(self):
        return self._inference_time

    @property
    def eva_parking_goal(self):
        return self._eva_parking_goal
