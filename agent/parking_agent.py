import sys
import carla
import math
import pathlib
import yaml
import torch
import numpy as np
import logging
import time
import pygame
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic
from dataset.carla_dataset import detokenize
from carla_data_generator.network_evaluator import NetworkEvaluator
from model.parking_model import ParkingModel


def show_control_info(window, control, steering_wheel_image, width, height, font):
    histogram_width = 15

    t_x = width - 30
    t_y = height - 50

    b_x = t_x - 30
    b_y = t_y

    s_x = t_x - 80
    s_y = t_y - 40

    r_x = t_x - 140
    r_y = t_y

    # throttle max = 0.5 in data gen
    throttle_height = (control['throttle'] * 200) * 0.8
    throttle_rect = pygame.Rect(t_x, t_y - throttle_height, histogram_width, throttle_height)
    pygame.draw.rect(window, (0, 255, 0), throttle_rect)

    brake_height = (control['brake'] * 100) * 0.8
    brake_rect = pygame.Rect(b_x, b_y - brake_height, histogram_width, brake_height)
    pygame.draw.rect(window, (255, 0, 0), brake_rect)

    steer = -control['steer'] * 90
    rotated_steering_wheel = pygame.transform.rotate(steering_wheel_image, steer)
    rotated_rect = rotated_steering_wheel.get_rect(center=(s_x, s_y))
    window.blit(rotated_steering_wheel, rotated_rect)

    reverse = bool(control['reverse'])
    rect = pygame.Rect((r_x, r_y - 10), (10, 10))
    pygame.draw.rect(window, (0, 0, 0), rect, 0 if reverse else 1)

    # show text
    throttle_text = font.render("T", True, (0, 255, 0))
    brake_text = font.render("B", True, (255, 0, 0))
    steer_text = font.render("S", True, (0, 0, 0))
    reverse_text = font.render("R", True, (0, 0, 0))

    window.blit(throttle_text, (t_x + 2, t_y + 10))
    window.blit(brake_text, (b_x + 2, b_y + 10))
    window.blit(steer_text, (s_x - 4, s_y + 50))
    window.blit(reverse_text, (r_x, r_y + 10))


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def grid_show(to_shows, cols):
    it = iter(to_shows)
    fig, axs = plt.subplots(1, cols, figsize=(cols * 2, cols))
    for j in range(cols):
        try:
            image, title = next(it)
        except StopIteration:
            image = np.zeros_like(to_shows[0][0])
            title = 'pad'
        axs[j].imshow(image)
        axs[j].set_title(title)
        axs[j].set_yticks([])
        axs[j].set_xticks([])
    plt.show()


def visualize_heads(att_map):
    to_shows = []
    att_map = att_map.squeeze()
    cols = att_map.shape[0] + 1
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def get_atten_avg_map(att_map, grid_index, image, grid_size=16):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    att_map = att_map.squeeze()
    average_att_map = att_map.mean(axis=0)
    atten_avg = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    atten_avg = Image.fromarray(atten_avg.numpy()).resize(image.size)
    return grid_image, atten_avg


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=16, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        mask = att_map[i][grid_index].reshape(grid_size[0], grid_size[1])
        mask = Image.fromarray(mask.numpy()).resize(image.size)
        to_shows.append((mask, f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    average_mask = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    average_mask = Image.fromarray(average_mask.numpy()).resize(image.size)
    to_shows.append((average_mask, 'Head Average'))

    plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
    rows = 1
    cols = 7

    it = iter(to_shows)
    for j in range(cols):
        try:
            mask, title = next(it)
        except StopIteration:
            mask = np.zeros_like(to_shows[0][0])
            title = 'pad'
        ax_attem = plt.subplot(rows, cols, j + 1)
        ax_attem.axis('off')
        ax_attem.set_title(title, fontsize=10)
        ax_attem.imshow(grid_image)
        ax_attem.imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')

    plt.pause(0.1)
    plt.clf()


class ParkingAgent:
    def __init__(self, network_evaluator: NetworkEvaluator, args):

        self.show_eva_imgs = args.show_eva_imgs

        self.atten_avg = None
        self.grid_image = None

        self.rgb_rear = None
        self.rgb_right = None
        self.rgb_left = None
        self.rgb_front = None
        self.seg_bev = None
        self.target_bev = None

        self.pre_target_point = None

        self.model = None
        self.device = None

        self.cfg = Configuration()
        self.load_cfg(args)

        self.log_path = pathlib.Path(self.cfg.log_dir)
        if not self.log_path.exists():
            self.log_path.mkdir()

        self.BOS_token = self.cfg.token_nums - 3

        self.hist_frame_nums = self.cfg.hist_frame_nums

        self.net_eva = network_evaluator
        self.world = network_evaluator.world
        self.player = network_evaluator.world.player

        self.is_init = False
        self.intrinsic_crop = None
        self.extrinsic = None
        self.image_process = None
        self.semantic_process = ProcessSemantic(self.cfg)

        self.process_frequency = 3  # process sensor data for every 3 steps 0.1s
        self.step = -1

        self.prev_xy_thea = None

        self.trans_control = carla.VehicleControl()
        self.gru_control = carla.VehicleControl()

        self.save_output = SaveOutput()
        self.hook_handle = None
        self.load_model(args.model_path)

        self.stop_count = 0
        self.boost = False
        self.boot_step = 0

        self.init_agent()

        plt.ion()

    def load_cfg(self, args):

        with open(args.model_config_path, 'r') as config_file:
            try:
                cfg_yaml = (yaml.safe_load(config_file))
            except yaml.YAMLError:
                logging.exception('Invalid YAML Config file {}', args.config)
        self.cfg = get_cfg(cfg_yaml)

    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModel(self.cfg)
        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        state_dict = OrderedDict([(k.replace('model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        patch_attention(self.model.feature_fusion.encoder.layers[-1].self_attn)
        self.hook_handle = self.model.feature_fusion.encoder.layers[-1].self_attn.register_forward_hook(
            self.save_output)

        logging.info('Load E2EParkingModel from %s', parking_pth_path)

    def save_seg_img(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]
        # image_file = pathlib.Path(self.cfg.log_dir) / ('%04d.png' % self.step)
        # Image.fromarray(np.uint8(pred_seg_img), mode='L').save(image_file)
        self.seg_bev = pred_seg_img

    def save_target_bev_img(self, target_bev):
        target_bev = target_bev[0]
        target_bev = target_bev.detach().cpu().numpy()
        target_bev[target_bev == 1] = 255
        target_bev_img = target_bev[0, :, :][::-1]
        self.target_bev = target_bev_img

    def save_prev_target(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]

        h, w = pred_seg_img.shape
        target_slot_x = []
        target_slot_y = []
        for row_idx in range(h):
            for col_idx in range(w):
                if pred_seg_img[row_idx, col_idx] == 255:
                    target_slot_x.append(row_idx)
                    target_slot_y.append(col_idx)

        # target point in bev
        if (len(target_slot_x) > 0) and (len(target_slot_y) > 0):
            new_target_x = int(np.average(target_slot_x))
            new_target_y = int(np.average(target_slot_y))
            self.pre_target_point = self.get_target_point_ego_coord(pred_seg_img, [new_target_x, new_target_y])

    def get_target_point_ego_coord(self, pred_seg_img, target_point_pixel_idx):
        bev_shape = pred_seg_img.shape[0]
        x = -(target_point_pixel_idx[0] - bev_shape / 2)
        y = target_point_pixel_idx[1] - bev_shape / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2]]
        return target_point_ego_coord

    def init_agent(self):
        w = self.world.cam_config['width']
        h = self.world.cam_config['height']

        self.intrinsic_crop = update_intrinsics(
            torch.from_numpy(self.world.intrinsic).float(),
            (h - self.cfg.image_crop) / 2,
            (w - self.cfg.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic_crop = self.intrinsic_crop.unsqueeze(0).expand(4, 3, 3)

        veh2cam_dict = self.world.veh2cam_dict
        front_to_ego = torch.from_numpy(veh2cam_dict['rgb_front']).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(veh2cam_dict['rgb_left']).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(veh2cam_dict['rgb_right']).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(veh2cam_dict['rgb_rear']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

        self.image_process = ProcessImage(veh2cam_dict, self.world.intrinsic,
                                          self.world.cam_config, self.cfg.image_crop)

        self.step = -1
        self.pre_target_point = None

    def save_atten_avg_map(self, data):
        atten = self.save_output.outputs[0].detach().cpu()
        # visualize_heads(atten)

        bev = data['segmentation']
        bev = bev.convert("RGB")
        # visualize_grid_to_grid(atten, 136, bev)
        grid_image, atten_avg = get_atten_avg_map(atten, 136, bev)
        grid_image = np.asarray(grid_image)[::-1, ...]
        atten_avg = np.asarray(atten_avg)[::-1, ...]
        return grid_image, atten_avg

    def tick(self):
        if self.net_eva.agent_need_init:
            self.init_agent()
            self.net_eva.agent_need_init = False

        self.step += 1

        # stop 1s for new eva
        if self.step < 30:
            self.player.apply_control(carla.VehicleControl())
            self.player.set_transform(self.net_eva.ego_transform)
            return

        if self.step % self.process_frequency == 0:
            data_frame = self.world.sensor_data_frame

            if not data_frame:
                return

            vehicle_transform = data_frame['veh_transfrom']
            imu_data = data_frame['imu']

            data = self.get_model_data(data_frame)

            self.model.eval()
            with torch.no_grad():
                start_time = time.time()

                pred_controls, pred_segmentation, _, target_bev = self.model.predict(data)

                end_time = time.time()
                self.net_eva.inference_time.append(end_time - start_time)

                self.save_prev_target(pred_segmentation)
                control_signal = detokenize(pred_controls[0].tolist()[1:], self.cfg.token_nums)

                self.trans_control.throttle = control_signal[0]
                self.trans_control.brake = control_signal[1]
                self.trans_control.steer = control_signal[2]
                self.trans_control.reverse = control_signal[3]

                self.speed_limit(data_frame)

                if self.show_eva_imgs:
                    self.grid_image, self.atten_avg = self.save_atten_avg_map(data)
                    self.save_seg_img(pred_segmentation)
                    self.save_target_bev_img(target_bev)
                    self.display_imgs()

                self.save_output.clear()

            self.prev_xy_thea = [vehicle_transform.location.x,
                                 vehicle_transform.location.y,
                                 imu_data.compass if np.isnan(imu_data.compass) else 0]

        self.player.apply_control(self.trans_control)

    def speed_limit(self, data_frame):
        # if vehicle stops at initialization, give throttle until Gear turns to 1
        if data_frame['veh_control'].gear == 0:
            self.trans_control.throttle = 0.5

        speed = (3.6 * math.sqrt(
            data_frame['veh_velocity'].x ** 2 + data_frame['veh_velocity'].y ** 2 + data_frame['veh_velocity'].z ** 2))

        # limit the vehicle speed within 15km/h when reverse is False
        if not self.trans_control.reverse and speed >= 12:
            self.trans_control.throttle = 0.0

        # limit the vehicle speed within 8km/h when reverse is True
        if self.trans_control.reverse and speed >= 10:
            self.trans_control.throttle = 0.0

        # if brake and throttle both not on, and speed < 2 for more than 2 seconds, give it a small throttle for 1
        # second
        if self.trans_control.throttle < 1e-5 and self.trans_control.brake < 1e-5 and speed < 2.0:
            self.stop_count += 1
        else:
            self.stop_count = 0.0

        if self.stop_count > 10:  # 1s
            self.boost = True

        if self.boost:
            self.trans_control.throttle = 0.3
            self.boot_step += 1

        if self.boot_step > 10 or self.trans_control.brake > 1e-5:  # 1s
            self.boot_step = 0
            self.boost = False

    def get_model_data(self, data_frame):

        vehicle_transform = data_frame['veh_transfrom']
        imu_data = data_frame['imu']
        vehicle_velocity = data_frame['veh_velocity']

        data = {}

        target_point = convert_slot_coord(vehicle_transform, self.net_eva.eva_parking_goal)

        front_final, self.rgb_front = self.image_process(target_point, cam_id='rgb_front',
                                                         image=data_frame['rgb_front'])
        left_final, self.rgb_left = self.image_process(target_point, cam_id='rgb_left', image=data_frame['rgb_left'])
        right_final, self.rgb_right = self.image_process(target_point, cam_id='rgb_right',
                                                         image=data_frame['rgb_right'])
        rear_final, self.rgb_rear = self.image_process(target_point, cam_id='rgb_rear', image=data_frame['rgb_rear'])

        images = [front_final, left_final, right_final, rear_final]
        images = torch.cat(images, dim=0)
        data['image'] = images.unsqueeze(0)

        data['extrinsics'] = self.extrinsic.unsqueeze(0)
        data['intrinsics'] = self.intrinsic_crop.unsqueeze(0)

        velocity = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2))
        data['ego_motion'] = torch.tensor([velocity, imu_data.accelerometer.x, imu_data.accelerometer.y],
                                          dtype=torch.float).unsqueeze(0).unsqueeze(0)

        if self.pre_target_point is not None:
            target_point = [self.pre_target_point[0], self.pre_target_point[1], target_point[2]]
        data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)

        data['gt_control'] = torch.tensor([self.BOS_token], dtype=torch.int64).unsqueeze(0)

        if self.show_eva_imgs:
            img = encode_npy_to_pil(np.asarray(data_frame['topdown'].squeeze().cpu()))
            img = np.moveaxis(img, 0, 2)
            img = Image.fromarray(img)
            seg_gt = self.semantic_process(path=None, image=img, scale=0.5, crop=200, target_slot=target_point)
            seg_gt[seg_gt == 1] = 128
            seg_gt[seg_gt == 2] = 255
            data['segmentation'] = Image.fromarray(seg_gt)

        return data

    def draw_waypoints(self, waypoints):
        ego_t = self.world.player.get_transform()
        ego_loc = carla.Location(x=ego_t.location.x, y=ego_t.location.y, z=0.20)
        self.world.world.debug.draw_string(ego_loc, 'O', draw_shadow=True, color=carla.Color(255, 0, 0))

        wp_list = waypoints[0].tolist()
        for wp in wp_list:
            logging.info('wp: dx: %4f; dy: %4f;', wp[0], wp[1])
            loc = carla.Location(x=ego_t.location.x + wp[0], y=ego_t.location.y + wp[1], z=0.20)
            self.world.world.debug.draw_string(loc, 'O', draw_shadow=True, color=carla.Color(0, 255, 0))

    def draw_control_info(self, trans_control):
        ego_t = carla.Transform(carla.Location(x=302.0, y=-248.239487, z=0.32682),
                                carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))

        throttle_loc = carla.Location(x=ego_t.location.x - 1.5, y=ego_t.location.y + 6.5, z=0.20)
        throttle = int(trans_control.throttle * 1000) / 1000
        self.world.world.debug.draw_string(throttle_loc, ' throttle: ' + str(throttle),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        brake_loc = carla.Location(x=ego_t.location.x - 2.5, y=ego_t.location.y + 6.5, z=0.20)
        brake = int(trans_control.brake * 1000) / 1000
        self.world.world.debug.draw_string(brake_loc, '    brake: ' + str(brake),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        steer_loc = carla.Location(x=ego_t.location.x - 3.5, y=ego_t.location.y + 6.5, z=0.20)
        steer = int(trans_control.steer * 1000) / 1000
        self.world.world.debug.draw_string(steer_loc, '     steer: ' + str(steer),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        reverse_loc = carla.Location(x=ego_t.location.x - 4.5, y=ego_t.location.y + 6.5, z=0.20)
        self.world.world.debug.draw_string(reverse_loc, 'reverse: ' + str(trans_control.reverse),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

    def get_gru_control(self, control_output):
        control = control_output[0][0].tolist()
        if control[0] > 0:
            throttle = control[0]
            brake = 0.0
        else:
            throttle = 0.0
            brake = -control[0]
        steer = control[1]
        reverse = (True if control[2] > 0 else False)
        return [throttle, brake, steer, reverse]

    def carla_to_nparray(self, carla_img):
        image_array = np.reshape(np.copy(carla_img.raw_data), (carla_img.height, carla_img.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]
        return image_array

    def display_imgs(self):
        plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
        rows = 2
        cols = 4
        ax_ctl = plt.subplot(rows, cols, 4)
        ax_ctl.axis('off')
        t_x = 0.2
        t_y = 0.8
        throttle_show = int(self.trans_control.throttle * 1000) / 1000
        steer_show = int(self.trans_control.steer * 1000) / 1000
        brake_show = int(self.trans_control.brake * 1000) / 1000
        reverse_show = self.trans_control.reverse
        ax_ctl.text(t_x, t_y, 'Throttle: ' + str(throttle_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.2, '    Steer: ' + str(steer_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.4, '   Brake: ' + str(brake_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.6, 'Reverse: ' + str(reverse_show), fontsize=10, color='red')

        ax_front = plt.subplot(rows, cols, 1)
        ax_front.axis('off')
        ax_front.set_title('front', fontsize=10)
        ax_front.imshow(self.rgb_front)

        ax_rear = plt.subplot(rows, cols, 2)
        ax_rear.axis('off')
        ax_rear.set_title('rear', fontsize=10)
        ax_rear.imshow(self.rgb_rear)

        ax_atten = plt.subplot(rows, cols, 7)
        ax_atten.axis('off')
        ax_atten.set_title('atten(output)', fontsize=10)
        ax_atten.imshow(self.grid_image)
        ax_atten.imshow(self.atten_avg / np.max(self.atten_avg), alpha=0.6, cmap='rainbow')

        ax_left = plt.subplot(rows, cols, 5)
        ax_left.axis('off')
        ax_left.set_title('left', fontsize=10)
        ax_left.imshow(self.rgb_left)

        ax_right = plt.subplot(rows, cols, 6)
        ax_right.axis('off')
        ax_right.set_title('right', fontsize=10)
        ax_right.imshow(self.rgb_right)

        ax_bev = plt.subplot(rows, cols, 3)
        ax_bev.axis('off')
        ax_bev.set_title('target_bev(input)', fontsize=10)
        ax_bev.imshow(self.target_bev)

        ax_bev = plt.subplot(rows, cols, 8)
        ax_bev.axis('off')
        ax_bev.set_title('seg_bev(output)', fontsize=10)
        ax_bev.imshow(self.seg_bev)

        plt.pause(0.1)
        plt.clf()

    def get_eva_control(self):
        control = {
            'throttle': self.trans_control.throttle,
            'steer': self.trans_control.steer,
            'brake': self.trans_control.brake,
            'reverse': self.trans_control.reverse,
        }
        return control
