import json
import os
import carla
import torch.utils.data
import numpy as np
import torchvision.transforms

from PIL import Image
from loguru import logger


def convert_slot_coord(ego_trans, target_point):
    """
    Convert target parking slot from world frame into self_veh frame
    :param ego_trans: veh2world transform
    :param target_point: target parking slot in world frame [x, y, yaw]
    :return: target parking slot in veh frame [x, y, yaw]
    """

    target_point_self_veh = convert_veh_coord(target_point[0], target_point[1], 1.0, ego_trans)

    yaw_diff = target_point[2] - ego_trans.rotation.yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    target_point = [target_point_self_veh[0], target_point_self_veh[1], yaw_diff]

    return target_point


def convert_veh_coord(x, y, z, ego_trans):
    """
    Convert coordinate (x,y,z) in world frame into self-veh frame
    :param x:
    :param y:
    :param z:
    :param ego_trans: veh2world transform
    :return: coordinate in self-veh frame
    """

    world2veh = np.array(ego_trans.get_inverse_matrix())
    target_array = np.array([x, y, z, 1.0], dtype=float)
    target_point_self_veh = world2veh @ target_array
    return target_point_self_veh


def scale_and_crop_image(image, scale=1.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array
    :param image: original image
    :param scale: scale factor
    :param crop: crop size
    :return: cropped image
    """

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), resample=Image.NEAREST)
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop].copy()
    return cropped_image


def tokenize(throttle, brake, steer, reverse, token_nums=200):
    """
    Tokenize control signal
    :param throttle: [0,1]
    :param brake: [0,1]
    :param steer: [-1,1]
    :param reverse: {0,1}
    :param token_nums: size of token
    :return: tokenized control range [0, token_nums-4]
    """

    valid_token = token_nums - 4
    half_token = valid_token / 2

    if brake != 0.0:
        throttle_brake_token = int(half_token * (-brake + 1))
    else:
        throttle_brake_token = int(half_token * (throttle + 1))
    steer_token = int((steer + 1) * half_token)
    reverse_token = int(reverse * valid_token)
    return [throttle_brake_token, steer_token, reverse_token]


def detokenize(token_list, token_nums=200):
    """
    Detokenize control signals
    :param token_list: [throttle_brake, steer, reverse]
    :param token_nums: size of token number
    :return: control signal values
    """

    valid_token = token_nums - 4
    half_token = float(valid_token / 2)

    if token_list[0] > half_token:
        throttle = token_list[0] / half_token - 1
        brake = 0.0
    else:
        throttle = 0.0
        brake = -(token_list[0] / half_token - 1)

    steer = (token_list[1] / half_token) - 1
    reverse = (True if token_list[2] > half_token else False)

    return [throttle, brake, steer, reverse]


def get_depth(depth_image_path, crop):
    """
    Convert carla RGB depth image into single channel depth in meters
    :param depth_image_path: carla depth image in RGB format
    :param crop: crop size
    :return: numpy array of depth image in meters
    """
    depth_image = Image.open(depth_image_path).convert('RGB')

    data = np.array(scale_and_crop_image(depth_image, scale=1.0, crop=crop))

    data = data.astype(np.float32)

    normalized = np.dot(data, [1.0, 256.0, 65536.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return torch.from_numpy(in_meters).unsqueeze(0)


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    update_intrinsic = intrinsics.clone()

    update_intrinsic[0, 0] *= scale_width
    update_intrinsic[0, 2] *= scale_width
    update_intrinsic[1, 1] *= scale_height
    update_intrinsic[1, 2] *= scale_height

    update_intrinsic[0, 2] -= left_crop
    update_intrinsic[1, 2] -= top_crop

    return update_intrinsic


def add_raw_control(data, throttle_brake, steer, reverse):
    if data['Brake'] != 0:
        throttle_brake.append(-data['Brake'])
    else:
        throttle_brake.append(data['Throttle'])
    steer.append(data['Steer'])
    reverse.append(int(data['Reverse']))


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, config):
        super(CarlaDataset, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums - 3
        self.EOS_token = self.BOS_token + 1
        self.PAD_token = self.EOS_token + 1

        self.root_dir = root_dir
        self.is_train = is_train

        # camera configs
        self.image_crop = self.cfg.image_crop
        self.intrinsic = None
        self.veh2cam_dict = {}
        self.extrinsic = None
        self.image_process = ProcessImage(self.image_crop)
        self.semantic_process = ProcessSemantic(self.cfg)

        self.init_camera_config()

        # data
        self.front = []
        self.left = []
        self.right = []
        self.rear = []

        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.rear_depth = []

        self.control = []

        self.velocity = []
        self.acc_x = []
        self.acc_y = []

        self.throttle_brake = []
        self.steer = []
        self.reverse = []

        self.target_point = []

        self.topdown = []

        self.get_data()

    def init_camera_config(self):
        cam_config = {'width': 400, 'height': 300, 'fov': 100}

        cam_specs = {
            'rgb_front': {
                'x': 1.5, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_left': {
                'x': 0.0, 'y': -0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': -90.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_right': {
                'x': 0.0, 'y': 0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': 90.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_rear': {
                'x': -2.2, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': -30.0, 'yaw': 180.0,
                'type': 'sensor.camera.rgb',
            },
        }

        # intrinsic
        w = cam_config['width']
        h = cam_config['height']
        fov = cam_config['fov']
        f = w / (2 * np.tan(fov * np.pi / 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic_original = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        self.intrinsic = update_intrinsics(
            torch.from_numpy(intrinsic_original).float(),
            (h - self.image_crop) / 2,
            (w - self.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic = self.intrinsic.unsqueeze(0).expand(4, 3, 3)

        # extrinsic
        cam2pixel = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        for cam_id, cam_spec in cam_specs.items():
            cam2veh = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                      carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                     roll=cam_spec['roll']))
            veh2cam = cam2pixel @ np.array(cam2veh.get_inverse_matrix())
            self.veh2cam_dict[cam_id] = veh2cam
        front_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_front']).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_left']).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_right']).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_rear']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

    def get_data(self):
        val_towns = self.cfg.validation_map
        train_towns = self.cfg.training_map
        train_data = os.path.join(self.root_dir, train_towns)
        val_data = os.path.join(self.root_dir, val_towns)

        town_dir = train_data if self.is_train == 1 else val_data

        # collect all parking data tasks
        root_dirs = os.listdir(town_dir)
        all_tasks = []
        for root_dir in root_dirs:
            root_path = os.path.join(town_dir, root_dir)
            for task_dir in os.listdir(root_path):
                task_path = os.path.join(root_path, task_dir)
                all_tasks.append(task_path)

        for task_path in all_tasks:
            total_frames = len(os.listdir(task_path + "/measurements/"))
            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                # collect data at current frame
                # image
                filename = f"{str(frame).zfill(4)}.png"
                self.front.append(task_path + "/rgb_front/" + filename)
                self.left.append(task_path + "/rgb_left/" + filename)
                self.right.append(task_path + "/rgb_right/" + filename)
                self.rear.append(task_path + "/rgb_rear/" + filename)

                # depth
                self.front_depth.append(task_path + "/depth_front/" + filename)
                self.left_depth.append(task_path + "/depth_left/" + filename)
                self.right_depth.append(task_path + "/depth_right/" + filename)
                self.rear_depth.append(task_path + "/depth_rear/" + filename)

                # BEV Semantic
                self.topdown.append(task_path + "/topdown/encoded_" + filename)

                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # ego position
                ego_trans = carla.Transform(carla.Location(x=data['x'], y=data['y'], z=data['z']),
                                            carla.Rotation(yaw=data['yaw'], pitch=data['pitch'], roll=data['roll']))

                # motion
                self.velocity.append(data['speed'])
                self.acc_x.append(data['acc_x'])
                self.acc_y.append(data['acc_y'])

                # control
                controls = []
                throttle_brakes = []
                steers = []
                reverse = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + 1 + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                    controls.append(
                        tokenize(data['Throttle'], data["Brake"], data["Steer"], data["Reverse"], self.cfg.token_nums))
                    add_raw_control(data, throttle_brakes, steers, reverse)

                controls = [item for sublist in controls for item in sublist]
                controls.insert(0, self.BOS_token)
                controls.append(self.EOS_token)
                controls.append(self.PAD_token)
                self.control.append(controls)

                self.throttle_brake.append(throttle_brakes)
                self.steer.append(steers)
                self.reverse.append(reverse)

                # target point
                with open(task_path + f"/parking_goal/0001.json", "r") as read_file:
                    data = json.load(read_file)
                parking_goal = [data['x'], data['y'], data['yaw']]
                parking_goal = convert_slot_coord(ego_trans, parking_goal)
                self.target_point.append(parking_goal)

        self.front = np.array(self.front).astype(np.string_)
        self.left = np.array(self.left).astype(np.string_)
        self.right = np.array(self.right).astype(np.string_)
        self.rear = np.array(self.rear).astype(np.string_)

        self.front_depth = np.array(self.front_depth).astype(np.string_)
        self.left_depth = np.array(self.left_depth).astype(np.string_)
        self.right_depth = np.array(self.right_depth).astype(np.string_)
        self.rear_depth = np.array(self.rear_depth).astype(np.string_)

        self.topdown = np.array(self.topdown).astype(np.string_)

        self.velocity = np.array(self.velocity).astype(np.float32)
        self.acc_x = np.array(self.acc_x).astype(np.float32)
        self.acc_y = np.array(self.acc_y).astype(np.float32)

        self.control = np.array(self.control).astype(np.int64)

        self.throttle_brake = np.array(self.throttle_brake).astype(np.float32)
        self.steer = np.array(self.steer).astype(np.float32)
        self.reverse = np.array(self.reverse).astype(np.int64)

        self.target_point = np.array(self.target_point).astype(np.float32)

        logger.info('Preloaded {} sequences', str(len(self.front)))

    def __len__(self):
        return len(self.front)

    def __getitem__(self, index):
        data = {}
        keys = ['image', 'depth', 'extrinsics', 'intrinsics', 'target_point', 'ego_motion', 'segmentation',
                'gt_control', 'gt_acc', 'gt_steer', 'gt_reverse']
        for key in keys:
            data[key] = []

        # image & extrinsics & intrinsics
        images = [self.image_process(self.front[index])[0], self.image_process(self.left[index])[0],
                  self.image_process(self.right[index])[0], self.image_process(self.rear[index])[0]]
        images = torch.cat(images, dim=0)
        data['image'] = images

        data['extrinsics'] = self.extrinsic
        data['intrinsics'] = self.intrinsic

        # depth
        depths = [get_depth(self.front_depth[index], self.image_crop),
                  get_depth(self.left_depth[index], self.image_crop),
                  get_depth(self.right_depth[index], self.image_crop),
                  get_depth(self.rear_depth[index], self.image_crop)]
        depths = torch.cat(depths, dim=0)
        data['depth'] = depths

        # segmentation
        segmentation = self.semantic_process(self.topdown[index], scale=0.5, crop=200,
                                             target_slot=self.target_point[index])
        data['segmentation'] = torch.from_numpy(segmentation).long().unsqueeze(0)

        # target_point
        data['target_point'] = torch.from_numpy(self.target_point[index])

        # ego_motion
        ego_motion = np.column_stack((self.velocity[index], self.acc_x[index], self.acc_y[index]))
        data['ego_motion'] = torch.from_numpy(ego_motion)

        # gt control token
        data['gt_control'] = torch.from_numpy(self.control[index])

        # gt control raw
        data['gt_acc'] = torch.from_numpy(self.throttle_brake[index])
        data['gt_steer'] = torch.from_numpy(self.steer[index])
        data['gt_reverse'] = torch.from_numpy(self.reverse[index])

        return data


class ProcessSemantic:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image, scale, crop, target_slot):
        """
        Process original BEV ground truth image; return cropped image with target slot
        :param image: PIL Image or path to image
        :param scale: scale factor
        :param crop: image crop size
        :param target_slot: center location of the target parking slot in meters; vehicle frame
        :return: processed BEV semantic ground truth
        """

        # read image from disk
        if not isinstance(image, carla.Image):
            image = Image.open(image)
        image = image.convert('L')

        # crop image
        cropped_image = scale_and_crop_image(image, scale, crop)

        # draw target slot on BEV semantic
        cropped_image = self.draw_target_slot(cropped_image, target_slot)

        # create a new BEV semantic GT
        h, w = cropped_image.shape
        vehicle_index = cropped_image == 75
        target_index = cropped_image == 255
        semantics = np.zeros((h, w))
        semantics[vehicle_index] = 1
        semantics[target_index] = 2
        # LSS method vehicle toward positive x-axis on image
        semantics = semantics[::-1]

        return semantics.copy()

    def draw_target_slot(self, image, target_slot):

        size = image.shape[0]

        # convert target slot position into pixels
        x_pixel = target_slot[0] / self.cfg.bev_x_bound[2]
        y_pixel = target_slot[1] / self.cfg.bev_y_bound[2]
        target_point = np.array([size / 2 - x_pixel, size / 2 + y_pixel], dtype=int)

        # draw the whole parking slot
        slot_points = []
        for x in range(-27, 28):
            for y in range(-15, 16):
                slot_points.append(np.array([x, y, 1, 1], dtype=int))

        # rotate parking slots points

        slot_trans = np.array(
            carla.Transform(carla.Location(), carla.Rotation(yaw=float(-target_slot[2]))).get_matrix())
        slot_points = np.vstack(slot_points).T
        slot_points_ego = (slot_trans @ slot_points)[0:2].astype(int)

        # get parking slot points on pixel frame
        slot_points_ego[0] += target_point[0]
        slot_points_ego[1] += target_point[1]

        image[tuple(slot_points_ego)] = 255

        return image


class ProcessImage:
    def __init__(self, crop):
        self.crop = crop

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

    def __call__(self, image):
        if isinstance(image, carla.Image):
            image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            image = image[:, :, :3]
            image = image[:, :, ::-1]
            image = Image.fromarray(image)
        else:
            image = Image.open(image).convert('RGB')

        crop_image = scale_and_crop_image(image, scale=1.0, crop=self.crop)

        return self.normalise_image(np.array(crop_image)).unsqueeze(0), crop_image
