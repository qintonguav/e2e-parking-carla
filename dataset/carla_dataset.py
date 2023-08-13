import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import random
from PIL import Image


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
    target_array = np.array([x, y, z], 1.0, dtype=float)
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
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
    return cropped_image


def tokenize(throttle, brake, steer, reverse, token_num=200):
    """
    Tokenize control signal
    :param throttle: [0,1]
    :param brake: [0,1]
    :param steer: [-1,1]
    :param reverse: {0,1}
    :param token_num: size of token
    :return: tokenized control range [0, token_num-4]
    """

    valid_token = token_num - 4
    half_token = valid_token / 2

    if brake != 0.0:
        throttle_brake_token = int(half_token * (-brake + 1))
    else:
        throttle_brake_token = int(half_token * (throttle + 1))
    steer_token = int((steer + 1) * half_token)
    reverse_token = int(reverse * valid_token)
    return [throttle_brake_token, steer_token, reverse_token]


def detokenize(token_list, token_num=200):
    """
    Detokenize control signals
    :param token_list: [throttle_brake, steer, reverse]
    :param token_num: size of token number
    :return: control signal values
    """

    valid_token = token_num - 4
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

    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return torch.from_numpy(in_meters).unsqueeze(0)


class CarlaDataset(torch.utils.data.Dataset):


class ProcessSemantic:


class ProcessImage:
