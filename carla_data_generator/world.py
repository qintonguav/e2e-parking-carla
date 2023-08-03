import sys
import carla
import re
import random
from hud import HUD
import math
from carla import ColorConverter as cc
import weakref
import collections
import pygame
import numpy as np
import parking_position
from datetime import datetime
import pathlib
import logging
from queue import Queue, Empty
from threading import Thread
import json
import cv2

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if sensor_queue.qsize() > 10:
        return
    sensor_queue.put((sensor_data, sensor_name))

class World(object):
    def __init__(self, carla_world, hud, args):

        self.world = carla_world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = float(1. / 30)
        settings.synchronous_model = True
        self.world.apply_settings(settings)

        self.parking_goal_index = 2
        self.parking_spawn_points = parking_position.parking_vehicle_locations_Town04.copy()
        self.target_parking_goal = self.parking_spawn_points[self.parking_goal_index]
        self.ego_transform = parking_position.EgoPostTown04(self.target_parking_goal)
        self.all_parking_goals = self.parking_spawn_points

        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        now = datetime.now()
        result_dir = '_'.join(map(lambda x : '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.save_path = pathlib.Path(args.save_path) / args.map/ result_dir
        self.save_path.mkdir(parents=True, exist_ok=False)

        # collection related settings
        self.save_frequency = 3
        self.step = -1
        self.num_tasks = args.task_num
        self.distance_diff_to_goal = 10000
        self.goal_reach_distance = 0.5
        self.rotation_diff_to_goal = 1000
        self.goal_reach_roation = 0.5
        self.num_frames_goal_need = 60
        self.num_frames_in_goal = 0
        # self.all_parking_goals = []
        self.task_index = 0
        self.sensor_list = []
        self.sensor_queue = Queue()
        self.veh2cam_dict = {}
        self.sensor_data_frame = {}
        self.batch_data_frames = []
        self.cam_config = {}
        self.cam_center = None
        self.cam_specs = None
        self.intrinsic = None
        self.cam2pixel = None

        # self.hud = HUD(args.width, args.height)
        self.hud = hud
        self.player = None
        self.spectator = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        # self.restart()
        self.world.on_tick(self.hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        self.init(args)

    def init(self, args):
        logging.info('***************init environment for task ****************', self.task_index)
        self.destroy()

        #self.init_static_npc()
        ego_transform = self.ego_transform.get_ego_transform()
        ego_vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        self.player = self.world.spawn_actor(ego_vehicle_bp, ego_transform)
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=283.85, y=-210.039, z=35),
                                                     carla.Rotation(pitch=-90)))
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = 0
        self.camera_manager.set_sensor(0, notify=False)

        self.setup_sensors()
        self.next_weather()

        logging.info('*************init enviroment for task %d done***********', self.task_index)

    def setup_sensors(self):
        # collision
        self.collision_sensor = CollisionSensor(self.player, self.hud)

        # gnss
        bp_gnss = self.world.get_blueprint_library().find('sensor.other.gnss')
        gnss = self.world.spawn_actor(bp_gnss, carla.Transform(), attach_to=self.player,
                                      attachment_type=carla.AttachmentType.Rigid)
        gnss.listen(lambda data: sensor_callback(data, self.sensor_queue, "gnss"))
        self.sensor_list.append(gnss)

        # imu
        bp_imu = self.world.get_blueprint_library().find('sensor.other.imu')
        imu = self.world.spawn_actor(bp_imu, carla.Transform(), attach_to=self.player,
                                      attachment_type=carla.AttachmentType.Rigid)
        imu.listen(lambda data: sensor_callback(data, self.sensor_queue, "imu"))
        self.sensor_list.append(imu)

        self.cam_config = {
            'width': 400,
            'height': 300,
            'fov': 100,
        }
        self.cam_center = np.array([self.cam_config['width'] / 2.0, self.cam_config['height'] / 2.0])
        self.cam_specs = {
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
            'depth_front': {
                'x': 1.5, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'type': 'sensor.camera.depth',
            },
            'depth_left': {
                'x': 0.0, 'y': -0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': -90.0,
                'type': 'sensor.camera.depth',
            },
            'depth_right': {
                'x': 0.0, 'y': 0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': 90.0,
                'type': 'sensor.camera.depth',
            },
            'depth_rear': {
                'x': -2.2, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': -30.0, 'yaw': 180.0,
                'type': 'sensor.camera.depth',
            },
        }

        for key, value in self.cam_specs.items():
            self.spawn_rgb_camera(key, value)

        w = self.cam_config['width']
        h = self.cam_config['height']
        fov = self.cam_config['fov']
        f = w / (2 * np.tan(fov * np.pi / 360.))
        cu = w / 2
        cv = h / 2
        self.intrinsic = np.array([
            [f, 0, cu],
            [0, f, cv],
            [0, 0, 1]
        ], dtype=np.float)

        self.cam2pixel = np.array([[0, 1, 0 , 0],
                                   [0, 0, -1, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 1]], dtype=float)

        for cam_id, cam_spec in self.cam_specs.items():
            if cam_id.startswith('rgb'):
                cam2veh = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                         carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                        roll=cam_spec['roll']))
                veh2cam = self.cam2pixel @ np.array(cam2veh.get_inverse_matrix())
                self.veh2cam_dict[cam_id] = veh2cam

    def spawn_rgb_camera(self, sensor_id, sensor_spec):
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.find(sensor_spec['type'])
        bp.set_attribute('image_size_x', str(self.cam_config['width']))
        bp.set_attribute('image_size_y', str(self.cam_config['height']))
        bp.set_attribute('fov', str(self.cam_config['fov']))
        sensor_location = carla.Location(x=sensor_spec['x'],
                                         y=sensor_spec['y'],
                                         z=sensor_spec['z'])
        sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                         roll=sensor_spec['roll'],
                                         yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        cam = self.world.spawn_actor(bp, sensor_transform, attach_to=self.player,
                                      attachment_type=carla.AttachmentType.Rigid)
        cam.listen(lambda data: sensor_callback(data, self.sensor_queue, sensor_id))
        self.sensor_list.append(cam)

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        # cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        # cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.camera_manager.clear_saved_images()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def tick(self, clock):
        try:
            t = self.player.get_transform()
            v = self.player.get_velocity()
            c = self.player.get_control()
            self.sensor_data_frame['veh_transform'] = t
            self.sensor_data_frame['veh_velocity'] = v
            self.sensor_data_frame['veh_control'] = c

            for i in range(0, len(self.sensor_list)):
                s_data = self.sensor_queue.get(block=True, timeout=1.0)
                self.sensor_data_frame[s_data[1]] = s_data[0]

        except Empty:
            logging.error("Some of the sensor information is missed")

        self.hud.tick(self, clock)

        self.world.debug.draw_string(self.target_parking_goal, 'T',
                                     draw_shadow=True, color=carla.Color(255, 0, 0))

        self.step += 1
        if self.step % self.save_frequency == 0:
            self.batch_data_frames.append(self.sensor_data_frame.copy())

        self.check_goal()

    def check_goal(self, save_sensors=True):
        t = self.player.get_transform().location
        r = self.player.get_transform().rotation

        self.distance_diff_to_goal = sys.float_info.max
        closest_goal = [0.0, 0.0, 0.0]
        for parking_goal in self.all_parking_goals:
            if t.distance(parking_goal) < self.distance_diff_to_goal:
                self.distance_diff_to_goal = t.distance(parking_goal)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y
                closest_goal[2] = r.yaw

        self.rotation_diff_to_goal = math.sqrt(min(abs(r.yaw), 90-abs(r.yaw)) ** 2 + r.roll **2 + r.pitch**2)

        if self.distance_diff_to_goal < self.goal_reach_distance and \
            self.rotation_diff_to_goal < self.goal_reach_roation:
            self.num_frames_in_goal += 1
        else:
            self.num_frames_in_goal = 0

        if self.num_frames_in_goal > self.num_frames_goal_need:
            logging.info('task %d goal reached; ready to save sensor data', self.task_index)
            self.save_sensor_data(closest_goal)
            logging.info('**************task %d done****************', self.task_index)
            self.task_index += 1

            if self.task_index >= self.num_tasks:
                logging.info('completed all tasks; Thank you!')
                exit(0)
            self.restart()


    def save_sensor_data(self, parking_goal):
        cur_save_path = pathlib.Path(self.save_path) / ('task' + str(self.task_index))
        cur_save_path.mkdir(parents=True, exist_ok=False)
        (cur_save_path / 'measurements').mkdir()
        (cur_save_path / 'parking_goal').mkdir()
        (cur_save_path / 'topdown').mkdir()

        for cam_id, cam_spec in self.cam_specs.items():
            if cam_id.startswith('rgb') or cam_id.startswith('depth'):
                (cur_save_path / cam_id).mkdir()

        total_frames = len(self.batch_data_frames)
        thread_num = 10
        frames_for_thread = total_frames // thread_num
        thread_list = []
        for t_idx in range(thread_num):
            start = t_idx * frames_for_thread
            if t_idx == thread_num - 1:
                end = total_frames
            else:
                end = (t_idx + 1) * frames_for_thread
            t = Thread(target=self.save_unit_data, args=(start, end, cur_save_path))
            t.start()
            thread_list.append(t)

        for thread in thread_list:
            thread.join()

        # save parking goal
        measurements_file = cur_save_path / 'parking_goal' / '0001.json'
        with open(measurements_file, 'w') as f:
            data = {'x': parking_goal[0],
                    'y': parking_goal[1],
                    'yaw': parking_goal[2]}
            json.dump(data, f, indent=4)

        self.camera_manager.save_video(cur_save_path)

        logging.info('saved sensor data for task %d in %s', self.task_index, str(cur_save_path))

    def save_unit_data(self, start, end, cur_save_path):
        for index in range(start, end):
            data_frame = self.batch_data_frames[index]

            for sensor in data_frame.keys():
                if sensor.startwith('rgb') or sensor.startswith('depth'):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))

            imu_data = data_frame['imu']
            gnss_data = data_frame['gnss']
            vehicle_transform = data_frame['veh_transform']
            vehicle_velocity = data_frame['veh_velocity']
            vehicle_control = data_frame['veh_control']

            data = {
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z,
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw,
                'roll': vehicle_transform.rotation.roll,
                'speed': 3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y**2 + vehicle_velocity.z**2),
                'Throttle': vehicle_control.throttle,
                'Steer': vehicle_control.steer,
                'Brake': vehicle_control.brake,
                'Reverse': vehicle_control.reverse,
                'Hand brake': vehicle_control.hand_brake,
                'Manual': vehicle_control.manual_gear_shift,
                'Gear': {-1: 'R', 0: 'N'}.get(vehicle_control.gear, vehicle_control.gear),
                'acc_x': imu_data.accelerometer.x,
                'acc_y': imu_data.accelerometer.y,
                'acc_z': imu_data.accelerometer.z,
                'gyr_x': imu_data.gyroscope.x,
                'gyr_y': imu_data.gyroscope.y,
                'gyr_z': imu_data.gyroscope.z,
                'compass': imu_data.compass,
                'lat': gnss_data.latitude,
                'lon': gnss_data.longitude
            }

            measurements_file = cur_save_path / 'measurements' / ('%04d.json' % index)
            with open(measurements_file, 'w') as f:
                json.dump(data, f, indent=4)

            def save_img(image, keyword=""):
                img_save = np.moveaxis(image, 0, 2)
                save_path = str(cur_save_path / 'topdown' / ('encoded_%04d' % index + keyword + '.png'))
                cv2.imwrite(save_path, img_save)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):

        sensors = self.sensor_list
        if self.camera_manager is not None:
            sensors.append(self.camera_manager.sensor)

        if self.collision_sensor is not None:
            sensors.append(self.collision_sensor.sensor)

        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        self.sensor_list.clear()
        self.camera_manager = None
        self.collision_sensor = None

        if self.player is not None:
            self.player.destroy()

        self.batch_data_frames.clear()
        self.sensor_data_frame.clear()
        self.sensor_queue = Queue()
        self.step = -1
        self.num_frames_in_goal = 0

        logging.info('destroying done.')

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None
        self.images = []

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def save_video(self, save_path):
        if len(self.images) == 0:
            return
        height, width = self.images[0].shape[:2]
        video_path = save_path / 'task.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
        for image in self.images:
            out.write(image)
        out.release()

    def clear_saved_images(self):
        self.images.clear()

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            image_array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            self.images.append(image_array)

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

