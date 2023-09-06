import logging
import sys
import random
import re
from queue import Queue, Empty

import numpy as np
import carla

from data_generation.hud import HUD, get_actor_display_name
from data_generation.sensors import CollisionSensor, CameraManager
from data_generation import parking_position
from data_generation.bev_render import BevRender

parking_vehicle_rotation = [
    carla.Rotation(yaw=180),
    carla.Rotation(yaw=0)
]


def find_weather_presets():
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), x) for x in presets]


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


class World(object):
    def __init__(self, carla_world, args):

        # set carla in sync + fixed time step mode
        self._world = carla_world
        settings = self._world.get_settings()
        settings.fixed_delta_seconds = float(1 / 30)  # 30 FPS
        settings.synchronous_mode = True
        self._world.apply_settings(settings)

        if args.map == 'Town04_Opt':
            self._parking_spawn_points = parking_position.parking_vehicle_locations_Town04.copy()
        else:
            logging.error('Invalid map %s', args.map)
            sys.exit(1)

        try:
            self._map = self._world.get_map()
        except RuntimeError as error:
            logging.error('RuntimeError: {}'.format(error))
            logging.error('The server could not send the OpenDRIVE (.xodr) file:')
            logging.error('Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self._step = -1

        # parking goal
        self._all_parking_goals = []

        self._hud = HUD(args.width, args.height)
        self._keyboard_restart_task = False

        self._collision_sensor = None
        self._camera_manager = None
        self._weather_presets = find_weather_presets()
        del self._weather_presets[5]
        self._weather_index = 0
        self._gamma = args.gamma

        # other vehicles in parking lot
        self._actor_list = []
        # ego vehicle
        self._player = None
        # spectator
        self._spectator = None
        # list of sensors on ego_vehicle
        self._sensor_list = []
        # sensor data queue on each frame; used for sensor callback
        self._sensor_queue = Queue()
        # sensor data on each frame
        self._sensor_data_frame = {}

        self._world.on_tick(self._hud.on_world_tick)

        self._shuffle_static_vhe = args.shuffle_veh
        self._shuffle_weather = args.shuffle_weather

        # Render BEV Segmentation map
        self._bev_render_device = args.bev_render_device
        self._bev_render = None

        # camera
        self._cam_config = {}
        self._cam_center = None
        self._cam_specs = {}
        self._intrinsic = None
        self._cam2pixel = None
        self._veh2cam_dict = {}

        self._x_diff_to_goal = 0
        self._y_diff_to_goal = 0
        self._distance_diff_to_goal = 0
        self._rotation_diff_to_goal = 0

        self._need_init_ego_state = True

    def restart(self, seed, target_index, ego_transform):

        # spawn static vehicles in the parking lot
        if self._shuffle_static_vhe:
            self.init_static_npc(seed, target_index)

        # init the player position
        self._player.set_transform(ego_transform)
        self._player.apply_control(carla.VehicleControl())
        self._player.set_target_velocity(carla.Vector3D(0, 0, 0))
        # self._spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=50),
        #                                               carla.Rotation(pitch=-90)))

        actor_type = get_actor_display_name(self._player)
        self._hud.notification(actor_type)

        if self._shuffle_weather:
            self.next_weather()

        self._camera_manager.clear_saved_images()

        self._need_init_ego_state = True

    def init_ego_vehicle(self, ego_transform):

        ego_vehicle_bp = self._world.get_blueprint_library().find('vehicle.tesla.model3')
        self._player = self._world.spawn_actor(ego_vehicle_bp, ego_transform)

        self._bev_render = BevRender(self, self._bev_render_device)

        self._spectator = self._world.get_spectator()
        # self._spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=50),
        #                                               carla.Rotation(pitch=-90)))
        self._spectator.set_transform(carla.Transform(carla.Location(x=283.825165, y=-210.039487, z=35.0),
                                                      carla.Rotation(pitch=-90)))

        actor_type = get_actor_display_name(self._player)
        self._hud.notification(actor_type)

    def init_static_npc(self, seed, target_index):

        random.seed(seed)

        target_parking_goal = self._parking_spawn_points[target_index]

        # get all possible spawn points in the parking lot
        logging.info("total parking points: %d", len(self._parking_spawn_points))

        static_vehicle_num = random.randint(int(len(self._parking_spawn_points) / 3),
                                            len(self._parking_spawn_points) - 1)
        logging.info("spawn %d static vehicle in parking lot", static_vehicle_num)

        parking_points_copy = self._parking_spawn_points.copy()
        random.shuffle(parking_points_copy)

        # choose only 4 wheels vehicles
        blueprints = self._world.get_blueprint_library().filter('vehicle')
        blueprints = [x for x in blueprints if self.valid_vehicle(x)]

        # spawn npc vehicles
        for index in range(static_vehicle_num):
            spawn_point = parking_points_copy[index]

            if spawn_point == target_parking_goal:
                self._all_parking_goals.append(spawn_point)
                continue

            npc_transform = carla.Transform(spawn_point, rotation=random.choice(parking_vehicle_rotation))
            npc_bp = random.choice(blueprints)
            npc = self._world.try_spawn_actor(npc_bp, npc_transform)
            if npc is not None:
                npc.set_simulate_physics(False)
                self._actor_list.append(npc)
            else:
                # logging.info("try_spawn_actor %s at (%.3f, %.3f, %.3f) failed!",
                #              npc_bp.id, spawn_point.x, spawn_point.y, spawn_point.z)
                self._all_parking_goals.append(spawn_point)

        # set parking goal
        for index in range(static_vehicle_num, len(parking_points_copy)):
            self._all_parking_goals.append(parking_points_copy[index])

        logging.info('set %d parking goal', len(self._all_parking_goals))

    def init_sensors(self):
        self._collision_sensor = CollisionSensor(self._player, self._hud)
        self._camera_manager = CameraManager(self._player, self._hud, self._gamma)
        self._camera_manager.transform_index = 0
        self._camera_manager.set_sensor(0, notify=False)

        # init sensors on ego_vehicle
        self.setup_sensors()

    def valid_vehicle(self, vehicle_bp):
        # is_no_bmw_isetta = (vehicle_bp.id != 'vehicle.bmw.isetta')
        # is_no_cybertruck = (vehicle_bp.id != 'vehicle.tesla.cybertruck')
        # is_no_carlacola = (vehicle_bp.id != 'vehicle.carlamotors.carlacola')
        is_four_wheels = int(vehicle_bp.get_attribute('number_of_wheels')) == 4
        return is_four_wheels

    def soft_restart(self, ego_transform):

        # clear cache
        self.sensor_data_frame.clear()
        self._sensor_queue = Queue()
        self._step = -1

        # init the player position
        self._player.set_transform(ego_transform)
        self._player.apply_control(carla.VehicleControl())
        self._player.set_target_velocity(carla.Vector3D(0, 0, 0))
        # self._spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=50),
        #                                               carla.Rotation(pitch=-90)))

        self._camera_manager.clear_saved_images()

        self._need_init_ego_state = True

    def setup_sensors(self):

        # gnss
        bp_gnss = self._world.get_blueprint_library().find('sensor.other.gnss')
        gnss = self._world.spawn_actor(bp_gnss, carla.Transform(), attach_to=self._player,
                                       attachment_type=carla.AttachmentType.Rigid)
        gnss.listen(lambda data: sensor_callback(data, self._sensor_queue, "gnss"))
        self._sensor_list.append(gnss)

        # imu
        bp_imu = self._world.get_blueprint_library().find('sensor.other.imu')
        imu = self._world.spawn_actor(bp_imu, carla.Transform(), attach_to=self._player,
                                      attachment_type=carla.AttachmentType.Rigid)
        imu.listen(lambda data: sensor_callback(data, self._sensor_queue, "imu"))
        self._sensor_list.append(imu)

        # camera
        self._cam_config = {
            'width': 400,
            'height': 300,
            'fov': 100,
        }
        self._cam_center = np.array([self._cam_config['width'] / 2.0, self._cam_config['height'] / 2.0])
        self._cam_specs = {
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

        for key, value in self._cam_specs.items():
            self.spawn_rgb_camera(key, value)

        # intrinsic
        w = self._cam_config['width']
        h = self._cam_config['height']
        fov = self._cam_config['fov']
        f = w / (2 * np.tan(fov * np.pi / 360))
        Cu = w / 2
        Cv = h / 2
        self._intrinsic = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=np.float)

        self._cam2pixel = np.array([[0, 1, 0, 0],
                                    [0, 0, -1, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 1]], dtype=float)

        for cam_id, cam_spec in self._cam_specs.items():
            if cam_id.startswith('rgb'):
                cam2veh = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                          carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                         roll=cam_spec['roll']))
                veh2cam = self._cam2pixel @ np.array(cam2veh.get_inverse_matrix())
                self._veh2cam_dict[cam_id] = veh2cam

    def spawn_rgb_camera(self, sensor_id, sensor_spec):
        blueprint_library = self._world.get_blueprint_library()
        bp = blueprint_library.find(sensor_spec['type'])
        bp.set_attribute('image_size_x', str(self._cam_config['width']))
        bp.set_attribute('image_size_y', str(self._cam_config['height']))
        bp.set_attribute('fov', str(self._cam_config['fov']))
        sensor_location = carla.Location(x=sensor_spec['x'],
                                         y=sensor_spec['y'],
                                         z=sensor_spec['z'])
        sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                         roll=sensor_spec['roll'],
                                         yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        cam = self._world.spawn_actor(bp, sensor_transform, attach_to=self._player,
                                      attachment_type=carla.AttachmentType.Rigid)
        cam.listen(lambda data: sensor_callback(data, self._sensor_queue, sensor_id))
        self._sensor_list.append(cam)

    def spawn_lidar(self, lidar_specs):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('rotation_frequency', str(lidar_specs['rotation_frequency']))
        lidar_bp.set_attribute('points_per_second', str(lidar_specs['points_per_second']))
        lidar_bp.set_attribute('channels', str(lidar_specs['channels']))
        lidar_bp.set_attribute('upper_fov', str(lidar_specs['upper_fov']))
        lidar_bp.set_attribute('atmosphere_attenuation_rate', str(lidar_specs['atmosphere_attenuation_rate']))
        lidar_bp.set_attribute('dropoff_general_rate', str(lidar_specs['dropoff_general_rate']))
        lidar_bp.set_attribute('dropoff_intensity_limit', str(lidar_specs['dropoff_intensity_limit']))
        lidar_bp.set_attribute('dropoff_zero_intensity', str(lidar_specs['dropoff_zero_intensity']))
        lidar_location = carla.Location(x=lidar_specs['x'],
                                        y=lidar_specs['y'],
                                        z=lidar_specs['z'])
        lidar_rotation = carla.Rotation(pitch=lidar_specs['pitch'],
                                        roll=lidar_specs['roll'],
                                        yaw=lidar_specs['yaw'])
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.player,
                                       attachment_type=carla.AttachmentType.Rigid)
        lidar.listen(lambda data: sensor_callback(data, self._sensor_queue, "lidar"))
        self._sensor_list.append(lidar)

    def next_weather(self, reverse=False):
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self._hud.notification('Weather: %s' % preset[1])
        self._player.get_world().set_weather(preset[0])
        self._weather_index += -1 if reverse else 1

    def world_tick(self):
        self._world.tick()

    @property
    def map(self):
        return self._map

    @property
    def step(self):
        return self._step

    @property
    def player(self):
        return self._player

    @property
    def world(self):
        return self._world

    @property
    def hud(self):
        return self._hud

    @property
    def camera_manager(self):
        return self._camera_manager

    @property
    def sensor_data_frame(self):
        return self._sensor_data_frame

    @property
    def bev_state(self):
        return self._bev_render.get_bev_states()

    @property
    def ego_transform(self):
        return self._player.get_transform()

    @property
    def all_parking_goals(self):
        return self._all_parking_goals

    @property
    def x_diff_to_goal(self):
        return self._x_diff_to_goal

    @x_diff_to_goal.setter
    def x_diff_to_goal(self, diff):
        self._x_diff_to_goal = diff

    @property
    def y_diff_to_goal(self):
        return self._y_diff_to_goal

    @y_diff_to_goal.setter
    def y_diff_to_goal(self, diff):
        self._y_diff_to_goal = diff

    @property
    def distance_diff_to_goal(self):
        return self._distance_diff_to_goal

    @distance_diff_to_goal.setter
    def distance_diff_to_goal(self, diff):
        self._distance_diff_to_goal = diff

    @property
    def rotation_diff_to_goal(self):
        return self._rotation_diff_to_goal

    @rotation_diff_to_goal.setter
    def rotation_diff_to_goal(self, diff):
        self._rotation_diff_to_goal = diff

    @property
    def cam_config(self):
        return self._cam_config

    @property
    def intrinsic(self):
        return self._intrinsic

    @property
    def veh2cam_dict(self):
        return self._veh2cam_dict

    @property
    def keyboard_restart_task(self):
        return self._keyboard_restart_task

    @keyboard_restart_task.setter
    def keyboard_restart_task(self, activate):
        self._keyboard_restart_task = activate

    @property
    def need_init_ego_state(self):
        return self._need_init_ego_state

    @need_init_ego_state.setter
    def need_init_ego_state(self, need_init_ego_state):
        self._need_init_ego_state = need_init_ego_state

    def render_BEV_from_state(self, state):
        return self._bev_render.render_BEV_from_state(state)

    def render_BEV(self):
        return self._bev_render.render_BEV()

    def save_video(self, path):
        self._camera_manager.save_video(path)

    def tick(self, clock, target_index):
        try:
            # collect vehicle data at each frame
            t = self._player.get_transform()
            v = self._player.get_velocity()
            c = self._player.get_control()
            self._sensor_data_frame['veh_transfrom'] = t
            self._sensor_data_frame['veh_velocity'] = v
            self._sensor_data_frame['veh_control'] = c

            # collect sensor data at each frame
            for i in range(0, len(self._sensor_list)):
                s_data = self._sensor_queue.get(block=True, timeout=1.0)
                self._sensor_data_frame[s_data[1]] = s_data[0]

                # if s_data[1] == 'rgb_left':
                #     target_ego = convert_veh_coord(self.target_parking_goal.x, self.target_parking_goal.y, self.target_parking_goal.z, t)
                #     self.image_process(target_ego, cam_id=s_data[1], image=s_data[0])

        except Empty:
            logging.error("Some of the sensor information is missed")

        self._hud.tick(self, clock)

        # draw target lot char T
        target_parking_goal = self._parking_spawn_points[target_index]
        self._world.debug.draw_string(target_parking_goal, 'T', draw_shadow=True, color=carla.Color(255, 0, 0))

        # update spectator
        t = self._player.get_transform().location
        self._spectator.set_transform(carla.Transform(t + carla.Location(z=30), carla.Rotation(pitch=-90)))

        self._step += 1

        # detect collision
        if self._collision_sensor.is_collision or self._keyboard_restart_task:
            self._collision_sensor.is_collision = False
            self._keyboard_restart_task = False
            return True

        return False

    def render(self, display):
        self._camera_manager.render(display)
        self._hud.render(display)

    def destroy(self):
        sensors = self._sensor_list
        if self._camera_manager is not None:
            sensors.append(self._camera_manager.sensor)
        if self._collision_sensor is not None:
            sensors.append(self._collision_sensor.sensor)
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self._sensor_list.clear()
        self._camera_manager = None
        self._collision_sensor = None

        if self._player is not None:
            self._player.destroy()

        for actor in self._actor_list:
            actor.destroy()
        self._actor_list.clear()

        self._sensor_data_frame.clear()
        self._sensor_queue = Queue()
        self._step = -1

    def soft_destroy(self):
        if self._shuffle_static_vhe:
            for actor in self._actor_list:
                actor.destroy()
            self._actor_list.clear()

        self._sensor_data_frame.clear()
        self._sensor_queue = Queue()
        self._step = -1
        if self._shuffle_static_vhe:
            self._all_parking_goals.clear()
