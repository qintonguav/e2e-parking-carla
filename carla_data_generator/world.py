import sys
import carla
import re
import random
import math
import numpy as np
import parking_position
from datetime import datetime
import pathlib
import logging
from queue import Queue, Empty
from threading import Thread
import json
import cv2

from camera_manager import CameraManager
from sensors import GnssSensor, IMUSensor, CollisionSensor, LaneInvasionSensor, RadarSensor
from tools import get_actor_display_name, encode_npy_to_pil
from bev_render import BevRender

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if sensor_queue.qsize() > 2000:
        #logging.info('sensor queue is full, throw sensor callbacks')
        return
    sensor_queue.put((sensor_data, sensor_name))

class World(object):
    def __init__(self, carla_world, hud, args):

        self.carla_world = carla_world
        settings = self.carla_world.get_settings()
        settings.fixed_delta_seconds = float(1. / 30)
        settings.synchronous_model = True
        self.carla_world.apply_settings(settings)

        self.parking_goal_index = 2
        self.parking_spawn_points = parking_position.parking_vehicle_locations_Town04.copy()
        self.target_parking_goal = self.parking_spawn_points[self.parking_goal_index]
        self.ego_transform = parking_position.EgoPostTown04(self.target_parking_goal)
        self.all_parking_goals = self.parking_spawn_points

        self.actor_role_name = args.rolename
        try:
            self.map = self.carla_world.get_map()
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
        self.veh2cam_dict = {}
        self.sensor_data_frame = {}
        self.batch_data_frames = []
        self.sensor_queue = Queue()
        self.cam_config = {}
        self.cam_center = None
        self.cam_specs = None
        self.intrinsic = None
        self.cam2pixel = None
        self.actor_list = []
        self.record_video = args.record_video
        self.bev_render = None

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
        self.carla_world.on_tick(self.hud.on_world_tick)
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

        self.bev_render = BevRender(self, 'cuda')
        self.init()


    def init(self):
        logging.info('***************init environment for task **************** {}'.format(self.task_index))
        self.destroy()

        self.init_static_vehicles()
        ego_transform = self.ego_transform.get_ego_transform()
        ego_vehicle_bp = self.carla_world.get_blueprint_library().find('vehicle.tesla.model3')
        self.player = self.carla_world.spawn_actor(ego_vehicle_bp, ego_transform)
        self.spectator = self.carla_world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=283.85, y=-210.039, z=35),
                                                     carla.Rotation(pitch=-90)))
        self.bev_render.set_player(self.player)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.next_weather()

        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self.record_video)
        self.camera_manager.transform_index = 0
        self.camera_manager.set_sensor(0, notify=False)

        self.setup_sensors()

        logging.info('*************init enviroment for task %d done*********** {}'.format(self.task_index))

    def init_static_vehicles(self):
        static_veh_num = random.randint(int(len(self.parking_spawn_points) / 3),
                                            len(self.parking_spawn_points) - 1)
        parking_points = self.parking_spawn_points.copy()

        random.shuffle(parking_points)

        # filter vehicles for 4 wheels
        blueprints = self.carla_world.get_blueprint_library().filter('vehicle')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        for index in range(static_veh_num):
            spawn_point = parking_points[index]

            if spawn_point == self.target_parking_goal:
                self.all_parking_goals.append(spawn_point)
                continue

            npc_transform = carla.Transform(spawn_point, rotation=random.choice([carla.Rotation(yaw=0),
                                                                                 carla.Rotation(yaw=180)]))
            npc_bp = random.choice(blueprints)
            npc = self.carla_world.try_spawn_actor(npc_bp, npc_transform)
            if npc is not None:
                npc.set_simulate_physics(False)
                self.actor_list.append(npc)
            else:
                logging.info('try generate npc fail!')
                self.all_parking_goals.append(spawn_point)

        for index in range(static_veh_num, len(parking_points)):
            self.all_parking_goals.append(parking_points[index])

        self.eval_parking_goal = [self.target_parking_goal.x, self.target_parking_goal.y, 180]


    def setup_sensors(self):
        # collision
        self.collision_sensor = CollisionSensor(self.player, self.hud)

        # gnss
        bp_gnss = self.carla_world.get_blueprint_library().find('sensor.other.gnss')
        gnss = self.carla_world.spawn_actor(bp_gnss, carla.Transform(), attach_to=self.player,
                                      attachment_type=carla.AttachmentType.Rigid)
        gnss.listen(lambda data: sensor_callback(data, self.sensor_queue, "gnss"))
        self.sensor_list.append(gnss)

        # imu
        bp_imu = self.carla_world.get_blueprint_library().find('sensor.other.imu')
        imu = self.carla_world.spawn_actor(bp_imu, carla.Transform(), attach_to=self.player,
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
        blueprint_library = self.carla_world.get_blueprint_library()
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
        cam = self.carla_world.spawn_actor(bp, sensor_transform, attach_to=self.player,
                                      attachment_type=carla.AttachmentType.Rigid)
        cam.listen(lambda data: sensor_callback(data, self.sensor_queue, sensor_id))
        self.sensor_list.append(cam)

    def restart(self):
        self.batch_data_frames = []
        self.sensor_queue = Queue()
        self.camera_manager.clear_saved_images()
        self.init()

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
            self.carla_world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.carla_world.load_map_layer(selected)

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

        self.carla_world.debug.draw_string(self.target_parking_goal, 'T',
                                     draw_shadow=True, color=carla.Color(255, 0, 0))

        self.step += 1
        if self.step % self.save_frequency == 0:
            self.sensor_data_frame['bev_state'] = self.get_bev_states()
            if len(self.batch_data_frames) > 2000:
                logging.info('too long for parking, this trip will be reset!')
            else:
                # print('len batch_data_frames', len(self.batch_data_frames))
                self.batch_data_frames.append(self.sensor_data_frame.copy())

        self.check_goal()

    def get_bev_states(self):
        elements = self.carla_world.get_actors().filter("*vehicle*")
        ts = [carla.Transform(element.get_transform().location, element.get_transform().rotation)
                for element in elements]

        return {
            "ego_t": carla.Transform(self.player.get_transform().location,
                                     self.player.get_transform().rotation),
            "vehicle_ts": ts
        }
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
                if sensor.startswith('rgb') or sensor.startswith('depth'):
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

            bev_view = self.bev_render.render_BEV(data_frame['bev_state'])
            img = encode_npy_to_pil(np.asarray(bev_view.squeeze().cpu()))
            save_img(img, "")

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

        for actor in self.actor_list:
            actor.destroy()
        self.actor_list.clear()

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