import torch
import pygame
import numpy as np
import torch.nn.functional as F
import carla
from PIL import Image
import os

# Global Flags
PIXELS_PER_METER = 5

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_TRAFFIC_RED = pygame.Color(255, 0, 0)
COLOR_TRAFFIC_YELLOW = pygame.Color(0, 255, 0)
COLOR_TRAFFIC_GREEN = pygame.Color(0, 0, 255)


class BevRender:
    def __init__(self, world, device):
        self._device = device
        self._world = world.world
        self._vehicle = world.player
        self._actors = None

        hd_map = self._map = self._world.get_map().to_opendrive()
        self.world_map = carla.Map("RouteMap", hd_map)

        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self._device)

        # create map for renderer
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device=self._device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self._device, dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = Renderer(world_offset, self.map_dims, data_generation=True, device=self._device)

        self.detection_radius = 50.0

    def set_player(self, player):
        self._vehicle = player

    def render_BEV(self):
        ego_t = self._vehicle.get_transform()
        semantic_grid = self.global_map

        # fetch local birdview per agent
        ego_pos = torch.tensor([ego_t.location.x, ego_t.location.y],
                               device=self._device, dtype=torch.float32)
        ego_yaw = torch.tensor([ego_t.rotation.yaw / 180 * np.pi], device=self._device,
                               dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if vehicle.get_location().distance(ego_t.location) < self.detection_radius:
                if vehicle.id != self._vehicle.id:
                    pos = torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y],
                                       device=self._device, dtype=torch.float32)
                    yaw = torch.tensor([vehicle.get_transform().rotation.yaw / 180 * np.pi], device=self._device,
                                       dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device=self._device)
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        return birdview

    def get_bev_states(self):
        def get_element_ts(keyword):
            elements = self._world.get_actors().filter(keyword)
            ts = [carla.Transform(element.get_transform().location, element.get_transform().rotation)
                  for element in elements]
            return ts

        return {
            "ego_t": carla.Transform(self._vehicle.get_transform().location,
                                     self._vehicle.get_transform().rotation),
            "vehicle_ts": get_element_ts("*vehicle*"),
        }

    def render_BEV_from_state(self, state):

        ego_t = state["ego_t"]

        semantic_grid = self.global_map

        # fetch local birdview per agent
        ego_pos = torch.tensor([ego_t.location.x, ego_t.location.y],
                               device=self._device, dtype=torch.float32)
        ego_yaw = torch.tensor([ego_t.rotation.yaw / 180 * np.pi], device=self._device,
                               dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        # vehicle is only used for id and bbox, which is never changed during the play
        # WARNING: the order of filter('*vehicle*') can't be changed or bug occurs here
        for vehicle_t, vehicle in zip(state["vehicle_ts"], self._world.get_actors().filter('*vehicle*')):
            if vehicle.id != self._vehicle.id:
                if vehicle_t.location.distance(ego_t.location) < self.detection_radius:
                    pos = torch.tensor([vehicle_t.location.x, vehicle_t.location.y],
                                       device=self._device, dtype=torch.float32)
                    yaw = torch.tensor([vehicle_t.rotation.yaw / 180 * np.pi], device=self._device,
                                       dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device=self._device)
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        return birdview


class Renderer():
    def __init__(self, map_offset, map_dims, data_generation=True, device='cpu'):
        self.args = {'device': device}
        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0  # ego car is central
            self.local_view_dims = (500, 500)
            self.crop_dims = (500, 500)
        else:
            self.PIXELS_AHEAD_VEHICLE = 100 + 10  # 10 is the weird shift the crop does in LBC
            self.local_view_dims = (320, 320)
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.local_view_scale = (
            self.local_view_dims[1] / self.map_dims[1],
            self.local_view_dims[0] / self.map_dims[0]
        )
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

    def world_to_pix(self, pos):
        pos_px = (pos - self.map_offset) * PIXELS_PER_METER

        return pos_px

    def world_to_pix_crop_batched(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        # TODO: should be able to handle batches

        # # FIXME: why do we need to do this everywhere?
        crop_yaw = crop_yaw + np.pi / 2
        batch_size = crop_pos.shape[0]

        # transform to crop pose
        rotation = torch.stack(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
             torch.sin(crop_yaw), torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1).unsqueeze(1) @ \
                       (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2],
                                                  device=self.args['device'])

        return pos_px_crop

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        # TODO: should be able to handle batches

        # # FIXME: why do we need to do this everywhere?
        crop_yaw = crop_yaw + np.pi / 2

        # transform to crop pose
        rotation = torch.tensor(
            [[torch.cos(crop_yaw), -torch.sin(crop_yaw)],
             [torch.sin(crop_yaw), torch.cos(crop_yaw)]],
            device=self.args['device'],
        )

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2],
                                                  device=self.args['device'])

        return pos_px_crop

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / torch.tensor([self.map_dims[1], self.map_dims[0]], device=self.args['device'])

        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def render_agent(self, grid, vehicle, position, orientation):
        """
        """
        orientation = orientation - np.pi / 2  # TODO
        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # convert position from world to relative image coordinates
        position = self.world_to_rel(position) * -1

        # TODO: build composite transform directly
        # build individual transforms
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        rotation_transform = torch.tensor(
            [[torch.cos(orientation), torch.sin(orientation), 0],
             [-torch.sin(orientation), torch.cos(orientation), 0],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, position[0]],
             [0, 1, position[1]],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        # chain transforms
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # expects Nx2x3
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        grid[:, 5, ...] += vehicle_rendering.squeeze()

        return grid

    def render_agent_bv(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
            state=None,  # traffic light_state
    ):
        """
        """
        # FIXME: why do we need to do this everywhere?
        orientation = orientation + np.pi / 2

        # Only render if visible in local view
        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)

        # to centered relative coordinates for STN
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])  # normalize over h and w
        pos_rel_bv = pos_rel_bv * 2 - 1  # change domain from [0, 1] to [-1, 1]
        pos_rel_bv = pos_rel_bv * -1  # Because the STN coordinates are weird

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # TODO: build composite transform directly
        # build individual transforms
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # this is the inverse of the rotation matrix for the visibility check
        # because now we want crop coordinates instead of world coordinates
        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [[torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
             [- torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)  # .to(self.args['device'])

        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]],
             [0, 1, pos_rel_bv[1]],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)  # .to(self.args['device'])

        # chain transforms
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # expects Nx2x3
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        if state == 'Green':
            channel = 4
        elif state == 'Yellow':
            channel = 3
        elif state == 'Red':
            channel = 2

        grid[:, channel, ...] += vehicle_rendering.squeeze()

    def render_agent_bv_batched(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
    ):
        """
        """
        # FIXME: why do we need to do this everywhere?
        orientation = orientation + np.pi / 2
        batch_size = position.shape[0]

        pos_pix_bv = self.world_to_pix_crop_batched(position, grid_pos, grid_orientation)

        # to centered relative coordinates for STN
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])  # normalize over h and w
        pos_rel_bv = pos_rel_bv * 2 - 1  # change domain from [0, 1] to [-1, 1]
        pos_rel_bv = pos_rel_bv * -1  # Because the STN coordinates are weird

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # TODO: build composite transform directly
        # build individual transforms
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3).expand(batch_size, -1, -1)

        # this is the inverse of the rotation matrix for the visibility check
        # because now we want crop coordinates instead of world coordinates
        grid_orientation = grid_orientation + np.pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)
        rotation_transform = torch.stack(
            [torch.cos(angle_delta), torch.sin(angle_delta), zeros,
             -torch.sin(angle_delta), torch.cos(angle_delta), zeros,
             zeros, zeros, ones],
            dim=-1
        ).view(batch_size, 3, 3)

        translation_transform = torch.stack(
            [ones, zeros, pos_rel_bv[..., 0:1],
             zeros, ones, pos_rel_bv[..., 1:2],
             zeros, zeros, ones],
            dim=-1,
        ).view(batch_size, 3, 3)

        # chain transforms
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # expects Nx2x3
            (batch_size, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        for i in range(batch_size):
            grid[:, int(channel[i].item()), ...] += vehicle_rendering[i].squeeze()

    def get_local_birdview(self, grid, position, orientation):
        """
        """

        # convert position from world to relative image coordinates
        position = self.world_to_rel(position)  # , self.map_dims)
        # FIXME: Inconsistent with global rendering function.
        orientation = orientation + np.pi / 2  # + np.pi

        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
             [0, self.crop_scale[0], 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # FIXME: Inconsistent with global rendering function.
        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0],
             [torch.sin(orientation), torch.cos(orientation), 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # shift cropping position so ego agent is at bottom boundary, including
        # this weird pixel shift that LBC does for some reason
        shift = torch.tensor([0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]], device=self.args['device'])
        position = position + rotation_transform[0, 0:2, 0:2] @ shift

        translation_transform = torch.tensor(
            [[1, 0, position[0] / self.crop_scale[0]],
             [0, 1, position[1] / self.crop_scale[1]],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # chain tansforms
        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (1, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        local_view = F.grid_sample(
            grid,
            affine_grid,
            align_corners=True,
        )

        return local_view

    def step(self, actions):
        """
        """
        # 1. update ego agent
        print(self.ego.state, actions)
        # actions['steer'] = torch.Tensor([0.])
        self.ego.set_state(self.ego.motion_model(self.ego.state, actions=actions))
        # self.ego.state['yaw'] *= 0
        # self.ego.state['yaw'] += np.pi * self.timestep / 100
        # self.ego.set_state(self.ego.state)
        self.adv.set_state(self.adv.motion_model(self.adv.state))

        # 2. update adversarial agents
        # ...
        self.timestep += 1

    def visualize_grid(self, grid, type='LTS_Reduced'):
        """
        """
        if type == 'LTS_Reduced':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
            ]

        elif type == 'Trajectory_planner':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                # (0, 0, 142), # vehicle
                # (220, 20, 60), # pedestrian
            ]

        elif type == 'LTS_Full':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                (204, 6, 5),  # red light
                (250, 210, 1),  # yellow light
                (39, 232, 51),  # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
            ]
        elif type == 'LTS_FullFuture':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                (204, 6, 5),  # red light
                (250, 210, 1),  # yellow light
                (39, 232, 51),  # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],  # vehicle future
            ]
        elif type == 'LTS_ReducedFuture':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],  # vehicle future
            ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [0, 47, 0]

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def bev_to_gray_img(self, grid):
        """
        """
        colors = [
            1,  # road
            2,  # lane
            3,  # red light
            4,  # yellow light
            5,  # green light
            6,  # vehicle
            7,  # pedestrian
        ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4]), dtype=np.uint8)

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img


class ModuleManager(object):
    def __init__(self):
        self.modules = []

    def register_module(self, module):
        self.modules.append(module)

    def clear_modules(self):
        del self.modules[:]

    def tick(self, clock):
        # Update all the modules
        for module in self.modules:
            module.tick(clock)

    def render(self, display, snapshot=None):
        display.fill(COLOR_ALUMINIUM_4)
        for module in self.modules:
            module.render(display, snapshot=snapshot)

    def get_module(self, name):
        for module in self.modules:
            if module.name == name:
                return module

    def start_modules(self):
        for module in self.modules:
            module.start()


module_manager = ModuleManager()


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        module_manager.clear_modules()

        pygame.init()
        display = pygame.display.set_mode((320, 320), 0, 32)
        # pygame.display.flip()

        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.draw_road_map(
            self.big_map_surface, self.big_lane_surface,
            carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        # map_surface.fill(COLOR_ALUMINIUM_4)
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def draw_lane_marking(surface, points, solid=True):
            if solid and len(points) > 1:
                # pygame.draw.lines(surface, COLOR_ORANGE_0, False, points, 2)
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
            else:
                broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    # pygame.draw.lines(surface, COLOR_ORANGE_0, False, line, 2)
                    pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            start = transform.location
            end = start + 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def draw_stop(surface, font_surface, transform, color=COLOR_ALUMINIUM_2):
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1))
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1))

                # Dian: Do not draw them arrows
                # for n, wp in enumerate(waypoints):
                #     if (n % 400) == 0:
                #         draw_arrow(map_surface, wp.transform)

        actors = carla_world.get_actors()
        stops_transform = [actor.get_transform() for actor in actors if 'stop' in actor.type_id]
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)
        font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        font_surface = pygame.transform.scale(font_surface, (font_surface.get_width(), font_surface.get_height() * 2))

        # Dian: do not draw stop sign

        # for stop in stops_transform:
        #     draw_stop(map_surface,font_surface, stop)

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
