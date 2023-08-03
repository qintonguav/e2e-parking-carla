import random
import carla

parking_vehicle_locations_Town04 = [
    carla.Location(x=298.5, y=-235.73, z=0.3),
]


class EgoPostTown04:
    def __init__(self, parking_goal: carla.Location):
        self.x = 285.6
        self.z = 0.32682

        self.yaw_to_r = 90.0
        self.yaw_to_l = -90.0

        self.goal_y = parking_goal.y
        self.y_max = self.goal_y + 8
        self.y_min = self.goal_y - 8

    def update_goal_y(self, goal_y):
        self.goal_y = goal_y
        self.y_max = self.goal_y + 8
        self.y_min = self.goal_y - 8

    def get_ego_transform(self):
        y = random.uniform(self.y_min, self.y_max)
        yaw = self.yaw_to_r if y < self.goal_y else self.yaw_to_l

        return carla.Transform(carla.Location(x=self.x, y=y, z=self.z),
                               carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))