#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import os
import argparse
import logging
import time
import pygame
from world import World
from keyboard_control import KeyboardControl
from hud import HUD
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        hud = HUD(args.width, args.height)
        world = World(carla_world, hud, args)
        controller = KeyboardControl(world, False)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        while True:
            world.carla_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if world:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--gama',
        default=0.0,
        type=float,
        help='Gamma correction of the camera (default: 0.0)')
    argparser.add_argument(
        '--save_path',
        default='./runs/',
        help='path to save sensor data (default: ./runs/)')
    argparser.add_argument(
        '--task_num',
        default=16,
        type=int,
        help='number of parking task (default: 5)')
    argparser.add_argument(
        '--map',
        default='Town04_Opt',
        help='map of carla (default: Town04_Opt)',
        choices=['Town04_Opt', 'Town05_Opt'])
    argparser.add_argument(
        '--shuffle_veh',
        default=False,
        help='shuffle static vehicles between tasks (default: False)')
    argparser.add_argument(
        '--shuffle_weather',
        default=False,
        help='shuffle weather between tasks (default: False)')
    argparser.add_argument(
        '--random_seedr',
        default=1,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 1)')
    argparser.add_argument(
        '--bev_render_device',
        default='cuda',
        help='device used for BEV Rendering (default: cuda)',
        choices=['cpu', 'cuda'])
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--record_video',
        default=True,
        type=str2bool,
        help='record parking video (default: True)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)


    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
