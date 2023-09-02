import argparse
import logging
import carla
import pygame

from carla_data_generator.network_evaluator import NetworkEvaluator
from carla_data_generator.keyboard_control import KeyboardControl
from parking_agent import ParkingAgent
from parking_agent import show_control_info


def game_loop(args):
    pygame.init()
    pygame.font.init()
    network_evaluator = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        network_evaluator = NetworkEvaluator(carla_world, args)
        parking_agent = ParkingAgent(network_evaluator, args)
        controller = KeyboardControl(network_evaluator.world)

        steer_wheel_img = pygame.image.load("./resource/steer_wheel.png")
        font = pygame.font.Font(pygame.font.get_default_font(), 25)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        while True:
            network_evaluator.world_tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, network_evaluator.world, clock):
                return
            parking_agent.tick()
            show_control_info(display, parking_agent.get_eva_control(), steer_wheel_img,
                              args.width, args.height, font)
            network_evaluator.tick(clock)
            network_evaluator.render(display)
            pygame.display.flip()

    finally:

        if network_evaluator:
            client.stop_recorder()

        if network_evaluator is not None:
            network_evaluator.destroy()

        pygame.quit()


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Data Generation')
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
        default='860x480',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--gamma',
        default=0.0,
        type=float,
        help='Gamma correction of the camera (default: 0.0)')
    argparser.add_argument(
        '--model_path',
        default='./ckpt/model.ckpt',
        help='path to model.ckpt')
    argparser.add_argument(
        '--model_config_path',
        default='./config/training.yaml',
        help='path to model training.yaml')
    argparser.add_argument(
        '--eva_epochs',
        default=4,
        type=int,
        help='number of eva epochs (default: 4')
    argparser.add_argument(
        '--eva_task_nums',
        default=16,
        type=int,
        help='number of parking slot task (default: 16')
    argparser.add_argument(
        '--eva_parking_nums',
        default=6,
        type=int,
        help='number of parking nums for every slot (default: 6')
    argparser.add_argument(
        '--map',
        default='Town04_Opt',
        help='map of carla (default: Town04_Opt)',
        choices=['Town04_Opt', 'Town05_Opt'])
    argparser.add_argument(
        '--shuffle_veh',
        default=True,
        type=str2bool,
        help='shuffle static vehicles between tasks (default: False)')
    argparser.add_argument(
        '--shuffle_weather',
        default=False,
        type=str2bool,
        help='shuffle weather between tasks (default: False)')
    argparser.add_argument(
        '--random_seed',
        default=11,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cuda',
        help='device used for BEV Rendering (default: cpu)',
        choices=['cpu', 'cuda'])
    argparser.add_argument(
        '--show_eva_imgs',
        default=False,
        type=str2bool,
        help='show eva figure in eva model (default: False)')
    argparser.add_argument(
        '--eva_result_path',
        default='./eva_result',
        help='path to save eva csv file')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()
