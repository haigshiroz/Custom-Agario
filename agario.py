import argparse

parser = argparse.ArgumentParser(
    description="Python implementation of game agar.io")
parser.add_argument(
    '-wt', '--width',
    dest='width',
    type=int,
    default=900,
    help='screen width')
parser.add_argument(
    '-ht', '--height',
    dest='height',
    type=int,
    default=600,
    help='screen height')
parser.add_argument(
    '-s', '--server',
    action='store_true',
    dest='server',
    help='start game server')
parser.add_argument(
    '-p', '--port',
    dest='port',
    type=int,
    default=9999,
    help='port number for server')
parser.add_argument(
    '-m', '--manual',
    action='store_true',
    dest='manual',
    help='control the player manually')

train_test_group = parser.add_mutually_exclusive_group()
train_test_group.add_argument(
    '--train',
    action='store_true',
    help='run in training mode (epsilon=1.0)')
train_test_group.add_argument(
    '--test',
    action='store_true',
    help='run in testing mode (epsilon=0.0)')

train_subgroup = parser.add_argument_group('training options')
train_subgroup.add_argument(
    '--empty',
    action='store_true',
    help='initialize empty Q-tables (default for training)')
train_subgroup.add_argument(
    '--history',
    action='store_true',
    help='continue training from existing Q-tables')

args = parser.parse_args()


if args.test and (args.empty or args.history):
    print("Warning: --empty/--history ignored in test mode")

if args.server:
    import game.network.server as server
    server.start(host='0.0.0.0', port=args.port)
else:
    import game.network.client as client

    train_mode = args.train

    load_from_history = args.history if args.train else False
    client.start(
        args.width, 
        args.height, 
        manual=args.manual, 
        train_mode=train_mode,
        load_from_history=load_from_history
    )