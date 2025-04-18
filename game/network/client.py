import socket
import pickle
import pygame
import time

from loguru import logger
from .menu import MyMenu
from .msgtype import MsgType
from .. import View

from ..helper.move_player import get_direction_and_keys
from train_q_learning import QTraining

BACKGROUND_COLOR = (40, 0, 40)
MAX_UDP_SIZE = 9000  

def receive_large_data(sock):
    """Receives fragmented UDP packets and reconstructs data"""
    chunks = {}
    expected_chunks = None

    while True:
        data, _ = sock.recvfrom(MAX_UDP_SIZE)
        i, total, chunk = pickle.loads(data)

        chunks[i] = chunk
        if expected_chunks is None:
            expected_chunks = total

        if len(chunks) == expected_chunks:
            break

    raw_data = b''.join(chunks[i] for i in range(expected_chunks))
    return pickle.loads(raw_data)

class GameConnection():
    def __init__(self, screen, q_trainer: QTraining):
        self.screen = screen
        self.player_id = None
        self.is_in_lobby = False
        self.host = None
        self.port = None
        self.addr_string = None
        self.q_trainer = q_trainer
        
    def connect_to_game(self, get_attrs, manual=False):
        attrs = get_attrs()
        self.addr_string = attrs['addr']
        nick = attrs['nick']
        self.host, self.port = self.addr_string.split(':')
        self.port = int(self.port)

        try:
            # Send nickname
            msg = pickle.dumps({'type': MsgType.CONNECT, 'data': nick})
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg, (self.host, self.port))
            logger.debug('Sending {} to {}'.format(msg, self.addr_string))

            # Receive player ID
            data = sock.recv(4096)
            self.player_id = pickle.loads(data)
            logger.debug('Received {!r} from {}'.format(self.player_id, self.addr_string))

            # Create view
            view = View(self.screen, None, None)

            # --- AI initializ state and action variable---
            action_timer = 0
            prev_state = 0
            action, direction_to_go = (1, 1)
            should_split = False
            players_reward = 0
            while True:
                keys = []
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        keys.append(event.key)

                if manual:
                    mouse_pos = view.mouse_pos_to_polar()
                else:
                    if action_timer == 5:
                        action, direction_to_go = self.q_trainer.get_action_and_direction(prev_state)
                        should_split = action == 4
                        # TODO: Remove if we can handle the split state
                        should_split = False
                        self.q_trainer.end_of_episode()

                    mouse_pos, keys = get_direction_and_keys(direction=direction_to_go, split=should_split)
                    should_split = False

                # Send control data
                msg = pickle.dumps({
                    'type': MsgType.UPDATE,
                    'data': {
                        'mouse_pos': mouse_pos,
                        'keys': keys,
                    },
                })
                sock.sendto(msg, (self.host, self.port))

                # Receive game state
                msg = receive_large_data(sock)
                view.player = None
                view.model = msg
                for pl in view.model.players:
                    if pl.id == self.player_id:
                        view.player = pl
                        break   
                if view.player is None:
                    logger.debug("Player was killed!")
                    if not manual:
                        self.q_trainer.update_qtable(prev_state, action, -3000, 101)
                    return

                if not manual:
                    new_state, reward = view.model.get_player_state(self.player_id)
                    players_reward += reward

                    if action_timer == 5:
                        print("Reward:", players_reward, "Prev state:", prev_state, "Action:", action, "Direction:", direction_to_go, "epsilon:", self.q_trainer.epsilon)
                        self.q_trainer.update_qtable(prev_state, action, players_reward, new_state)
                        prev_state = new_state
                        players_reward = 0
                        
                        action_timer = 0

                    action_timer += 1

                view.redraw()
                time.sleep(1 / 40)

        except socket.timeout:
            logger.error('Server not responding')
def start(width=900, height=600, manual=False):
    print("Start - print")
    q_learner = QTraining()
    while True:

        socket.setdefaulttimeout(2)

        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('agar.io')

        gameconn = GameConnection(screen, q_learner)

        # calling connect_to_game directly with these attributes, skipping the start menu
        def get_attrs():
            return {
                # 'addr': '0.0.0.0:9999',  # port 9999 as it shows in-game
                'addr': 'localhost:9999',
                'nick': 'user'  # can set as per preference
            }

        gameconn.connect_to_game(get_attrs, manual=manual)

if __name__ == '__main__':
    start()