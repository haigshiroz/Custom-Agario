import socket
import pickle
import pygame
import time

from loguru import logger
from .menu import MyMenu
from .msgtype import MsgType
from .. import View

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
    def __init__(self, screen):
        self.screen = screen
        self.player_id = None
        self.is_in_lobby = False
        self.host = None
        self.port = None
        self.addr_string = None

    def connect_to_game(self, get_attrs):
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

            # Receive player info
            data = sock.recv(4096)
            self.player_id = pickle.loads(data)
            logger.debug('Received {!r} from {}'.format(self.player_id, self.addr_string))

            # Create game view
            view = View(self.screen, None, None)
            while True:
                keys = []
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        keys.append(event.key)

                mouse_pos = view.mouse_pos_to_polar()
                msg = pickle.dumps({'type': MsgType.UPDATE, 'data': {'mouse_pos': mouse_pos, 'keys': keys}})
                sock.sendto(msg, (self.host, self.port))

                # Receive game state using fragmentation
                msg = receive_large_data(sock)

                view.player = None
                view.model = msg
                for pl in view.model.players:
                    if pl.id == self.player_id:
                        view.player = pl
                        break

                if view.player is None:
                    logger.debug("Player was killed!")
                    return

                view.redraw()
                time.sleep(1/40)
        except socket.timeout:
            logger.error('Server not responding')

def start(width=900, height=600):
    socket.setdefaulttimeout(2)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('agar.io')

    gameconn = GameConnection(screen)

    # calling connect_to_game directly with these attributes, skipping the start menu
    def get_attrs():
        return {
            'addr': '0.0.0.0:9999',  # port 9999 as it shows in-game
            'nick': 'user'  # can set as per preference
        }

    gameconn.connect_to_game(get_attrs)

if __name__ == '__main__':
    start()