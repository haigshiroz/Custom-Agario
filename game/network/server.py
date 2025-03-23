
import socketserver
import pickle
import math

from loguru import logger
import pygame

from .msgtype import MsgType
from .. import Model
from ..entities import Player

# UDP Max Packet Size
MAX_UDP_SIZE = 8000  

bounds = [1000, 1000]
cell_num = 150
model = Model(list(), bounds=bounds)
model.spawn_cells(cell_num)

clients = dict()
p = Player.make_random("Jetraid", bounds)

def send_large_data(socket, data, client_address):
    """Splits and sends large data over multiple UDP packets"""
    raw_data = pickle.dumps(data)
    total_size = len(raw_data)
    num_chunks = math.ceil(total_size / MAX_UDP_SIZE)  

    for i in range(num_chunks):
        start = i * MAX_UDP_SIZE
        end = start + MAX_UDP_SIZE
        chunk = raw_data[start:end]

        packet = pickle.dumps((i, num_chunks, chunk))
        socket.sendto(packet, client_address)

class UDPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        msg = pickle.loads(self.request[0])
        msgtype = msg['type']
        data = msg['data']

        global clients, model, bounds

        if msgtype == MsgType.CONNECT:
            nick = data
            logger.debug('Received {!r} from {}'.format(nick, self.client_address))

            new_player = Player.make_random(nick, bounds)

            # Send player ID
            data = pickle.dumps(new_player.id)
            logger.debug('Sending {!r} to {}'.format(data, self.client_address))
            socket = self.request[1]
            socket.sendto(data, self.client_address)

            clients[self.client_address] = new_player
            model.add_player(new_player)

        elif msgtype == MsgType.UPDATE:
            mouse_pos = data['mouse_pos']
            keys = data['keys']
            player = clients.get(self.client_address)

            if not player:
                logger.error("Player not found for address {}".format(self.client_address))
                return

            for key in keys:
                if key == pygame.K_w:
                    model.shoot(player, mouse_pos[0])
                elif key == pygame.K_SPACE:
                    model.split(player, mouse_pos[0])

            model.update_velocity(player, *mouse_pos)
            model.update()

            # Send the game state using fragmentation
            send_large_data(self.request[1], model.copy_for_client(player.center()), self.client_address)

def start(host='localhost', port=9999):
    with socketserver.UDPServer((host, port), UDPHandler) as server:
        logger.info('Server started at {}:{}'.format(host, port))
        server.serve_forever()

if __name__ == '__main__':
    start()
