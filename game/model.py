import itertools
import time

from loguru import logger

from .entities import Cell


# [0, 99]
"""
Hashes based off state

most_food is the direction that has the most food
= up (0), right (1), down (2), left (3)

largest_bigger is the direction that has the largest enemy larger than the player
= up (0), right (1), down (2), left (3), none (4)

largest_smaller is the direction that has the largest enemy smaller than the player
= up (0), right (1), down (2), left (3), none (4)
"""
def custom_hash(state):
    most_food = state['most_food'] # [0, 3]
    largest_bigger = state['largest_bigger'] # [0, 4]
    largest_smaller = state['largest_smaller'] # [0, 4]
    return (most_food * (5 * 5)) + (largest_bigger * 5) + largest_smaller


class Model():
    """Class that represents game state."""
    

    class Chunk():
        def __init__(self, players=None, cells=None):
            players = list() if players is None else players
            cells = list() if cells is None else cells
            self.players = players
            self.cells = cells

    # duration of round in seconds
    ROUND_DURATION = 240

    '''
    chunk_size shows how far to render the screen for the player 
    '''
    def __init__(self, players=None, cells=None, bounds=(1000, 1000), chunk_size=300):
        players = list() if players is None else players
        cells = list() if cells is None else cells
        # means that size of world is [-world_size, world_size]
        self.bounds = bounds
        self.chunk_size = chunk_size
        self.chunks = list()
        for i in range((self.bounds[0] * 2) // chunk_size + 1):
            self.chunks.append(list())
            for j in range((self.bounds[1] * 2) // chunk_size + 1):
                self.chunks[-1].append(self.Chunk())

        for player in players:
            self.add_player(player)
        for cell in cells:
            self.add_cell(cell)

        self.round_start = time.time()

    def update_velocity(self, player, angle, speed):
        """Update passed player velocity."""
        player.update_velocity(angle, speed)

    def shoot(self, player, angle):
        """Shoots into given direction."""
        emitted_cells = player.shoot(angle)
        for cell in emitted_cells:
            self.add_cell(cell)

        if emitted_cells:
            logger.debug(f'{player} shot')
        else:
            logger.debug(f'{player} tried to shoot, but he can\'t')

    def split(self, player, angle):
        """Splits player."""
        self.remove_player(player)
        new_parts = player.split(angle)
        self.add_player(player)

        if new_parts:
            logger.debug(f'{player} splitted')
        else:
            logger.debug(f'{player} tried to split, but he can\'t')

    def get_player_state(self, player_id):
        # player = None
        # for temp_player in self.players:
        #     if temp_player.id == player_id:
        #         player = temp_player
        #         break

        # if (player == None):
        #     raise Exception("Player is not in game")

        for player in self.players:
            if player.id == player_id:
                # get chuncks around player
                chunks = self.__nearby_chunks_custom(player.center())

                # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
                # bottom left, left, top left, bottom, middle, top, bottom right, right, top right

                # Encapsulate all 3 cells in a direction + middle
                top_idxs = [2, 4, 5, 8] # 0
                right_idxs = [4, 6, 7, 8] # 1
                bot_idxs = [0, 3, 4, 6] # 2
                left_idxs = [0,1,2, 4] # 3
                # none # 4
                directions = [top_idxs, right_idxs, bot_idxs, left_idxs]

                max_food_amt = 0
                largest_bigger_amt = 0
                largest_smaller_amt = 0
                state = {"most_food": 0,
                            "largest_bigger": 4,
                            "largest_smaller": 4}
                # For each direction
                for direction_idx in range(len(directions)):
                    direction = directions[direction_idx]
                    num_cells_in_direction = -1
                    # For each chunk in that direction
                    for chunk_idx in direction:
                        temp_chunk = chunks[chunk_idx]

                        # Go to next chunk if isn't a valid chunk (ex: wall)
                        if (temp_chunk == None):
                            continue

                        # Add amount of food in that chunk
                        num_cells_in_direction += len(temp_chunk.cells)
                        # Check if there is a larger bigger player and larger smaller player
                        for player2 in temp_chunk.players:
                            if player2.id != player.id:
                                difference_in_mass = player2.score() - player.score()
                                if (difference_in_mass < 0):
                                    # Other player is smaller than this player 
                                    if (difference_in_mass > largest_smaller_amt):
                                        state["largest_smaller"] = direction_idx
                                        largest_smaller_amt = difference_in_mass
                                else:
                                    # Other player is bigger than this player
                                    if (difference_in_mass > largest_bigger_amt):
                                        state["largest_bigger"] = direction_idx
                                        largest_bigger_amt = difference_in_mass
                        if (num_cells_in_direction > max_food_amt):
                            state["most_food"] = direction_idx
                            max_food_amt = num_cells_in_direction

                state_hash = custom_hash(state)
                ret = (state_hash, player.reward)
                return ret
    
        raise Exception("Player is not in game")


    def update(self):
        """Updates game state."""
        if time.time() - self.round_start >= self.ROUND_DURATION:
            logger.debug('New round was started.')
            self.__reset_players()
            self.round_start = time.time()

        # update cells
        for cell in self.cells:
            self.remove_cell(cell)
            cell.move()
            self.bound_cell(cell)
            self.add_cell(cell)

        # update players
        observable_players = self.players
        for player in observable_players:
            self.remove_player(player)
            player.move()
            self.bound_player(player)
            self.add_player(player)

            player.reset_reward()

            # get chuncks around player
            chunks = self.__nearby_chunks(player.center())
            # get objects that stored in chunks
            players = list()
            cells = list()
            for chunk in chunks:
                players.extend(chunk.players)
                cells.extend(chunk.cells)


            # print(len(cells), ", ", len(players), ", ", len(chunks))
            
            # check is player killed some cells
            for cell in cells:
                killed_cell = player.attempt_murder(cell)
                if killed_cell:
                    logger.debug(f'{player} ate {killed_cell}')
                    self.remove_cell(killed_cell)
                    player.ate_cell_reward()
                    # self.cells.remove(killed_cell)
            
            # check is player killed other players or their parts
            for another_player in players:
                if player == another_player:
                    continue
                killed_cell = player.attempt_murder(another_player)
                if killed_cell:
                    if len(another_player.parts) == 1:
                        logger.debug(f'{player} ate {another_player}')
                        self.remove_player(another_player)
                        observable_players.remove(another_player)
                        another_player.remove_part(killed_cell)
                        player.ate_player_reward()
                        another_player.death_reward()
                    else:
                        logger.debug(f'{player} ate {another_player} part {killed_cell}')

            player.idle_reward()

    def spawn_cells(self, amount):
        """Spawn passed amount of cells on the field."""
        for _ in range(amount):
            self.add_cell(Cell.make_random(self.bounds))

    def bound_cell(self, cell):
        cell.pos[0] = self.bounds[0] if cell.pos[0] > self.bounds[0] else cell.pos[0]
        cell.pos[0] = -self.bounds[0] if cell.pos[0] < -self.bounds[0] else cell.pos[0]

        cell.pos[1] = self.bounds[1] if cell.pos[1] > self.bounds[1] else cell.pos[1]
        cell.pos[1] = -self.bounds[1] if cell.pos[1] < -self.bounds[1] else cell.pos[1]

    def bound_player(self, player):
        for cell in player.parts:
            self.bound_cell(cell)

    def add_player(self, player):
        self.__pos_to_chunk(player.center()).players.append(player)

    def add_cell(self, cell):
        self.__pos_to_chunk(cell.pos).cells.append(cell)

    def remove_player(self, player):
        self.__pos_to_chunk(player.center()).players.remove(player)

    def remove_cell(self, cell):
        self.__pos_to_chunk(cell.pos).cells.remove(cell)

    def copy_for_client(self, pos):
        chunks = self.__nearby_chunks(pos)
        players = list()
        cells = list()
        for chunk in chunks:
            players.extend(chunk.players)
            cells.extend(chunk.cells)

        model = Model(players, cells, self.bounds, self.chunk_size)
        model.round_start = self.round_start
        return model

    def __reset_players(self):
        for player in self.players:
            player.reset()

    def __pos_to_chunk(self, pos):
        chunk_pos = self.__chunk_pos(pos)
        return self.chunks[chunk_pos[0]][chunk_pos[1]]

    def __chunk_pos(self, pos):
        return [
            int((pos[0] + self.bounds[0]) // self.chunk_size),
            int((pos[1] + self.bounds[1]) // self.chunk_size)]

    def __nearby_chunks(self, pos):
        chunks = list()
        chunk_pos = self.__chunk_pos(pos)

        def is_valid_chunk_pos(pos):
            if pos[0] >= 0 and pos[0] < len(self.chunks) and \
                    pos[1] >= 0 and pos[1] < len(self.chunks[0]):
                return True
            return False

        diffs = []
        for diff in itertools.product([-1, 0, 1], repeat=2):
            diffs.append((diff[0], diff[1]))
            pos = [chunk_pos[0] + diff[0], chunk_pos[1] + diff[1]]
            if is_valid_chunk_pos(pos):
                chunks.append(self.chunks[pos[0]][pos[1]])
        
        return chunks
    
    def __nearby_chunks_custom(self, pos):
        chunks = list()
        chunk_pos = self.__chunk_pos(pos)
        # print(chunk_pos)

        def is_valid_chunk_pos(pos):
            if pos[0] >= 0 and pos[0] < len(self.chunks) and \
                    pos[1] >= 0 and pos[1] < len(self.chunks[0]):
                return True
            return False

        diffs = []
        for diff in itertools.product([-1, 0, 1], repeat=2):
            diffs.append((diff[0], diff[1]))
            pos = [chunk_pos[0] + diff[0], chunk_pos[1] + diff[1]]
            if is_valid_chunk_pos(pos):
                chunks.append(self.chunks[pos[0]][pos[1]])
            else:
                chunks.append(None)
        
        return chunks

    @property
    def cells(self):
        cells = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                cells.extend(chunk.cells)
        return cells

    @property
    def players(self):
        players = list()
        for chunks_line in self.chunks:
            for chunk in chunks_line:
                players.extend(chunk.players)
        return players
    