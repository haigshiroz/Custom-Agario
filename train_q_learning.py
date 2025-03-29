# import pandas as pd
import numpy as np
import pickle
import random
from game.helper.move_player import get_direction_and_keys
from filelock import FileLock



# --- CONFIGURATION ---
EXCEL_PATH = "state_scores.xlsx"      # The excel file with rough data
Q_TABLE_PATH = "Q_table.pickle"       # Output file for the Q-table
Q_NUM_UPDATES_PATH = "Q_num_updates.pickle"       # Output file for the Q-table's number of updates
NUM_ACTIONS = 5                       # kept it at 6 due to 6 action states

ALPHA = 0.1                           # Learning rate factor taken at rought as of now
GAMMA = 0.95                          # Discount factor(it is also a rough estiamte)
EPISODES = 100                        # Total episodes to train
MAX_STEPS_PER_EPISODE = 10           # Steps per episode


class QTraining():
    def __init__(self):
        self.epsilon = 1 # Probablity of choosing a random action versus from QTable
        self.decay_rate = 0.99 # Decay rate for epsilon

    '''
    Purpose: 
    Returns what action the agent should do. 
    Checks based off current epsilon value and returns a random action or the best action given the state

    Arguments:
    state - a hash of the agent's current state

    Returns:
    A tuple
    - Item 1: action (1 = up, 2 = right, 3 = down, 4 = left, 5 = shoot)
    - Item 2: direction to move (1 = up, 2 = right, 3 = down, 4 = left)
    '''
    def get_action_and_direction(self, prev_state: int) -> tuple[int, int]:
        lock_table = FileLock(Q_TABLE_PATH + ".lock")
        # Get the most up-to-date q_table
        with lock_table:
            while True:
                try:
                    q_table = np.load(Q_TABLE_PATH, allow_pickle=True)
                    break
                except EOFError as e:
                    raise Exception()
                    print("EOF Error - try again")
                

        # Choose a random action
        if random.random() < self.epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)

            # If the action we randomly chose was to split, get another random value for the direction to split in
            if (action == 4):
                direction_to_go = random.randint(0, NUM_ACTIONS - 2)
            else:
                direction_to_go = action
        # Choose best action from QTable
        else:
            action = np.argmax(q_table[prev_state])
            direction_to_go = np.argmax(q_table[prev_state][:-1])

        return (action, direction_to_go)

    def update_qtable(self, prev_state: int, action: int, reward: int,  new_state: int) -> None:
        lock_table = FileLock(Q_TABLE_PATH + ".lock")
        lock_updates = FileLock(Q_NUM_UPDATES_PATH + ".lock")

        # Load the tables
        with lock_table:
            while True:
                try:
                    q_table = np.load(Q_TABLE_PATH, allow_pickle=True)
                    break
                except EOFError as e:
                    raise Exception()
                    print("EOF Error - try again")
        with lock_updates:
            while True:
                try:
                    q_num_updates = np.load(Q_NUM_UPDATES_PATH, allow_pickle=True)
                    break
                except EOFError as e:
                    raise Exception()
                    print("EOF Error - try again")
            

        # Updating the given Q-Learning

        # Calculate eta
        temp = q_num_updates[prev_state]
        num_updpates = temp[action]
        eta = 1.0 / (1.0 + num_updpates)

        # Calculate new value
        old_value = q_table[prev_state][action]
        next_max = np.max(q_table[new_state])
        new_value = ((1 - eta) * old_value) + (eta * (reward + GAMMA * next_max))

        # Update tables
        q_table[prev_state][action] = new_value
        q_num_updates[prev_state][action] += 1

        # Save new tables
        self._save_qtable(q_table, q_num_updates)

    def end_of_episode(self):
        # End of episode happens when a player died or a round resets
        self.epsilon *= self.decay_rate

    def _save_qtable(self, q_table, q_num_updates):
        lock_table = FileLock(Q_TABLE_PATH  + ".lock")
        lock_updates = FileLock(Q_NUM_UPDATES_PATH + ".lock")

        with lock_table:
            with open(Q_TABLE_PATH, "wb") as f:
                pickle.dump(q_table, f)

        with lock_updates:
            with open(Q_NUM_UPDATES_PATH, "wb") as f:
                pickle.dump(q_num_updates, f)



# # Loading our data from excel
# df = pd.read_excel(EXCEL_PATH)
# states = df['State'].tolist()
# scores = df['Score'].tolist()

# max_state = max(states)
# q_table = np.zeros((max_state + 1, NUM_ACTIONS))

# # Look up for rewards
# reward_lookup = dict(zip(states, scores))

# # Training data
# for episode in range(EPISODES):
#     state = random.choice(states)

#     for step in range(MAX_STEPS_PER_EPISODE):
#         epsilon = 0.1
#         if random.random() < epsilon:
#             action = random.randint(0, NUM_ACTIONS - 1)
#         else:
#             action = np.argmax(q_table[state])

#         reward = reward_lookup.get(state, 0)

#         if random.random() < 0.5:
#             next_state = min(max_state, state + random.choice([-1, 1]))
#         else:
#             next_state = state

#         # Updating the given Q-Learning
#         old_value = q_table[state][action]
#         next_max = np.max(q_table[next_state])
#         new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
#         q_table[state][action] = new_value

#         state = next_state

# print(" Episodic Q-learning training complete.")

# # Saving trained table
# with open(Q_TABLE_PATH, "wb") as f:
#     pickle.dump(q_table, f)

# print(f" Trained Q-table saved to: {"C:\Users\prana\OneDrive\Desktop\FAI_New\Custom-Agario\q_table.py"}" )


'''
  File "C:\0-Projects\Courses\CS5100\Custom-Agario\game\network\client.py", line 82, in connect_to_game
    action, direction_to_go = self.q_trainer.get_action_and_direction(prev_state)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\0-Projects\Courses\CS5100\Custom-Agario\train_q_learning.py", line 40, in get_action_and_direction
    q_table = np.load(Q_TABLE_PATH, allow_pickle=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\0-Projects\Courses\CS5100\Custom-Agario\venv\Lib\site-packages\numpy\lib\_npyio_impl.py", line 460, in load
    raise EOFError("No data left in file")
'''
