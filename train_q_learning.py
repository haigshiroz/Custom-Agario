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
NUM_ACTIONS = 4                       # kept it at 6 due to 6 action states

ALPHA = 0.1                           # Learning rate factor taken at rought as of now
GAMMA = 0.95                          # Discount factor(it is also a rough estiamte)
EPISODES = 100                        # Total episodes to train
MAX_STEPS_PER_EPISODE = 10           # Steps per episode


class QTraining():
    def __init__(self):
        #TODO: only for testing change back 1 for trainning
        self.epsilon =  0.0  # Probablity of choosing a random action versus from QTable
        # self.epsilon = 0.01 # For demo
        
        # self.decay_rate = 0.99990408 # Decay rate for epsilon: 1 to 0.1 take 56 mins and 29 second
        # self.decay_rate = 0.9999990408 # Decay rate for epsilon: 1 to 0.1 take about 93 hours and 46 mins
        self.decay_rate = 0.999990408 # Decay rate for epsilo: 1 to 0.1 take 9 hrs and 38 mins
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
            print("acton: Random")
        # Choose best action from QTable
        else:
            action = np.argmax(q_table[prev_state])
            direction_to_go = np.argmax(q_table[prev_state][:-1])
            print("acton: Q_table")
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