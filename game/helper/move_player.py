import math
import random
import numpy as np
import os


def get_direction_and_keys(mouse_pos, direction = None, shoot = None):
         # Get the absolute path of the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Move up two levels to reach "Custom-Agario"
        project_root = os.path.dirname(os.path.dirname(script_dir))

        # Construct the absolute path to Q_table.pickle
        q_table_path = os.path.join(project_root, "Q_table.pickle")

        # Load the Q-table
        Q_table = np.load(q_table_path, allow_pickle=True)

        keys = []

        angle,speed = mouse_pos
        if direction is None:
            direction = random.random() 
        if shoot is None:
             shoot = random.random() > 0.5 

        if (shoot):
            keys = [32] # For space bar

        if direction == 1: #left
            return ((math.pi,speed), keys)
        elif direction == 2:#up
            return ((1/2*math.pi,speed), keys)
        elif direction == 3:#right
            return ((0,speed), keys)
        else:#down, value = 4
            return ((3/2*math.pi,speed), keys)

        # if 0 <= value < 0.25: #left
        #     return math.pi,speed
        # elif 0.25 <= value < 0.5:#right
        #     return 0,speed
        # elif 0.5 <= value < 0.75:#up
        #     return 1/2*math.pi,speed
        # else:#down
        #     return 3/2*math.pi,speed