import math
import random

def get_direction_and_keys(direction = None, split = None):
        keys = []

        # Value retrieved from view's mouse_to_polar. Float between [0, 1] 
        speed = 1

        if direction is None:
            direction = random.random() 
        if split is None:
             split = random.random() > 0.9

        if (split):
            keys = [32] # For space bar

        if direction == 3: #left
            return ((math.pi,speed), keys)
        elif direction == 0:#up
            return ((1/2*math.pi,speed), keys)
        elif direction == 1:#right
            return ((0,speed), keys)
        else:#down, value = 3
            return ((3/2*math.pi,speed), keys)