import math
import random


def get_random_dir(mouse_pos, value = None):
        angle,speed = mouse_pos
        if value is None:
            value = random.random() 

        if 0 <= value < 0.25: #left
            return math.pi,speed
        elif 0.25 <= value < 0.5:#right
            return 0,speed
        elif 0.5 <= value < 0.75:#up
            return 1/2*math.pi,speed
        else:#down
            return 3/2*math.pi,speed