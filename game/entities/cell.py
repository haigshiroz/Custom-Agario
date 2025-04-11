import random
from operator import add, sub

from .. import gameutils as gu
from . import interfaces
from .circle import Circle

import math

class Cell(Circle, interfaces.Victim):
    """Represents cell(food) state."""

    BORDER_WIDTH=0
    FRICTION = 0.1
    MAX_SPEED = 5
    SIZES = (5, 7, 10)
    SIZES_CUM = (70, 20, 10)
    SPEED_DECAY = 0.03
    MIN_SPEED = 1.0

    def __init__(self, pos, radius, color, angle=0, speed=0):
        super().__init__(pos, radius)
        # cell color [r, g, b]
        self.color = color
        # angle of speed in rad
        self.angle = angle
        # speed coeff from 0.0 to 1.0
        self.speed = max(speed, self.MIN_SPEED)

    def get_effective_speed(self):
        """Calc the speed based on the mass """
        return self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * math.exp(-self.SPEED_DECAY * self.radius)
    
    def move(self):
        self.speed = max(0, self.speed - self.FRICTION)
        
        move_distance = self.speed * self.get_effective_speed()
        diff_xy = gu.polar_to_cartesian(self.angle, move_distance)
        self.pos = list(map(add, self.pos, diff_xy))
    
    def update_velocity(self, angle, speed):
        self.angle = angle % (2 * math.pi)
        self.speed = max(speed, self.MIN_SPEED)

    def try_to_kill_by(self, killer):
        """Check is killer cell could eat current cell."""
        if 2*self.area() <= killer.area() and \
                self.distance_to(killer) <= killer.radius - self.radius:
            return self
        return None

    @classmethod
    def make_random(cls, bounds):
        """Creates random cell."""
        pos = gu.random_pos(bounds)
        radius = random.choices(cls.SIZES, cls.SIZES_CUM)[0]
        color = gu.random_safe_color()
        return cls(pos, radius, color)

    def __repr__(self):
        return '<{} pos={} radius={}>'.format(
            self.__class__.__name__,
            list(map(int, self.pos)),
            int(self.radius))