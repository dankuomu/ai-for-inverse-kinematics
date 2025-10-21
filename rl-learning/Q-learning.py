import numpy as np
from enum import Enum as PyEnum
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:\t %(name)s - %(message)s",
)
logger = logging.getLogger("RL")

class Direction(str, PyEnum):
    UP = "UP"
    DOWN = "DOWN"
    RIGHT = "RIGHT"
    LEFT = "LEFT"

class Environment:
    def __init__(self):
        self.state = (1, 1)
        self.map =\
        [[-100, -100, -100, -100, -100, -100],
         [-100,    0,    0,   -1,  +10, -100],
         [-100,    -1,   0,    0,    0, -100],
         [-100,    -1,   0,   -1,    0, -100],
         [-100,    -1,   0,   +5,    0, -100],
         [-100, -100, -100, -100, -100, -100]]
        self.is_playing = True

    def _move_up(self):
        try:
            self.state = (self.state[0], self.state[1] + 1)
            logger.info(f"Moved  up   to {self.state}")
        except KeyError:
            logger.error(f"Invalid direction")
            self.is_playing = False

    def _move_down(self):
        try:
            self.state = (self.state[0], self.state[1] - 1)
            logger.info(f"Moved down  to {self.state}")
        except KeyError:
            logger.error(f"Invalid direction")
            self.is_playing = False

    def _move_right(self):
        try:
            self.state = (self.state[0] + 1, self.state[1])
            logger.info(f"Moved right to {self.state}")
        except KeyError:
            logger.error(f"Invalid direction")
            self.is_playing = False

    def _move_left(self):
        try:
            self.state = (self.state[0] - 1, self.state[1])
            logger.info(f"Moved left  to {self.state}")
        except KeyError:
            logger.error(f"Invalid direction")
            self.is_playing = False

    def _get_reward(self):
        return self.map[self.state[0]][self.state[1]]

    def action(self, direction):
        if direction == Direction.UP:
            self._move_up()
        elif direction == Direction.DOWN:
            self._move_down()
        elif direction == Direction.RIGHT:
            self._move_right()
        elif direction == Direction.LEFT:
            self._move_left()
        return self.is_playing, self._get_reward()

class Agent:
    def __init__(self):
        self.disc_factor = 0.95
        self.epoch_reward = 0.0
        