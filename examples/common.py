"""
Общая кинематическая схема и цель для примеров (7-DOF DH как в main.py).
Запуск примеров из корня репозитория: ``python examples/genetic_ik.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Корень репозитория в sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robots.robot import Robot
from robots.utils import Coords, Sphere

L_UPPER = 0.5
L_FOREARM = 0.5
L_WRIST = 0.20

DH_PARAMETERS = [
    (0.0, np.pi / 2, 0.0),
    (0.0, -np.pi / 2, 0.0),
    (L_UPPER, 0.0, 0.0),
    (L_FOREARM, 0.0, 0.0),
    (0.0, np.pi / 2, 0.0),
    (0.0, -np.pi / 2, 0.0),
    (L_WRIST, 0.0, 0.0),
]

ANGLE_LIMITS = [
    (-np.pi / 2, np.pi / 4),
    (0, np.pi),
    (0, 0),
    (0, np.pi),
    (-np.pi / 2, np.pi / 2),
    (-np.pi / 2, np.pi / 2),
    (0, 2 * np.pi),
]

DEFAULT_TARGET = Coords(
    np.array([-0.3, -0.10, 0.4]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
)

DEFAULT_OBSTACLES = [
    Sphere(Coords([0.2, 0.0, 0.0]), 0.1),
    Sphere(Coords([0.0, -0.2, 0.2]), 0.1),
    Sphere(Coords([0.0, 0.2, 0.2]), 0.1),
]


def make_robot() -> Robot:
    return Robot(DH_PARAMETERS)
