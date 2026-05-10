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

# Длины плеча / предплечья / кисти как в main; по a — короткие «суставные» сегменты
# (чуть-чуть, чтобы цилиндры в 3D не слипались в одну линию), заметно меньше старых 0.1 м.
L_UPPER = 0.5
L_FOREARM = 0.5
L_WRIST = 0.20
JOINT_LINK_A = 0.2

DH_PARAMETERS = [
    (JOINT_LINK_A, np.pi / 2, 0.0),
    (JOINT_LINK_A, -np.pi / 2, 0.0),
    (L_UPPER, 0.0, 0.0),
    (L_FOREARM, 0.0, 0.0),
    (JOINT_LINK_A, np.pi / 2, 0.0),
    (JOINT_LINK_A, -np.pi / 2, 0.0),
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

# Сферы поменьше и чуть разнесены (меньше перекрытий и «шумной» зоны вокруг цели).
OBSTACLE_RADIUS = 0.1
DEFAULT_OBSTACLES = [
    Sphere(Coords([0.26, 0.06, 0.02]), OBSTACLE_RADIUS),
]

# Результат tune DDPG (можно распаковать в set_inverse: **DDPG_TUNED_KWARGS)
DDPG_TUNED_KWARGS = {
    "actor_lr": 1e-4,
    "critic_lr": 5e-4,
    "action_scale": 0.06,
    "noise_sigma": 0.15,
    "exp_weight_alpha": 4.0,
    "tau": 0.005,
}


def make_robot() -> Robot:
    return Robot(DH_PARAMETERS)
