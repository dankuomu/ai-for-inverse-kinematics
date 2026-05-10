"""
Общие функции для бенчмарка IK: робот, цели, ошибки, штраф препятствий, время.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from robots.robot import Robot
from robots.utils import Coords, Obstacle


def standard_dh_parameters() -> List[Tuple[float, float, float]]:
    L_upper, L_forearm, L_wrist = 0.5, 0.5, 0.20
    return [
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_upper, 0.0, 0.0),
        (L_forearm, 0.0, 0.0),
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_wrist, 0.0, 0.0),
    ]


def standard_angle_limits() -> List[Tuple[float, float]]:
    return [
        (-np.pi / 2, np.pi / 4),
        (0, np.pi),
        (0, 0),
        (0, np.pi),
        (-np.pi / 2, np.pi / 2),
        (-np.pi / 2, np.pi / 2),
        (0, 2 * np.pi),
    ]


def standard_obstacles() -> List[Obstacle]:
    from robots.utils import Sphere
    r = 0.055
    return [
        Sphere(Coords([0.26, 0.06, 0.02]), r),
        Sphere(Coords([0.05, -0.30, 0.22]), r),
        Sphere(Coords([-0.08, 0.28, 0.18]), r),
    ]


def fixed_targets_from_angles(
    robot: Robot,
    angle_rows: Sequence[Sequence[float]],
) -> List[Coords]:
    return [robot.forward_kinematics(np.asarray(row, dtype=float)) for row in angle_rows]


def position_error_fk(robot: Robot, angles: np.ndarray, target: Coords) -> float:
    fk = robot.forward_kinematics(angles)
    return float(np.linalg.norm(fk.pos - target.pos))


def orientation_error_fk(robot: Robot, angles: np.ndarray, target: Coords) -> float:
    fk = robot.forward_kinematics(angles)
    R_rel = fk.rot_matrix.T @ target.rot_matrix
    tr = np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)
    return float(np.arccos(tr))


def obstacle_exp_penalty_sum(
    robot: Robot,
    angles: np.ndarray,
    obstacles: List[Obstacle],
    sigma: float = 0.01,
) -> float:
    """Сумма exp(-d/sigma) по всем отрезкам звеньев и препятствиям (как в GeneticIK)."""
    if not obstacles:
        return 0.0
    positions = robot.get_joint_positions(angles)
    points = [Coords(positions[i]) for i in range(len(positions))]
    total = 0.0
    for i in range(len(points) - 1):
        for obs in obstacles:
            d = obs.dist_to_me(points[i], points[i + 1])
            total += float(np.exp(-d / sigma))
    return total


def benchmark_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    return int(v)


def default_fixed_angle_rows() -> List[Tuple[float, ...]]:
    """Фиксированные конфигурации для воспроизводимых целей FK."""
    return [
        (0.0, 1.0, 0.0, 1.2, 0.1, -0.2, 1.0),
        (-0.4, 0.6, 0.0, 0.8, 0.0, 0.3, 2.0),
        (0.2, 1.5, 0.0, 2.0, -0.3, 0.1, 0.5),
        (-0.5, 2.2, 0.0, 1.5, 0.2, -0.1, 3.0),
        (0.1, 0.9, 0.0, 1.8, -0.2, 0.4, 1.2),
    ]
