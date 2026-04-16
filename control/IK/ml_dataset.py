"""
Общие утилиты для ML-IK: признаки позы, метрики и генерация конфигураций суставов (random / grid).
"""
from __future__ import annotations

import logging
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from robots.utils import Coords

logger = logging.getLogger(__name__)


def pose_feature_vector(coords: Coords) -> np.ndarray:
    """Вектор признаков: позиция (3) + flatten(R) (9) → длина 12."""
    return np.concatenate([coords.pos, coords.rot_matrix.reshape(-1)]).astype(np.float32)


def orientation_error_radians(achieved: Coords, target: Coords) -> float:
    """Угол ориентации между достигнутой и целевой системой (как в GeneticIK)."""
    R_rel = achieved.rot_matrix.T @ target.rot_matrix
    trace_val = np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)
    return float(np.arccos(trace_val))


def position_error(achieved: Coords, target: Coords) -> float:
    return float(np.linalg.norm(achieved.pos - target.pos))


def default_angles_from_limits(angle_limits: List[Tuple[float, float]]) -> np.ndarray:
    """Середина каждого допустимого интервала (для сустава lo==hi — это значение)."""
    return np.array([0.5 * (lo + hi) for lo, hi in angle_limits], dtype=np.float64)


def sample_random_joint_configs(
    angle_limits: List[Tuple[float, float]],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Случайные углы в пределах bounds, форма (n_samples, n_joints)."""
    rng = rng or np.random.default_rng()
    n_j = len(angle_limits)
    out = np.empty((n_samples, n_j), dtype=np.float64)
    for i in range(n_samples):
        out[i] = [rng.uniform(lo, hi) for lo, hi in angle_limits]
    return out


def joint_grid_to_configurations(
    n_joints: int,
    angle_limits: List[Tuple[float, float]],
    joint_value_grid: Dict[int, Sequence[float]],
    default_angles: Optional[np.ndarray] = None,
    max_combinations: int = 50_000,
) -> np.ndarray:
    """
    Декартово произведение значений по указанным суставам.

    :param joint_value_grid: ``{индекс_сустава: [v1, v2, ...]}``
    :param default_angles: базовая конфигурация для суставов вне сетки; иначе середины ``angle_limits``
    :param max_combinations: защита от взрыва размера
    """
    if not joint_value_grid:
        raise ValueError("joint_value_grid не должен быть пустым")
    for j in joint_value_grid:
        if j < 0 or j >= n_joints:
            raise ValueError(f"Индекс сустава вне диапазона: {j} (n_joints={n_joints})")

    keys = sorted(joint_value_grid.keys())
    value_lists = [list(joint_value_grid[k]) for k in keys]
    n_comb = 1
    for lst in value_lists:
        n_comb *= len(lst)
    if n_comb > max_combinations:
        raise ValueError(
            f"Слишком много комбинаций сетки: {n_comb} > max_combinations={max_combinations}. "
            "Уменьши сетку или увеличь max_combinations."
        )

    base = default_angles.astype(np.float64).copy() if default_angles is not None else default_angles_from_limits(angle_limits)
    if base.shape != (n_joints,):
        raise ValueError(f"default_angles должен иметь длину {n_joints}")

    rows = []
    for combo in product(*value_lists):
        ang = base.copy()
        for idx, val in zip(keys, combo):
            lo, hi = angle_limits[idx]
            v = float(val)
            if v < lo - 1e-9 or v > hi + 1e-9:
                logger.warning("Значение сустава %s=%s вне bounds [%s, %s], клип", idx, v, lo, hi)
            ang[idx] = np.clip(v, lo, hi)
        rows.append(ang)
    return np.array(rows, dtype=np.float64)


def build_xy_from_robot(robot, angles_matrix: np.ndarray, dtype_x=np.float32, dtype_y=np.float32):
    """FK по каждой строке angles_matrix → X (признаки), y (углы)."""
    X_list, y_list = [], []
    for row in angles_matrix:
        coords = robot.forward_kinematics(row)
        X_list.append(pose_feature_vector(coords))
        y_list.append(np.asarray(row, dtype=dtype_y))
    X = np.stack(X_list, axis=0).astype(dtype_x)
    y = np.stack(y_list, axis=0).astype(dtype_y)
    return X, y
