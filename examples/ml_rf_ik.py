"""
Пример: Random Forest IK — датасет по сетке выбранных суставов + обучение + solve.

Пользователь задаёт ``joint_value_grid``: {индекс_сустава: [значения...]}.
Остальные суставы берутся из середины ``angle_limits`` (или из ``default_angles``).

Запуск:
    python examples/ml_rf_ik.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from examples.common import ANGLE_LIMITS, DEFAULT_TARGET, make_robot

from control.IK.decision_trees import RandomForestIK
from control.IK.ml_dataset import default_angles_from_limits


def main():
    robot = make_robot()
    robot.set_inverse(
        RandomForestIK,
        angle_limits=ANGLE_LIMITS,
        n_estimators=150,
        max_depth=20,
        dataset_size=5000,
    )

    # Сетка: перебираем сустав 0 и 1; остальные — базовая конфигурация (середины интервалов)
    joint_grid = {
        0: np.linspace(-0.8, 0.5, num=6),
        1: np.linspace(0.3, 2.5, num=5),
    }
    default_angles = default_angles_from_limits(ANGLE_LIMITS)

    robot.ik_solver.generate_dataset_grid(
        ANGLE_LIMITS,
        joint_value_grid=joint_grid,
        default_angles=default_angles,
        max_combinations=50_000,
    )
    robot.ik_solver.train(plot_learning_curve=False)

    pred, metrics = robot.ik_solver.solve(DEFAULT_TARGET)
    robot.visualize(pred, target=DEFAULT_TARGET)

    print("RandomForestIK (обучение на сетке суставов):")
    print(f"  dataset: {robot.ik_solver.X.shape[0]} samples")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f} rad")


if __name__ == "__main__":
    main()
