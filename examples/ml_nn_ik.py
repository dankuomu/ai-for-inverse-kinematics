"""
Пример: нейросетевой IK (MLP) — смесь сетки по суставам и случайных сэмплов.

Требуется: torch, scikit-learn.

Запуск:
    python examples/ml_nn_ik.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from examples.common import ANGLE_LIMITS, DEFAULT_TARGET, make_robot

from control.IK.ml_dataset import default_angles_from_limits
from control.IK.nn import ForwardNeuralIK


def main():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Установи torch: pip install torch")
        raise SystemExit(1)

    robot = make_robot()
    n_j = len(ANGLE_LIMITS)
    # вход: 12 (pos + R), выход: n_j углов
    layers = [12, 128, 128, n_j]
    robot.set_inverse(
        ForwardNeuralIK,
        layers=layers,
        angle_limits=ANGLE_LIMITS,
        epochs=40,
        batch_size=256,
        lr=1e-3,
        dataset_size=8000,
        model_path="checkpoints/nn_ik.pt",
    )

    # 1) детерминированная сетка (мало точек)
    joint_grid = {
        0: np.linspace(-0.6, 0.4, num=4),
        1: np.linspace(0.5, 2.0, num=4),
    }
    robot.ik_solver.generate_dataset_grid(
        ANGLE_LIMITS,
        joint_value_grid=joint_grid,
        default_angles=default_angles_from_limits(ANGLE_LIMITS),
        max_combinations=20_000,
    )
    Xg, yg = robot.ik_solver.X.copy(), robot.ik_solver.y.copy()

    # 2) случайные сэмплы и объединение
    robot.ik_solver.generate_dataset_random(ANGLE_LIMITS, n_samples=6000)
    robot.ik_solver.X = np.vstack([Xg, robot.ik_solver.X])
    robot.ik_solver.y = np.vstack([yg, robot.ik_solver.y])

    robot.ik_solver.train()

    pred, metrics = robot.ik_solver.solve(DEFAULT_TARGET)
    robot.visualize(pred, target=DEFAULT_TARGET)

    print("ForwardNeuralIK:")
    print(f"  dataset: {robot.ik_solver.X.shape[0]} samples")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f} rad")


if __name__ == "__main__":
    main()
