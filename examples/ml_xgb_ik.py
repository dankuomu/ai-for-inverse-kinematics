"""
Пример: XGBoost IK — датасет по сетке суставов + обучение.

Требуется: xgboost, scikit-learn, joblib (для сохранения модели).

Запуск:
    python examples/ml_xgb_ik.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PLOTS_DIR = _ROOT / "plots"

import numpy as np

from examples.common import ANGLE_LIMITS, DEFAULT_OBSTACLES, DEFAULT_TARGET, make_robot

from control.IK.decision_trees import XGBoostIK
from control.IK.ml_dataset import default_angles_from_limits


def main():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("Установи xgboost: pip install xgboost")
        raise SystemExit(1)

    robot = make_robot()
    robot.set_inverse(
        XGBoostIK,
        angle_limits=ANGLE_LIMITS,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        dataset_size=8000,
        model_path="checkpoints/xgb_ik.joblib",
    )

    joint_grid = {
        0: np.linspace(-0.7, 0.4, num=5),
        3: np.linspace(0.5, 2.5, num=5),
    }
    robot.ik_solver.generate_dataset_grid(
        ANGLE_LIMITS,
        joint_value_grid=joint_grid,
        default_angles=default_angles_from_limits(ANGLE_LIMITS),
        max_combinations=30_000,
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_frames = PLOTS_DIR / "xgb_tree_frames"
    xgb_video = PLOTS_DIR / "xgb_approach.mp4"
    robot.ik_solver.train(
        plot_learning_curve=False,
        progression_target=DEFAULT_TARGET,
        progression_obstacles=DEFAULT_OBSTACLES,
        progression_frames_dir=str(xgb_frames),
        progression_snapshots=36,
        progression_video_path=str(xgb_video),
    )

    pred, metrics = robot.ik_solver.solve(DEFAULT_TARGET)
    robot.visualize(pred, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES, show=False)

    print("XGBoostIK:")
    print(f"  кадры по числу деревьев: {xgb_frames}")
    print(f"  видео подхода к цели (MP4): {xgb_video}")
    print(f"  dataset: {robot.ik_solver.X.shape[0]} samples")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f} rad")


if __name__ == "__main__":
    main()
