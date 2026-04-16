"""
Пример: DDPG IK (обучение online при solve), визуализация, опционально чекпоинт.

Запуск из корня репозитория:
    python examples/ddpg_ik.py
"""
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from examples.common import ANGLE_LIMITS, DEFAULT_OBSTACLES, DEFAULT_TARGET, make_robot

from control.IK.ddpg import DDPGIK


def main():
    robot = make_robot()
    robot.set_inverse(
        DDPGIK,
        bounds=ANGLE_LIMITS,
        obstacles=DEFAULT_OBSTACLES,
        episodes=300,
        max_steps=120,
        hidden_dims=[256, 256],
        actor_lr=1e-4,
        critic_lr=1e-3,
        batch_size=128,
        buffer_size=100_000,
        error_weight_mode="exp",
        exp_weight_alpha=5.0,
        max_orientation_weight=0.95,
        position_tolerance=1e-2,
        orientation_tolerance=1e-2,
        save_episode_images=False,
        weights_path="checkpoints/ddpg_ik.pt",
        save_weights_after_run=True,
        load_weights_if_exist=True,
    )

    angles, metrics = robot.solve(DEFAULT_TARGET)
    robot.visualize(angles, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)

    print("МЕТРИКИ DDPG:")
    print(f"  total_time: {metrics['total_time']:.4f} s")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f}")

    angles_refined, m2 = robot.op_solve(angles, DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    robot.visualize(angles_refined, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    print("После op_solve:")
    print(f"  position_error: {m2['position_error']:.6f}")
    print(f"  orientation_error: {m2['orientation_error']:.6f}")


if __name__ == "__main__":
    main()
