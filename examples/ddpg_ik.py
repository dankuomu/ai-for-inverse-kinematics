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

from examples.common import (
    ANGLE_LIMITS,
    DEFAULT_OBSTACLES,
    DEFAULT_TARGET,
    DDPG_TUNED_KWARGS,
    make_robot,
)

from control.IK.ddpg import DDPGIK


def main():
    plots_dir = _ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    episode_frames = plots_dir / "ddpg_episode_frames"
    curve_png = plots_dir / "ddpg_training_curve.png"
    approach_video = plots_dir / "ddpg_approach.mp4"

    robot = make_robot()
    robot.set_inverse(
        DDPGIK,
        bounds=ANGLE_LIMITS,
        obstacles=DEFAULT_OBSTACLES,
        episodes=300,
        max_steps=120,
        hidden_dims=[256, 256],
        gamma=0.99,
        batch_size=128,
        buffer_size=100_000,
        noise_theta=0.15,
        error_weight_mode="exp",
        **DDPG_TUNED_KWARGS,
        max_orientation_weight=0.95,
        position_tolerance=1e-2,
        orientation_tolerance=1e-2,
        save_episode_images=True,
        image_dir=str(episode_frames),
        weights_path="checkpoints/ddpg_ik.pt",
        save_weights_after_run=True,
        load_weights_if_exist=True,
        save_training_plot=True,
        training_plot_path=str(curve_png),
    )

    angles, metrics = robot.solve(DEFAULT_TARGET)
    robot.visualize(angles, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    robot.ik_solver.create_animation(str(approach_video), frame_interval=200)

    print("МЕТРИКИ DDPG:")
    print(f"  total_time: {metrics['total_time']:.4f} s")
    print(f"  combined_task_error: {metrics['combined_task_error']:.6f}")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f}")
    print(f"  кривая обучения (ошибки по эпизодам): {curve_png}")
    print(f"  кадры эпизодов: {episode_frames}")
    print(f"  видео подхода к цели (MP4): {approach_video}")

    angles_refined, m2 = robot.op_solve(angles, DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    robot.visualize(angles_refined, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    print("После op_solve:")
    print(f"  position_error: {m2['position_error']:.6f}")
    print(f"  orientation_error: {m2['orientation_error']:.6f}")


if __name__ == "__main__":
    main()
