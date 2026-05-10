"""
Пример: Genetic IK + визуализация + уточнение op_solve.

Запуск из корня репозитория:
    python examples/genetic_ik.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.common import ANGLE_LIMITS, DEFAULT_OBSTACLES, DEFAULT_TARGET, make_robot

from control.IK.genetic import GeneticIK


def main():
    plots_dir = _ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = plots_dir / "genetic_frames"
    video_path = plots_dir / "genetic_approach.mp4"

    robot = make_robot()
    robot.set_inverse(
        GeneticIK,
        bounds=ANGLE_LIMITS,
        obstacles=DEFAULT_OBSTACLES,
        generations=120,
        population_size=200,
        save_generation_images=True,
        image_dir=str(frames_dir),
    )

    angles, metrics = robot.solve(DEFAULT_TARGET)
    robot.visualize(angles, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    robot.ik_solver.create_animation(str(video_path), frame_interval=250)

    print(f"Кадры поколений: {frames_dir}")
    print(f"видео подхода к цели (MP4): {video_path}")
    print("МЕТРИКИ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
    print(f"  total_time: {metrics['total_time']:.4f} s")
    print(f"  position_error: {metrics['position_error']:.6f}")
    print(f"  orientation_error: {metrics['orientation_error']:.6f}")

    angles_refined, m2 = robot.op_solve(angles, DEFAULT_TARGET)
    robot.visualize(angles_refined, target=DEFAULT_TARGET)
    print("После op_solve:")
    print(f"  position_error: {m2['position_error']:.6f}")
    print(f"  orientation_error: {m2['orientation_error']:.6f}")


if __name__ == "__main__":
    main()
