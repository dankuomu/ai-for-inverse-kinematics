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

from examples.common import DEFAULT_OBSTACLES, DEFAULT_TARGET, make_robot

from control.IK.genetic import GeneticIK


def main():
    robot = make_robot()
    robot.set_inverse(GeneticIK, obstacles=DEFAULT_OBSTACLES)

    angles, metrics = robot.solve(DEFAULT_TARGET)
    robot.visualize(angles, target=DEFAULT_TARGET, obstacles=DEFAULT_OBSTACLES)
    robot.ik_solver.create_animation("genetic_ik_solution.gif", frame_interval=300)

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
