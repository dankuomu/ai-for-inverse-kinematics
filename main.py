import numpy as np

from robots.robot import Robot
from robots.utils import Coords, Sphere, Capsule

from control.IK.genetic import GeneticIK

def generate_random_targets(robot, bounds, n_samples=100):
    targets = []
    for _ in range(n_samples):
        angles = np.array([
            np.random.uniform(low, high) for (low, high) in bounds
        ])
        coords = robot.forward_kinematics(angles)
        targets.append(coords)
    return targets

L_upper   = 0.5   # плечо (m)
L_forearm = 0.5   # предплечье (m)
L_wrist   = 0.20  # кисть (m)

dh_parameters = [
    # (a,        alpha,       d)
    (0.0,        np.pi/2,     0.0),
    (0.0,       -np.pi/2,     0.0),
    (L_upper,     0.0,        0.0),
    (L_forearm,   0.0,        0.0),
    (0.0,        np.pi/2,     0.0),
    (0.0,       -np.pi/2,     0.0),
    (L_wrist,     0.0,        0.0),
]


# Ограничения на углы
angle_limits = [
    (-np.pi/2, np.pi/4),   # α
    (0, np.pi),            # β
    (0, 0),                # γ
    (0, np.pi),            # δ
    (-np.pi/2, np.pi/2),   # ε
    (-np.pi/2, np.pi/2),   # θ
    (0, 2*np.pi)           # ι
]
bounds = angle_limits

target_position = np.array([-0.3, -0.10, 0.4])
target_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
target = Coords(target_position, target_rotation)

obstacles = [
    Sphere(Coords([0.2, 0.0, 0.0]), 0.1),
    Sphere(Coords([0.0, -0.2, 0.2]), 0.1),
    Sphere(Coords([0.0, 0.2, 0.2]), 0.1),
]

robot = Robot(dh_parameters)
robot.set_inverse(GeneticIK, obstacles=obstacles)

angles, metrics = robot.solve(target)
robot.visualize(angles, target=target, obstacles=obstacles)
robot.ik_solver.create_animation("genetic_ik_solution.gif", frame_interval=300)

print("МЕТРИКИ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
print(f"Общее время выполнения: {metrics['total_time']:.4f} секунд")
print(f"Лучшее значение фитнеса: {metrics['best_fitness']:.6f}")
print(f"Целевая позиция: {metrics['target_position']}")
print(f"Достигнутая позиция: {metrics['achieved_position']}")
print(f"Ошибка позиции: {metrics['position_error']:.6f}")
print(f"Целевая ориентация: {metrics['target_orientation']}")
print(f"Достигнутая ориентация: {metrics['achieved_orientation']}")
print(f"Ошибка ориентации: {metrics['orientation_error']:.6f} радиан")

angles_refined, metrics = robot.op_solve(angles, target)
robot.visualize(angles_refined, target=target)

print("МЕТРИКИ НЬЮТОНА-РАФСОНА ПОСЛЕ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
print(f"Общее время выполнения: {metrics['total_time']:.4f} секунд")
print(f"Лучшее значение фитнеса: {metrics['best_fitness']:.6f}")
print(f"Целевая позиция: {metrics['target_position']}")
print(f"Достигнутая позиция: {metrics['achieved_position']}")
print(f"Ошибка позиции: {metrics['position_error']:.6f}")
print(f"Целевая ориентация: {metrics['target_orientation']}")
print(f"Достигнутая ориентация: {metrics['achieved_orientation']}")
print(f"Ошибка ориентации: {metrics['orientation_error']:.6f} радиан")


# Другие алгоритмы и ML — готовые сценарии в каталоге examples/ (из корня репозитория):
#   python examples/genetic_ik.py
#   python examples/ddpg_ik.py
#   python examples/ml_rf_ik.py
#   python examples/ml_xgb_ik.py
#   python examples/ml_nn_ik.py
#
# Тюнинг DDPG (сетка гиперпараметров): python main_ddpg_tune.py
#
# ML: датасет по сетке суставов — robot.ik_solver.generate_dataset_grid(
#     angle_limits, joint_value_grid={0: [...], 1: [...]}, default_angles=...)
# см. examples/ml_rf_ik.py
