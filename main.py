import numpy as np

from robots.robot import Robot
from robots.utils import Coords
from IK.genetic import GeneticIK

def generate_random_targets(robot, bounds, n_samples=100):
    targets = []
    for _ in range(n_samples):
        angles = np.array([
            np.random.uniform(low, high) for (low, high) in bounds
        ])
        coords = robot.forward_kinematics(angles)
        targets.append(coords)
    return targets

# Длины звеньев (предполагаемые значения)
L1 = 0.5  # длина плеча
L2 = 0.5  # длина предплечья
L3 = 0.3  # длина кисти

# Параметры Денавита-Хартенберга
dh_parameters = [
    # (a, alpha, d) - параметры для каждого звена
    (0, 0, 0),          # Плечевой сустав (α)
    (0, -np.pi/2, 0),   # Плечевой сустав (β)
    (0, np.pi/2, 0),    # Плечевой сустав (γ)
    (L1, 0, 0),         # Локтевой сустав (δ)
    (0, 0, L2),         # Кистевой сустав (ε)
    (0, np.pi/2, 0),    # Кистевой сустав (θ)
    (0, -np.pi/2, L3)   # Кистевой сустав (ι)
]

# Ограничения на углы
angle_limits = [
    (-np.pi/2, np.pi/4),   # α
    (0, np.pi),            # β
    (-np.pi/2, np.pi/2),   # γ
    (0, np.pi),            # δ
    (-np.pi/2, np.pi/2),   # ε
    (-np.pi/2, np.pi/2),   # θ
    (0, 2*np.pi)           # ι
]
bounds = angle_limits

# Целевое положение и ориентация

target_position = np.array([0.5, 0.0, 0.5])
target_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
target = Coords(target_position, target_rotation)

# Создание робота
robot = Robot(dh_parameters)
robot.set_inverse(GeneticIK)

#train_targets = generate_random_targets(robot, bounds, n_samples=100)

#best_params, all_results = robot.ik_solver.tune(train_targets)

angles, metrics = robot.solve(target)
robot.visualize(angles, target_point=target.pos) # TODO по поколениям, как выглядит поверхность ошибки, другой критерий выхода, углы

print("МЕТРИКИ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
print(f"Общее время выполнения: {metrics['total_time']:.4f} секунд")
print(f"Лучшее значение фитнеса: {metrics['best_fitness']:.6f}")
print(f"Целевая позиция: {metrics['target_position']}")
print(f"Достигнутая позиция: {metrics['achieved_position']}")
print(f"Ошибка позиции: {metrics['position_error']:.6f}")
print(f"Целевая ориентация: {metrics['target_orientation']}")
print(f"Достигнутая ориентация: {metrics['achieved_orientation']}")
print(f"Ошибка ориентации: {metrics['orientation_error']:.6f} радиан")