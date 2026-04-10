import numpy as np
import logging

from robots.robot import Robot
from robots.utils import Coords, Sphere

from control.IK.ddpg import DDPGIK

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')

L_upper   = 0.5
L_forearm = 0.5
L_wrist   = 0.20

dh_parameters = [
    (0.0,        np.pi/2,     0.0),
    (0.0,       -np.pi/2,     0.0),
    (L_upper,     0.0,        0.0),
    (L_forearm,   0.0,        0.0),
    (0.0,        np.pi/2,     0.0),
    (0.0,       -np.pi/2,     0.0),
    (L_wrist,     0.0,        0.0),
]

angle_limits = [
    (-np.pi/2, np.pi/4),
    (0, np.pi),
    (0, 0),
    (0, np.pi),
    (-np.pi/2, np.pi/2),
    (-np.pi/2, np.pi/2),
    (0, 2*np.pi),
]

target_position = np.array([-0.3, -0.10, 0.4])
target_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
target = Coords(target_position, target_rotation)

obstacles = [
    Sphere(Coords([0.2, 0.0, 0.0]), 0.1),
    Sphere(Coords([0.0, -0.2, 0.2]), 0.1),
    Sphere(Coords([0.0, 0.2, 0.2]), 0.1),
]

robot = Robot(dh_parameters)
robot.set_inverse(DDPGIK,
                  bounds=angle_limits,
                  obstacles=obstacles,
                  episodes=500,
                  max_steps=150,
                  hidden_dims=[256, 256],
                  actor_lr=1e-4,
                  critic_lr=1e-3,
                  gamma=0.99,
                  tau=0.005,
                  batch_size=128,
                  buffer_size=200_000,
                  action_scale=0.05,
                  noise_sigma=0.2,
                  noise_theta=0.15,
                  warmup_steps=256,
                  error_weight_mode="exp",
                  exp_weight_alpha=5.0,
                  max_orientation_weight=0.95,
                  position_tolerance=1e-2,
                  orientation_tolerance=1e-2,
                  save_episode_images=True,
                  image_dir="ddpg_ik_frames",
                  weights_path="checkpoints/ddpg_ik.pt",
                  save_weights_after_run=True,
                  load_weights_if_exist=True)

angles, metrics = robot.solve(target)
robot.visualize(angles, target=target, obstacles=obstacles)
robot.ik_solver.create_animation("ddpg_ik_solution.gif", frame_interval=300)

print("\nМЕТРИКИ DDPG:")
print(f"Общее время выполнения: {metrics['total_time']:.4f} секунд")
print(f"Лучшее значение фитнеса: {metrics['best_fitness']:.6f}")
print(f"Эпизодов завершено: {metrics['episodes_completed']}")
print(f"Всего шагов: {metrics['total_steps']}")
print(f"Целевая позиция: {metrics['target_position']}")
print(f"Достигнутая позиция: {metrics['achieved_position']}")
print(f"Ошибка позиции: {metrics['position_error']:.6f}")
print(f"Целевая ориентация:\n{metrics['target_orientation']}")
print(f"Достигнутая ориентация:\n{metrics['achieved_orientation']}")
print(f"Ошибка ориентации: {metrics['orientation_error']:.6f} радиан")

# Быстрый прогон только с загруженными весами (без обучения), другая цель:
# new_target = Coords(np.array([0.2, 0.1, 0.5]), target_rotation)
# angles_fast, m = robot.solve(new_target, episodes=0)

# Уточнение методом Левенберга-Марквардта
angles_refined, metrics_ref = robot.op_solve(angles, target, obstacles=obstacles)
robot.visualize(angles_refined, target=target, obstacles=obstacles)

print("\nМЕТРИКИ ЛЕВЕНБЕРГА-МАРКВАРДТА ПОСЛЕ DDPG:")
print(f"Общее время выполнения: {metrics_ref['total_time']:.4f} секунд")
print(f"Лучшее значение фитнеса: {metrics_ref['best_fitness']:.6f}")
print(f"Целевая позиция: {metrics_ref['target_position']}")
print(f"Достигнутая позиция: {metrics_ref['achieved_position']}")
print(f"Ошибка позиции: {metrics_ref['position_error']:.6f}")
print(f"Целевая ориентация:\n{metrics_ref['target_orientation']}")
print(f"Достигнутая ориентация:\n{metrics_ref['achieved_orientation']}")
print(f"Ошибка ориентации: {metrics_ref['orientation_error']:.6f} радиан")
