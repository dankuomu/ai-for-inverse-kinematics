"""
Подбор гиперпараметров DDPGIK по сетке (как GeneticIK.tune / сигнатура ik_base.tune).

Запуск из корня репозитория:
    python main_ddpg_tune.py

После тюнинга в консоль выводятся best_params; дальше можно передать их в set_inverse
или вызывать solve() без kwargs — подставятся self.best_params.
"""
import json
import logging
import numpy as np

from robots.robot import Robot
from robots.utils import Coords, Sphere

from control.IK.ddpg import DDPGIK

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


def generate_random_targets(robot, bounds, n_samples: int = 8):
    targets = []
    for _ in range(n_samples):
        angles = np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds])
        targets.append(robot.forward_kinematics(angles))
    return targets


def main():
    L_upper, L_forearm, L_wrist = 0.5, 0.5, 0.20
    dh_parameters = [
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_upper, 0.0, 0.0),
        (L_forearm, 0.0, 0.0),
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_wrist, 0.0, 0.0),
    ]
    angle_limits = [
        (-np.pi / 2, np.pi / 4),
        (0, np.pi),
        (0, 0),
        (0, np.pi),
        (-np.pi / 2, np.pi / 2),
        (-np.pi / 2, np.pi / 2),
        (0, 2 * np.pi),
    ]

    obstacles = [
        Sphere(Coords([0.2, 0.0, 0.0]), 0.1),
        Sphere(Coords([0.0, -0.2, 0.2]), 0.1),
        Sphere(Coords([0.0, 0.2, 0.2]), 0.1),
    ]

    robot = Robot(dh_parameters)
    robot.set_inverse(
        DDPGIK,
        bounds=angle_limits,
        obstacles=obstacles,
        episodes=200,
        max_steps=120,
        hidden_dims=[256, 256],
        batch_size=128,
        buffer_size=100_000,
        warmup_steps=256,
        error_weight_mode="exp",
        exp_weight_alpha=5.0,
        max_orientation_weight=0.95,
        position_tolerance=1e-2,
        orientation_tolerance=1e-2,
        save_episode_images=False,
        weights_path=None,
        save_weights_after_run=False,
        load_weights_if_exist=False,
    )

    np.random.seed(42)
    targets = generate_random_targets(robot, angle_limits, n_samples=6)

    param_grid = {
        "actor_lr": [1e-4, 3e-4],
        "critic_lr": [5e-4, 1e-3],
        "action_scale": [0.04, 0.06, 0.08],
        "noise_sigma": [0.15, 0.25],
        "exp_weight_alpha": [4.0, 6.0],
        "tau": [0.005, 0.01],
    }

    result = robot.ik_solver.tune(targets, param_grid)

    print("\n=== РЕЗУЛЬТАТ ТЮНИНГА DDPG ===")
    print(f"best_mean_combined_error: {result['best_mean_combined_error']:.6f}")
    print("best_params:", json.dumps(result["best_params"], indent=2, default=str))
    with open("ddpg_tune_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nПолный history + best_params записаны в ddpg_tune_result.json")


if __name__ == "__main__":
    main()
