import os
import sys
import pathlib
import unittest

import numpy as np

# Ensure headless-friendly matplotlib imports (some modules import pyplot on import).
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robots.robot import Robot
from control.IK.genetic import GeneticIK


def _to_10_pow(x: float, digits: int = 3) -> str:
    if x == 0.0:
        return "0·10^0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return f"{mantissa:.{digits}f}·10^{exponent}"


def _build_robot_7dof(*, population_size: int, generations: int, error_weight_mode: str,
                      constant_orientation_weight: float) -> Robot:
    # Geometry from `main.py`.
    L_upper = 0.5  # плечо (m)
    L_forearm = 0.5  # предплечье (m)
    L_wrist = 0.20  # кисть (m)

    dh_parameters = [
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_upper, 0.0, 0.0),
        (L_forearm, 0.0, 0.0),
        (0.0, np.pi / 2, 0.0),
        (0.0, -np.pi / 2, 0.0),
        (L_wrist, 0.0, 0.0),
    ]

    robot = Robot(dh_parameters)
    robot.set_inverse(
        GeneticIK,
        obstacles=[],
        population_size=population_size,
        generations=generations,
        elite_size=max(5, population_size // 8),
        error_weight_mode=error_weight_mode,
        constant_orientation_weight=constant_orientation_weight,
        # Keep early stopping conservative so results are repeatable across machines.
        early_stopping_patience=10,
        max_no_improvement=10,
        save_generation_images=False,
    )
    return robot


def _build_robot_8dof(*, population_size: int, generations: int, error_weight_mode: str,
                      constant_orientation_weight: float) -> Robot:
    # Add one extra revolute joint after the original chain.
    # (a=0, alpha=0, d=0) => pure end-frame rotation, no extra translation.
    base = _build_robot_7dof(
        population_size=population_size,
        generations=generations,
        error_weight_mode=error_weight_mode,
        constant_orientation_weight=constant_orientation_weight,
    )
    dh_parameters = list(base.dh_params) + [(0.0, 0.0, 0.0)]
    robot = Robot(dh_parameters)
    robot.set_inverse(
        GeneticIK,
        obstacles=[],
        population_size=population_size,
        generations=generations,
        elite_size=max(5, population_size // 8),
        error_weight_mode=error_weight_mode,
        constant_orientation_weight=constant_orientation_weight,
        early_stopping_patience=10,
        max_no_improvement=10,
        save_generation_images=False,
    )
    return robot


def _build_robot_15dof(*, population_size: int, generations: int, error_weight_mode: str,
                       constant_orientation_weight: float) -> Robot:
    # Start from base 7DOF chain and append 8 pure revolute joints.
    base = _build_robot_7dof(
        population_size=population_size,
        generations=generations,
        error_weight_mode=error_weight_mode,
        constant_orientation_weight=constant_orientation_weight,
    )
    dh_parameters = list(base.dh_params) + [(0.0, 0.0, 0.0)] * 8
    robot = Robot(dh_parameters)
    robot.set_inverse(
        GeneticIK,
        obstacles=[],
        population_size=population_size,
        generations=generations,
        elite_size=max(5, population_size // 8),
        error_weight_mode=error_weight_mode,
        constant_orientation_weight=constant_orientation_weight,
        early_stopping_patience=10,
        max_no_improvement=10,
        save_generation_images=False,
    )
    return robot


def _generate_random_targets(*, robot: Robot, n_samples: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    dof = len(robot.dh_params)
    targets = []
    for _ in range(n_samples):
        angles = rng.uniform(low=-np.pi, high=np.pi, size=dof)
        targets.append(robot.forward_kinematics(angles))
    return targets


def _solve_and_collect_errors(*, robot: Robot, targets: list, seed: int) -> tuple[list, list, list, list]:
    pos_errors: list[float] = []
    rot_errors: list[float] = []
    pos_errors_refined: list[float] = []
    rot_errors_refined: list[float] = []

    for i, target in enumerate(targets):
        # Make each IK run deterministic w.r.t. its index.
        np.random.seed(seed + i)
        angles, metrics = robot.solve(target)
        _, refined_metrics = robot.op_solve(angles, target, max_iter=1)

        pos_errors.append(float(metrics["position_error"]))
        rot_errors.append(float(metrics["orientation_error"]))
        pos_errors_refined.append(float(refined_metrics["position_error"]))
        rot_errors_refined.append(float(refined_metrics["orientation_error"]))

    return pos_errors, rot_errors, pos_errors_refined, rot_errors_refined


class GeneticIKMeanErrorTests(unittest.TestCase):
    def test_mean_error_7dof_exp_weights_100_points(self):
        n = int(os.getenv("IK_TEST_N", "100"))
        seed = int(os.getenv("IK_TEST_SEED", "0"))

        population_size = int(os.getenv("IK_TEST_POP_SIZE", "40"))
        generations = int(os.getenv("IK_TEST_GENERATIONS", "15"))
        const_w_rot = float(os.getenv("IK_TEST_CONSTANT_W_ROT", "0.5"))

        robot = _build_robot_7dof(
            population_size=population_size,
            generations=generations,
            error_weight_mode="exp",
            constant_orientation_weight=const_w_rot,
        )
        targets = _generate_random_targets(robot=robot, n_samples=n, seed=seed)

        pos_errors, rot_errors, pos_errors_refined, rot_errors_refined = _solve_and_collect_errors(
            robot=robot,
            targets=targets,
            seed=seed,
        )
        mean_pos = float(np.mean(pos_errors))
        mean_rot = float(np.mean(rot_errors))
        mean_pos_refined = float(np.mean(pos_errors_refined))
        mean_rot_refined = float(np.mean(rot_errors_refined))

        # Printed output is useful for tuning GA parameters.
        print(f"[7dof|exp weights] N={n} mean position error={mean_pos:.6f} m, mean orientation error={mean_rot:.6f} rad")
        print(
            f"[7dof|exp weights|refine=1] refined mean position error={_to_10_pow(mean_pos_refined)} m, "
            f"refined mean orientation error={_to_10_pow(mean_rot_refined)} rad"
        )

        # Very loose sanity checks: we mainly want "doesn't crash" + "not completely diverged".
        self.assertTrue(np.isfinite(mean_pos))
        self.assertTrue(np.isfinite(mean_rot))
        self.assertTrue(np.isfinite(mean_pos_refined))
        self.assertTrue(np.isfinite(mean_rot_refined))
        self.assertLess(mean_pos, float(os.getenv("IK_TEST_MAX_MEAN_POS_ERR", "1.0")))

    def test_mean_error_7dof_constant_weights_100_points(self):
        n = int(os.getenv("IK_TEST_N", "100"))
        seed = int(os.getenv("IK_TEST_SEED", "0"))

        population_size = int(os.getenv("IK_TEST_POP_SIZE", "40"))
        generations = int(os.getenv("IK_TEST_GENERATIONS", "15"))
        const_w_rot = float(os.getenv("IK_TEST_CONSTANT_W_ROT", "0.5"))

        robot = _build_robot_7dof(
            population_size=population_size,
            generations=generations,
            error_weight_mode="constant",
            constant_orientation_weight=const_w_rot,
        )
        targets = _generate_random_targets(robot=robot, n_samples=n, seed=seed)

        pos_errors, rot_errors, pos_errors_refined, rot_errors_refined = _solve_and_collect_errors(
            robot=robot,
            targets=targets,
            seed=seed,
        )
        mean_pos = float(np.mean(pos_errors))
        mean_rot = float(np.mean(rot_errors))
        mean_pos_refined = float(np.mean(pos_errors_refined))
        mean_rot_refined = float(np.mean(rot_errors_refined))

        print(f"[7dof|constant weights w_rot={const_w_rot:.2f}] N={n} mean position error={mean_pos:.6f} m, mean orientation error={mean_rot:.6f} rad")
        print(
            f"[7dof|constant weights|refine=1] refined mean position error={_to_10_pow(mean_pos_refined)} m, "
            f"refined mean orientation error={_to_10_pow(mean_rot_refined)} rad"
        )

        self.assertTrue(np.isfinite(mean_pos))
        self.assertTrue(np.isfinite(mean_rot))
        self.assertTrue(np.isfinite(mean_pos_refined))
        self.assertTrue(np.isfinite(mean_rot_refined))
        self.assertLess(mean_pos, float(os.getenv("IK_TEST_MAX_MEAN_POS_ERR", "1.0")))

    def test_mean_error_15dof_exp_weights_100_points(self):
        n = int(os.getenv("IK_TEST_N", "100"))
        seed = int(os.getenv("IK_TEST_SEED", "0"))

        population_size = int(os.getenv("IK_TEST_POP_SIZE", "40"))
        generations = int(os.getenv("IK_TEST_GENERATIONS", "15"))
        const_w_rot = float(os.getenv("IK_TEST_CONSTANT_W_ROT", "0.5"))

        robot = _build_robot_15dof(
            population_size=population_size,
            generations=generations,
            error_weight_mode="exp",
            constant_orientation_weight=const_w_rot,
        )
        targets = _generate_random_targets(robot=robot, n_samples=n, seed=seed)

        pos_errors, rot_errors, pos_errors_refined, rot_errors_refined = _solve_and_collect_errors(
            robot=robot,
            targets=targets,
            seed=seed,
        )
        mean_pos = float(np.mean(pos_errors))
        mean_rot = float(np.mean(rot_errors))
        mean_pos_refined = float(np.mean(pos_errors_refined))
        mean_rot_refined = float(np.mean(rot_errors_refined))

        print(f"[15dof|exp weights] N={n} mean position error={mean_pos:.6f} m, mean orientation error={mean_rot:.6f} rad")
        print(
            f"[15dof|exp weights|refine=1] refined mean position error={_to_10_pow(mean_pos_refined)} m, "
            f"refined mean orientation error={_to_10_pow(mean_rot_refined)} rad"
        )

        self.assertTrue(np.isfinite(mean_pos))
        self.assertTrue(np.isfinite(mean_rot))
        self.assertTrue(np.isfinite(mean_pos_refined))
        self.assertTrue(np.isfinite(mean_rot_refined))
        self.assertLess(mean_pos, float(os.getenv("IK_TEST_MAX_MEAN_POS_ERR", "1.0")))

    def test_mean_error_15dof_constant_weights_100_points(self):
        n = int(os.getenv("IK_TEST_N", "100"))
        seed = int(os.getenv("IK_TEST_SEED", "0"))

        population_size = int(os.getenv("IK_TEST_POP_SIZE", "40"))
        generations = int(os.getenv("IK_TEST_GENERATIONS", "15"))
        const_w_rot = float(os.getenv("IK_TEST_CONSTANT_W_ROT", "0.5"))

        robot = _build_robot_15dof(
            population_size=population_size,
            generations=generations,
            error_weight_mode="constant",
            constant_orientation_weight=const_w_rot,
        )
        targets = _generate_random_targets(robot=robot, n_samples=n, seed=seed)

        pos_errors, rot_errors, pos_errors_refined, rot_errors_refined = _solve_and_collect_errors(
            robot=robot,
            targets=targets,
            seed=seed,
        )
        mean_pos = float(np.mean(pos_errors))
        mean_rot = float(np.mean(rot_errors))
        mean_pos_refined = float(np.mean(pos_errors_refined))
        mean_rot_refined = float(np.mean(rot_errors_refined))

        print(
            f"[15dof|constant weights w_rot={const_w_rot:.2f}] N={n} mean position error={mean_pos:.6f} m, "
            f"mean orientation error={mean_rot:.6f} rad"
        )
        print(
            f"[15dof|constant weights|refine=1] refined mean position error={_to_10_pow(mean_pos_refined)} m, "
            f"refined mean orientation error={_to_10_pow(mean_rot_refined)} rad"
        )

        self.assertTrue(np.isfinite(mean_pos))
        self.assertTrue(np.isfinite(mean_rot))
        self.assertTrue(np.isfinite(mean_pos_refined))
        self.assertTrue(np.isfinite(mean_rot_refined))
        self.assertLess(mean_pos, float(os.getenv("IK_TEST_MAX_MEAN_POS_ERR", "1.0")))

