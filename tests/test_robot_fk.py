import os
import sys
import pathlib
import unittest

import numpy as np

# Headless matplotlib — robot.py тянет pyplot на импорте.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robots.robot import Robot
from robots.utils import Coords


# Геометрия 7-DOF манипулятора из main.py.
L_UPPER = 0.5
L_FOREARM = 0.5
L_WRIST = 0.20

DH_PARAMETERS_7DOF = [
    (0.0, np.pi / 2, 0.0),
    (0.0, -np.pi / 2, 0.0),
    (L_UPPER, 0.0, 0.0),
    (L_FOREARM, 0.0, 0.0),
    (0.0, np.pi / 2, 0.0),
    (0.0, -np.pi / 2, 0.0),
    (L_WRIST, 0.0, 0.0),
]


class ForwardKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(DH_PARAMETERS_7DOF)
        self.dof = len(DH_PARAMETERS_7DOF)

    def test_fk_at_zero_angles_is_extended_arm(self):
        # При нулевых углах все Rx(±π/2) гасят друг друга, остаётся вытянутая
        # рука вдоль X длиной L_upper + L_forearm + L_wrist = 1.2 м.
        angles = np.zeros(self.dof)
        pose = self.robot.forward_kinematics(angles)
        self.assertIsInstance(pose, Coords)
        np.testing.assert_allclose(pose.pos, [L_UPPER + L_FOREARM + L_WRIST, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(pose.rot_matrix, np.eye(3), atol=1e-12)

    def test_fk_wrong_number_of_angles_raises(self):
        with self.assertRaises(ValueError):
            self.robot.forward_kinematics(np.zeros(self.dof - 1))

    def test_fk_returns_orthogonal_rotation(self):
        # Случайные углы — ориентация всё равно должна быть ортогональной.
        rng = np.random.default_rng(42)
        angles = rng.uniform(-np.pi, np.pi, size=self.dof)
        pose = self.robot.forward_kinematics(angles)
        rot = pose.rot_matrix
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, delta=1e-9)


class JointPositionsTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(DH_PARAMETERS_7DOF)
        self.dof = len(DH_PARAMETERS_7DOF)

    def test_joint_positions_count_is_dof_plus_one(self):
        # Должны вернуться база + по точке после каждого звена.
        positions = self.robot.get_joint_positions(np.zeros(self.dof))
        self.assertEqual(positions.shape, (self.dof + 1, 3))

    def test_joint_positions_starts_at_origin(self):
        positions = self.robot.get_joint_positions(np.zeros(self.dof))
        np.testing.assert_allclose(positions[0], np.zeros(3), atol=1e-12)

    def test_joint_positions_last_matches_fk(self):
        # Кончик в get_joint_positions должен совпасть с forward_kinematics.pos.
        rng = np.random.default_rng(7)
        angles = rng.uniform(-np.pi, np.pi, size=self.dof)
        positions = self.robot.get_joint_positions(angles)
        pose = self.robot.forward_kinematics(angles)
        np.testing.assert_allclose(positions[-1], pose.pos, atol=1e-12)


class JointFramesTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(DH_PARAMETERS_7DOF)
        self.dof = len(DH_PARAMETERS_7DOF)

    def test_joint_frames_count_is_dof_plus_one(self):
        frames = self.robot.get_joint_frames(np.zeros(self.dof))
        self.assertEqual(len(frames), self.dof + 1)
        for frame in frames:
            self.assertEqual(frame.shape, (4, 4))

    def test_joint_frames_first_is_identity(self):
        frames = self.robot.get_joint_frames(np.zeros(self.dof))
        np.testing.assert_allclose(frames[0], np.eye(4), atol=1e-12)

    def test_joint_frames_last_matches_fk(self):
        # Последний кадр должен соответствовать позе кончика.
        rng = np.random.default_rng(123)
        angles = rng.uniform(-np.pi, np.pi, size=self.dof)
        frames = self.robot.get_joint_frames(angles)
        pose = self.robot.forward_kinematics(angles)
        np.testing.assert_allclose(frames[-1][:3, 3], pose.pos, atol=1e-12)
        np.testing.assert_allclose(frames[-1][:3, :3], pose.rot_matrix, atol=1e-12)

    def test_joint_frames_wrong_number_of_angles_raises(self):
        with self.assertRaises(ValueError):
            self.robot.get_joint_frames(np.zeros(self.dof + 1))


class IkSolverNotSetTests(unittest.TestCase):
    def test_solve_without_inverse_raises(self):
        # Без вызова set_inverse — solve должен явно ругаться.
        robot = Robot(DH_PARAMETERS_7DOF)
        target = robot.forward_kinematics(np.zeros(len(DH_PARAMETERS_7DOF)))
        with self.assertRaises(RuntimeError):
            robot.solve(target)


if __name__ == "__main__":
    unittest.main()
