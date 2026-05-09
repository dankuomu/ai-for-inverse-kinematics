import os
import sys
import pathlib
import unittest

import numpy as np

# Headless matplotlib — utils.py тянет pyplot на импорте.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robots.utils import Coords


ATOL = 1e-9


def _is_orthonormal(rot: np.ndarray) -> bool:
    """Проверяет, что матрица ортогональная и det=+1 (правая тройка)."""
    return (
        np.allclose(rot @ rot.T, np.eye(3), atol=ATOL)
        and np.isclose(np.linalg.det(rot), 1.0, atol=ATOL)
    )


class CoordsInitTests(unittest.TestCase):
    def test_default_init_is_origin_identity(self):
        c = Coords()
        np.testing.assert_allclose(c.pos, np.zeros(3), atol=ATOL)
        np.testing.assert_allclose(c.rot_matrix, np.eye(3), atol=ATOL)

    def test_position_only_keeps_identity_rotation(self):
        c = Coords(position=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(c.pos, [1.0, 2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(c.rot_matrix, np.eye(3), atol=ATOL)

    def test_two_rotation_specifiers_raise(self):
        # Передавать одновременно rpy и quaternion запрещено.
        with self.assertRaises(ValueError):
            Coords(rpy=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0])

    def test_rotation_matrix_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Coords(rotation_matrix=np.eye(4))

    def test_rpy_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Coords(rpy=[0.0, 0.0])

    def test_quaternion_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Coords(quaternion=[1.0, 0.0, 0.0])

    def test_axis_angle_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Coords(axis_angle=[1.0, 0.0, 0.0])


class CoordsCopySemanticsTests(unittest.TestCase):
    def test_pos_property_returns_independent_copy(self):
        c = Coords(position=[1.0, 2.0, 3.0])
        p = c.pos
        p[0] = 99.0
        np.testing.assert_allclose(c.pos, [1.0, 2.0, 3.0], atol=ATOL)

    def test_rot_matrix_property_returns_independent_copy(self):
        c = Coords(rpy=[0.1, 0.2, 0.3])
        r = c.rot_matrix
        r[0, 0] = 99.0
        # Внутренняя матрица не должна испортиться.
        self.assertTrue(_is_orthonormal(c.rot_matrix))


class CoordsRotationRoundtripTests(unittest.TestCase):
    def test_rotation_matrix_init_preserves_matrix(self):
        # Простой Rz(π/4) — заранее ортогональная матрица.
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        rot = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        coords = Coords(rotation_matrix=rot)
        np.testing.assert_allclose(coords.rot_matrix, rot, atol=ATOL)
        self.assertTrue(_is_orthonormal(coords.rot_matrix))

    def test_rpy_roundtrip_non_singular(self):
        rpy = np.array([0.1, 0.2, 0.3])
        coords = Coords(rpy=rpy)
        self.assertTrue(_is_orthonormal(coords.rot_matrix))
        # Извлечённые углы должны совпасть со входом (вне сингулярности).
        np.testing.assert_allclose(coords.RPY, rpy, atol=1e-9)

    def test_quaternion_input_normalized(self):
        # Ненормированный кватернион → внутри Coords нормируется,
        # rot_matrix всё равно ортогональная.
        coords = Coords(quaternion=[2.0, 0.0, 0.0, 0.0])
        self.assertTrue(_is_orthonormal(coords.rot_matrix))
        np.testing.assert_allclose(coords.rot_matrix, np.eye(3), atol=ATOL)

    def test_quaternion_roundtrip_via_matrix(self):
        # Поворот вокруг Z на π/3.
        angle = np.pi / 3
        q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        coords_in = Coords(quaternion=q)
        # Извлекаем кватернион и собираем заново — матрицы должны совпасть.
        coords_back = Coords(quaternion=coords_in.quaternion)
        np.testing.assert_allclose(coords_back.rot_matrix, coords_in.rot_matrix, atol=1e-9)

    def test_axis_angle_roundtrip_via_matrix(self):
        # Поворот вокруг (1, 1, 0) на 0.7 рад.
        axis = np.array([1.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        angle = 0.7
        coords_in = Coords(axis_angle=[axis[0], axis[1], axis[2], angle])
        self.assertTrue(_is_orthonormal(coords_in.rot_matrix))
        coords_back = Coords(axis_angle=coords_in.axis_angle)
        np.testing.assert_allclose(coords_back.rot_matrix, coords_in.rot_matrix, atol=1e-9)

    def test_axis_angle_zero_norm_gives_identity(self):
        # Нулевая ось — спецслучай, должен дать единичную матрицу.
        coords = Coords(axis_angle=[0.0, 0.0, 0.0, 1.23])
        np.testing.assert_allclose(coords.rot_matrix, np.eye(3), atol=ATOL)


class CoordsTransformTests(unittest.TestCase):
    def test_transform_point_identity_is_translation(self):
        coords = Coords(position=[1.0, 2.0, 3.0])
        out = coords.transform_point([10.0, 20.0, 30.0])
        np.testing.assert_allclose(out, [11.0, 22.0, 33.0], atol=ATOL)

    def test_transform_point_with_z_rotation(self):
        # Rz(π/2): точка (1, 0, 0) → (0, 1, 0), затем сдвиг.
        coords = Coords(position=[0.0, 0.0, 5.0], rpy=[0.0, 0.0, np.pi / 2])
        out = coords.transform_point([1.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 1.0, 5.0], atol=1e-9)

    def test_inverse_composes_to_identity_on_points(self):
        # T.inverse().transform_point(T.transform_point(p)) ≈ p.
        coords = Coords(position=[1.5, -2.0, 0.7], rpy=[0.4, -0.3, 0.6])
        inv = coords.inverse()
        p = np.array([0.2, 0.5, -1.1])
        roundtrip = inv.transform_point(coords.transform_point(p))
        np.testing.assert_allclose(roundtrip, p, atol=1e-9)

    def test_inverse_rotation_is_transpose(self):
        coords = Coords(position=[1.0, 2.0, 3.0], rpy=[0.4, -0.3, 0.6])
        inv = coords.inverse()
        np.testing.assert_allclose(inv.rot_matrix, coords.rot_matrix.T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
