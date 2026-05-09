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

from robots.utils import Coords, Obstacle, Sphere, Capsule


ATOL = 1e-9


class SegToSegDistanceTests(unittest.TestCase):
    """Тесты статика Obstacle._dist_seg_to_seg — чистая геометрия."""

    def test_intersecting_segments_zero_distance(self):
        # Два отрезка, пересекающиеся в начале координат.
        a1 = np.array([-1.0, 0.0, 0.0])
        a2 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([0.0, -1.0, 0.0])
        b2 = np.array([0.0, 1.0, 0.0])
        d = Obstacle._dist_seg_to_seg(a1, a2, b1, b2)
        self.assertAlmostEqual(d, 0.0, delta=1e-9)

    def test_parallel_segments_offset_returns_offset(self):
        # Параллельные по X отрезки, разнесённые по Y на 2.0 — расстояние ровно 2.0.
        a1 = np.array([0.0, 0.0, 0.0])
        a2 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([0.0, 2.0, 0.0])
        b2 = np.array([1.0, 2.0, 0.0])
        d = Obstacle._dist_seg_to_seg(a1, a2, b1, b2)
        self.assertAlmostEqual(d, 2.0, delta=1e-9)

    def test_skew_segments_known_distance(self):
        # Отрезок A вдоль X на z=0, отрезок B вдоль Y на z=3 — расстояние ровно 3.
        a1 = np.array([0.0, 0.0, 0.0])
        a2 = np.array([2.0, 0.0, 0.0])
        b1 = np.array([1.0, -1.0, 3.0])
        b2 = np.array([1.0,  1.0, 3.0])
        d = Obstacle._dist_seg_to_seg(a1, a2, b1, b2)
        self.assertAlmostEqual(d, 3.0, delta=1e-9)

    def test_degenerate_point_to_point(self):
        # Оба «отрезка» — одна точка: должна вернуться обычная евклидова норма.
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        d = Obstacle._dist_seg_to_seg(p1, p1, p2, p2)
        self.assertAlmostEqual(d, 5.0, delta=1e-9)

    def test_collinear_non_overlapping_segments(self):
        # Два коллинеарных отрезка вдоль X с зазором 1.0.
        a1 = np.array([0.0, 0.0, 0.0])
        a2 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([2.0, 0.0, 0.0])
        b2 = np.array([3.0, 0.0, 0.0])
        d = Obstacle._dist_seg_to_seg(a1, a2, b1, b2)
        self.assertAlmostEqual(d, 1.0, delta=1e-9)


class SphereTests(unittest.TestCase):
    def setUp(self):
        self.center = Coords(position=[1.0, 2.0, 3.0])
        self.radius = 0.5
        self.sphere = Sphere(self.center, self.radius)

    def test_distance_to_point_at_center_is_negative_radius(self):
        d = self.sphere.distance_to_point(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(d, -self.radius, delta=1e-9)

    def test_distance_to_point_on_surface_is_zero(self):
        # Точка строго на поверхности.
        p = np.array([1.0 + self.radius, 2.0, 3.0])
        d = self.sphere.distance_to_point(p)
        self.assertAlmostEqual(d, 0.0, delta=1e-9)

    def test_distance_to_point_outside(self):
        # Точка на расстоянии 2.0 от центра — distance = 2.0 - r.
        p = np.array([3.0, 2.0, 3.0])
        d = self.sphere.distance_to_point(p)
        self.assertAlmostEqual(d, 2.0 - self.radius, delta=1e-9)

    def test_dist_to_me_segment_through_center_returns_zero(self):
        # Отрезок проходит через центр сферы — должен вернуться 0 (зазор отрицательный, max'нулся).
        seg_1 = Coords(position=[0.0, 2.0, 3.0])
        seg_2 = Coords(position=[2.0, 2.0, 3.0])
        d = self.sphere.dist_to_me(seg_1, seg_2)
        self.assertEqual(d, 0.0)

    def test_dist_to_me_segment_far_away(self):
        # Отрезок параллельно X на расстоянии 5 от центра — d = 5 - r.
        seg_1 = Coords(position=[0.0, 7.0, 3.0])
        seg_2 = Coords(position=[2.0, 7.0, 3.0])
        d = self.sphere.dist_to_me(seg_1, seg_2)
        self.assertAlmostEqual(d, 5.0 - self.radius, delta=1e-9)


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        # Капсула высотой 2 вдоль Z, радиус 0.3, центр в (0, 0, 0).
        self.center = Coords(position=[0.0, 0.0, 0.0])
        self.capsule = Capsule(self.center, radius=0.3, height=2.0,
                               local_axis=np.array([0.0, 0.0, 1.0]))

    def test_world_axis_default_orientation_matches_local(self):
        # У Coords без поворота world_axis должен совпасть с local_axis.
        np.testing.assert_allclose(self.capsule.world_axis, [0.0, 0.0, 1.0], atol=1e-12)

    def test_world_axis_with_rotated_center(self):
        # Если центр повернуть на π/2 вокруг Y, локальная ось Z станет мировой X.
        center = Coords(position=[0.0, 0.0, 0.0], rpy=[0.0, np.pi / 2, 0.0])
        capsule = Capsule(center, radius=0.3, height=2.0,
                          local_axis=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(capsule.world_axis, [1.0, 0.0, 0.0], atol=1e-12)

    def test_distance_to_point_on_lateral_surface(self):
        # Точка на боковой поверхности цилиндра — distance ≈ 0.
        p = np.array([self.capsule.radius, 0.0, 0.0])
        d = self.capsule.distance_to_point(p)
        self.assertAlmostEqual(d, 0.0, delta=1e-9)

    def test_distance_to_point_on_axis_inside_is_negative_radius(self):
        # Точка на оси внутри высоты капсулы — distance = -radius.
        d = self.capsule.distance_to_point(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(d, -self.capsule.radius, delta=1e-9)

    def test_distance_to_point_above_hemispherical_cap(self):
        # Точка над верхним полушарием на 1.0 от его центра — distance = 1.0 - r.
        # Верхний центр полушария: (0, 0, height/2) = (0, 0, 1).
        p = np.array([0.0, 0.0, 1.0 + 1.0])
        d = self.capsule.distance_to_point(p)
        self.assertAlmostEqual(d, 1.0 - self.capsule.radius, delta=1e-9)

    def test_dist_to_me_parallel_segment(self):
        # Отрезок параллельно оси капсулы на расстоянии 1.5 от оси — d = 1.5 - r.
        seg_1 = Coords(position=[1.5, 0.0, -0.5])
        seg_2 = Coords(position=[1.5, 0.0,  0.5])
        d = self.capsule.dist_to_me(seg_1, seg_2)
        self.assertAlmostEqual(d, 1.5 - self.capsule.radius, delta=1e-9)

    def test_dist_to_me_segment_through_axis_returns_zero(self):
        # Отрезок пересекает ось капсулы — зазор отрицательный, max'нулся в 0.
        seg_1 = Coords(position=[-1.0, 0.0, 0.0])
        seg_2 = Coords(position=[1.0, 0.0, 0.0])
        d = self.capsule.dist_to_me(seg_1, seg_2)
        self.assertEqual(d, 0.0)


if __name__ == "__main__":
    unittest.main()
