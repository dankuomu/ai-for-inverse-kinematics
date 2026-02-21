import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Coords:
    def __init__(self, position=None, rotation_matrix=None, rpy=None, quaternion=None, axis_angle=None):
        """
        Инициализация объекта координат.

        Можно задать одним из способов:
        - position + rotation_matrix
        - position + rpy (roll, pitch, yaw)
        - position + quaternion (w, x, y, z)
        - position + axis_angle (vx, vy, vz, angle)
        - только position (поворот - единичная матрица)
        - только поворот (позиция - нулевая)

        :param position: Вектор позиции (x, y, z)
        :param rotation_matrix: Матрица поворота 3x3
        :param rpy: Углы Эйлера (roll, pitch, yaw) в радианах
        :param quaternion: Кватернион (w, x, y, z)
        :param axis_angle: Ось и угол вращения (vx, vy, vz, angle)
        """
        # Инициализация позиции
        self._pos = np.array([0.0, 0.0, 0.0]) if position is None else np.array(position, dtype=float)

        # Подсчет количества переданных способов задания поворота
        rotation_params = [rotation_matrix, rpy, quaternion, axis_angle]
        non_none_params = sum(1 for param in rotation_params if param is not None)

        if non_none_params > 1:
            raise ValueError("Можно задать только один способ определения поворота")

        # Инициализация поворота
        if rotation_matrix is not None:
            self._init_from_rotation_matrix(rotation_matrix)
        elif rpy is not None:
            self._init_from_rpy(rpy)
        elif quaternion is not None:
            self._init_from_quaternion(quaternion)
        elif axis_angle is not None:
            self._init_from_axis_angle(axis_angle)
        else:
            self._rot = np.eye(3)

    def _init_from_rotation_matrix(self, rotation_matrix):
        """Инициализация из матрицы поворота"""
        rot = np.array(rotation_matrix, dtype=float)
        if rot.shape != (3, 3):
            raise ValueError("Матрица поворота должна быть 3x3")
        self._rot = rot

    def _init_from_rpy(self, rpy):
        """Инициализация из углов Эйлера (roll, pitch, yaw)"""
        rpy = np.array(rpy, dtype=float)
        if rpy.shape != (3,):
            raise ValueError("Углы Эйлера должны быть вектором из 3 элементов")

        roll, pitch, yaw = rpy

        # Матрицы поворота вокруг осей
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Композиция поворотов: Rz * Ry * Rx
        self._rot = Rz @ Ry @ Rx

    def _init_from_quaternion(self, quaternion):
        """Инициализация из кватерниона (w, x, y, z)"""
        q = np.array(quaternion, dtype=float)
        if q.shape != (4,):
            raise ValueError("Кватернион должен быть вектором из 4 элементов")

        w, x, y, z = q
        # Нормализация кватерниона
        norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Преобразование кватерниона в матрицу поворота
        self._rot = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

    def _init_from_axis_angle(self, axis_angle):
        """Инициализация из оси и угла вращения (vx, vy, vz, angle)"""
        axis_angle = np.array(axis_angle, dtype=float)
        if axis_angle.shape != (4,):
            raise ValueError("Ось и угол должны быть вектором из 4 элементов")

        vx, vy, vz, angle = axis_angle

        # Нормализация оси
        axis_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        if axis_norm < 1e-10:
            self._rot = np.eye(3)
            return

        vx, vy, vz = vx / axis_norm, vy / axis_norm, vz / axis_norm

        # Формула Родригеса
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        one_minus_cos = 1 - cos_a

        self._rot = np.array([
            [cos_a + vx * vx * one_minus_cos, vx * vy * one_minus_cos - vz * sin_a,
             vx * vz * one_minus_cos + vy * sin_a],
            [vx * vy * one_minus_cos + vz * sin_a, cos_a + vy * vy * one_minus_cos,
             vy * vz * one_minus_cos - vx * sin_a],
            [vx * vz * one_minus_cos - vy * sin_a, vy * vz * one_minus_cos + vx * sin_a,
             cos_a + vz * vz * one_minus_cos]
        ])

    @classmethod
    def from_rotation_matrix(cls, position, rotation_matrix):
        """Альтернативный конструктор из позиции и матрицы поворота"""
        return cls(position=position, rotation_matrix=rotation_matrix)

    @classmethod
    def from_rpy(cls, position, rpy):
        """Альтернативный конструктор из позиции и углов Эйлера"""
        return cls(position=position, rpy=rpy)

    @classmethod
    def from_quaternion(cls, position, quaternion):
        """Альтернативный конструктор из позиции и кватерниона"""
        return cls(position=position, quaternion=quaternion)

    @classmethod
    def from_axis_angle(cls, position, axis_angle):
        """Альтернативный конструктор из позиции и оси-угла"""
        return cls(position=position, axis_angle=axis_angle)

    @property
    def pos(self):
        """Возвращает позицию (x, y, z)"""
        return self._pos.copy()

    @property
    def rot_matrix(self):
        """Возвращает матрицу поворота 3x3"""
        return self._rot.copy()

    @property
    def RPY(self):
        """Вычисляет углы Эйлера (roll, pitch, yaw) в радианах"""
        sy = np.sqrt(self._rot[0, 0] ** 2 + self._rot[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(self._rot[2, 1], self._rot[2, 2])
            pitch = np.arctan2(-self._rot[2, 0], sy)
            yaw = np.arctan2(self._rot[1, 0], self._rot[0, 0])
        else:
            roll = np.arctan2(-self._rot[1, 2], self._rot[1, 1])
            pitch = np.arctan2(-self._rot[2, 0], sy)
            yaw = 0.0

        return np.array([roll, pitch, yaw])

    @property
    def quaternion(self):
        """Вычисляет кватернион ориентации (w, x, y, z)"""
        trace = np.trace(self._rot)

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (self._rot[2, 1] - self._rot[1, 2]) / S
            y = (self._rot[0, 2] - self._rot[2, 0]) / S
            z = (self._rot[1, 0] - self._rot[0, 1]) / S
        elif (self._rot[0, 0] > self._rot[1, 1]) and (self._rot[0, 0] > self._rot[2, 2]):
            S = np.sqrt(1.0 + self._rot[0, 0] - self._rot[1, 1] - self._rot[2, 2]) * 2
            w = (self._rot[2, 1] - self._rot[1, 2]) / S
            x = 0.25 * S
            y = (self._rot[0, 1] + self._rot[1, 0]) / S
            z = (self._rot[0, 2] + self._rot[2, 0]) / S
        elif self._rot[1, 1] > self._rot[2, 2]:
            S = np.sqrt(1.0 + self._rot[1, 1] - self._rot[0, 0] - self._rot[2, 2]) * 2
            w = (self._rot[0, 2] - self._rot[2, 0]) / S
            x = (self._rot[0, 1] + self._rot[1, 0]) / S
            y = 0.25 * S
            z = (self._rot[1, 2] + self._rot[2, 1]) / S
        else:
            S = np.sqrt(1.0 + self._rot[2, 2] - self._rot[0, 0] - self._rot[1, 1]) * 2
            w = (self._rot[1, 0] - self._rot[0, 1]) / S
            x = (self._rot[0, 2] + self._rot[2, 0]) / S
            y = (self._rot[1, 2] + self._rot[2, 1]) / S
            z = 0.25 * S

        # Нормализация
        norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        return np.array([w / norm, x / norm, y / norm, z / norm])

    @property
    def axis_angle(self):
        """Возвращает ось и угол вращения (vx, vy, vz, angle)"""
        angle = np.arccos(np.clip((np.trace(self._rot) - 1) / 2, -1, 1))

        if angle < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis = np.array([
            self._rot[2, 1] - self._rot[1, 2],
            self._rot[0, 2] - self._rot[2, 0],
            self._rot[1, 0] - self._rot[0, 1]
        ])
        axis /= np.linalg.norm(axis)

        return np.array([axis[0], axis[1], axis[2], angle])

    def __str__(self):
        """Человекочитаемое представление"""
        return f"Position: {self._pos}\nRotation:\n{self._rot}"

    def transform_point(self, point):
        """Преобразует точку из локальной системы координат в глобальную"""
        return self._rot @ np.array(point) + self._pos

    def inverse(self):
        """Возвращает обратное преобразование"""
        inv_rot = self._rot.T
        inv_pos = -inv_rot @ self._pos
        return Coords(inv_pos, inv_rot)


class Obstacle:
    """Базовый класс для всех препятствий"""

    @staticmethod
    def _dist_seg_to_seg(a1: np.ndarray, a2: np.ndarray,
                         b1: np.ndarray, b2: np.ndarray) -> float:
        """
        Минимальное евклидово расстояние между двумя отрезками в трёхмерном пространстве.
        Отрезки заданы парами точек: [a1, a2] и [b1, b2].
        Возвращает неотрицательное число.
        """
        u = a2 - a1
        v = b2 - b1
        w = a1 - b1

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        D = a * c - b * b

        if D >= 1e-9:
            s = (b * e - c * d) / D
            t = (a * e - b * d) / D

            if s < 0:
                s = 0
                t = e / c if c != 0 else 0
                t = np.clip(t, 0, 1)
            elif s > 1:
                s = 1
                t = (e + b) / c if c != 0 else 0
                t = np.clip(t, 0, 1)

            if t < 0:
                t = 0
                s = -d / a if a != 0 else 0
                s = np.clip(s, 0, 1)
            elif t > 1:
                t = 1
                s = (b - d) / a if a != 0 else 0
                s = np.clip(s, 0, 1)

            point_a = a1 + s * u
            point_b = b1 + t * v
            return np.linalg.norm(point_a - point_b)

        else:
            if a > 0:
                direction = u / np.sqrt(a)
            elif c > 0:
                direction = v / np.sqrt(c)
            else:
                return np.linalg.norm(a1 - b1)

            proj_a1 = np.dot(a1, direction)
            proj_a2 = np.dot(a2, direction)
            proj_b1 = np.dot(b1, direction)
            proj_b2 = np.dot(b2, direction)

            minA = min(proj_a1, proj_a2)
            maxA = max(proj_a1, proj_a2)
            minB = min(proj_b1, proj_b2)
            maxB = max(proj_b1, proj_b2)

            w_vec = a1 - b1
            h_sq = np.dot(w_vec, w_vec) - (np.dot(w_vec, direction)) ** 2
            h = np.sqrt(max(0.0, h_sq))  # защита от малых отрицательных из-за погрешностей

            if max(minA, minB) <= min(maxA, maxB):
                return h
            else:
                if maxA < minB:
                    gap = minB - maxA
                else:  # maxB < minA
                    gap = minA - maxB
                return np.sqrt(h * h + gap * gap)

    def dist_to_me(self, seg_1: 'Coords', seg_2: 'Coords') -> float:
        """
        Возвращает минимальное расстояние от препятствия до отрезка,
        заданного двумя точками в виде объектов Coords.
        """
        raise NotImplementedError("Метод должен быть реализован в наследниках")

    def visualize(self, ax: Axes3D, color='r', alpha=0.3):
        """Визуализирует препятствие в 3D"""
        raise NotImplementedError("Метод должен быть реализован в наследниках")


class Sphere(Obstacle):
    def __init__(self, center: Coords, radius: float):
        self.center = center
        self.radius = radius

    def dist_to_me(self, seg_1: Coords, seg_2: Coords) -> float:
        d_center = Obstacle._dist_seg_to_seg(seg_1.pos, seg_2.pos,
                                             self.center.pos, self.center.pos)
        return max(0.0, d_center - self.radius)

    def visualize(self, ax: Axes3D, color='r', alpha=0.3):
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = self.center.pos[0] + self.radius * np.cos(u) * np.sin(v)
        y = self.center.pos[1] + self.radius * np.sin(u) * np.sin(v)
        z = self.center.pos[2] + self.radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=alpha)


class Capsule(Obstacle):
    def __init__(self, center: Coords, radius: float, height: float,
                 local_axis: np.ndarray = np.array([0, 0, 1])):
        self.center = center
        self.radius = radius
        self.height = height
        self.local_axis = local_axis / np.linalg.norm(local_axis)

    @property
    def world_axis(self) -> np.ndarray:
        """Ось капсулы в мировых координатах (с учётом поворота центра)."""
        return self.center.rot @ self.local_axis

    def dist_to_me(self, seg_1: Coords, seg_2: Coords) -> float:
        axis = self.world_axis
        half_h = self.height / 2
        c1 = self.center.pos - half_h * axis
        c2 = self.center.pos + half_h * axis
        d_line = Obstacle._dist_seg_to_seg(seg_1.pos, seg_2.pos, c1, c2)
        return max(0.0, d_line - self.radius)

    def visualize(self, ax: Axes3D, color='r', alpha=0.3, resolution=30):
        def rotation_matrix_from_z_to_axis(axis):
            z = np.array([0, 0, 1])
            a = axis / np.linalg.norm(axis)
            v = np.cross(z, a)
            c = np.dot(z, a)
            s = np.linalg.norm(v)
            if s < 1e-9:
                return np.eye(3)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

        R = rotation_matrix_from_z_to_axis(self.world_axis)

        z = np.linspace(-self.height/2, self.height/2, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = self.radius * np.cos(theta_grid)
        y_grid = self.radius * np.sin(theta_grid)

        xyz = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        xyz_rot = R @ xyz
        X = xyz_rot[0, :].reshape(x_grid.shape) + self.center.pos[0]
        Y = xyz_rot[1, :].reshape(y_grid.shape) + self.center.pos[1]
        Z = xyz_rot[2, :].reshape(z_grid.shape) + self.center.pos[2]
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

        phi = np.linspace(0, np.pi, resolution)
        theta_s = np.linspace(0, 2*np.pi, resolution)
        phi_grid, theta_grid = np.meshgrid(phi, theta_s)

        x_s = self.radius * np.sin(phi_grid) * np.cos(theta_grid)
        y_s = self.radius * np.sin(phi_grid) * np.sin(theta_grid)
        z_s_low = -self.height/2 + self.radius * np.cos(phi_grid)
        z_s_high = self.height/2 + self.radius * np.cos(phi_grid)

        for z_s in (z_s_low, z_s_high):
            xyz_s = np.vstack([x_s.ravel(), y_s.ravel(), z_s.ravel()])
            xyz_s_rot = R @ xyz_s
            X = xyz_s_rot[0, :].reshape(x_s.shape) + self.center.pos[0]
            Y = xyz_s_rot[1, :].reshape(y_s.shape) + self.center.pos[1]
            Z = xyz_s_rot[2, :].reshape(z_s.shape) + self.center.pos[2]
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha)