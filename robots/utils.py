import numpy as np


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