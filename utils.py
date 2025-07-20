import numpy as np


class Coords:
    def __init__(self, position=None, rotation_matrix=None):
        """
        Инициализация объекта координат.

        :param position: Вектор позиции (x, y, z)
        :param rotation_matrix: Матрица поворота 3x3 (построчно)
        """
        self._pos = np.array([0.0, 0.0, 0.0]) if position is None else np.array(position, dtype=float)

        if rotation_matrix is None:
            self._rot = np.eye(3)
        else:
            # Проверяем и сохраняем матрицу поворота
            rot = np.array(rotation_matrix, dtype=float)
            if rot.shape != (3, 3):
                raise ValueError("Матрица поворота должна быть 3x3")
            self._rot = rot

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
