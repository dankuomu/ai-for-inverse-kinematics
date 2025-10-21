import numpy as np
from robots.utils import Coords
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, dh_parameters):
        """
        Инициализация робота с параметрами D-H.

        :param dh_parameters: Список кортежей с параметрами D-H для каждого звена в формате (a, alpha, d)
        """
        self.dh_params = dh_parameters

    def forward_kinematics(self, angles):
        """
        Вычисляет прямую кинематику для заданных углов суставов.

        :param angles: Список углов вращения для каждого сустава (в радианах)
        :return: Объект Coords с позицией и ориентацией
        """
        if len(angles) != len(self.dh_params):
            raise ValueError("Количество углов должно совпадать с количеством звеньев")

        # Начальная матрица преобразования (единичная матрица 4x4)
        T = np.eye(4)

        for i, angle in enumerate(angles):
            a, alpha, d = self.dh_params[i]
            theta = angle

            # Вычисляем матрицу преобразования для текущего звена
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            Ti = np.array([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

            # Накопление общего преобразования
            T = T @ Ti

        # Извлекаем позицию и матрицу поворота
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]

        return Coords(position, rotation_matrix)

    def set_inverse(self, ik: object):
        """
        Зарегистрировать решатель обратной задачи.
        Параметр ik может быть:
         - класс (наследник InverseKinematics) — тогда при solve будет создаваться экземпляр
         - или экземпляр решателя (уже проинициализированный).
        """
        self._ik_solver = ik

    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Решить обратную задачу для заданной цели.
        Если _ik_solver — класс, создаём экземпляр: solver = _ik_solver(self)
        Если _ik_solver — экземпляр, используем его (и обновляем robot внутри, если нужно).
        kwargs пробрасываются в solver.solve (например: generations=100).
        """
        if self._ik_solver is None:
            raise RuntimeError("Inverse solver is not set. Call Robot.set_inverse(...) first.")

        if isinstance(self._ik_solver, type):
            solver = self._ik_solver(self)
        else:
            solver = self._ik_solver
            if getattr(solver, 'robot', None) is not self:
                solver.robot = self

        return solver.solve(target, **kwargs)

    def get_joint_positions(self, angles):
        """
        Вычисляет позиции всех суставов включая начало и конечный эффектор.

        :param angles: Список углов вращения для каждого сустава
        :return: Массив позиций суставов в форме (N+1, 3)
        """
        positions = [np.zeros(3)]  # Начальная позиция (база)
        T = np.eye(4)

        for i, angle in enumerate(angles):
            a, alpha, d = self.dh_params[i]
            theta = angle

            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            Ti = np.array([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

            T = T @ Ti
            positions.append(T[:3, 3])

        return np.array(positions)

    def visualize(self, angles, target_point=None, ax=None, show=True):
        """
        Визуализация робота и опциональной целевой точки в 3D-пространстве.

        :param angles: Список углов суставов
        :param target_point: Опциональная целевая точка (array-like форма (3,))
        :param ax: Существующие оси matplotlib (если None, создаются новые)
        :param show: Флаг немедленного показа графика
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Получаем позиции всех суставов
        joint_positions = self.get_joint_positions(angles)

        # Рисуем звенья робота
        ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                'o-', linewidth=3, markersize=8, label='Робот')

        # Рисуем целевую точку если предоставлена
        if target_point is not None:
            target_point = np.array(target_point)
            ax.scatter(*target_point, color='red', s=100, label='Целевая точка')

            # Показываем ошибку позиционирования
            end_effector_pos = joint_positions[-1]
            error = np.linalg.norm(end_effector_pos - target_point)
            ax.text2D(0.05, 0.95, f'Ошибка: {error:.4f}', transform=ax.transAxes)

        # Настройка графика
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Кинематическая схема робота')
        ax.legend()
        ax.grid(True)

        # Автоматическое масштабирование
        max_range = max([
            joint_positions[:, 0].max() - joint_positions[:, 0].min(),
            joint_positions[:, 1].max() - joint_positions[:, 1].min(),
            joint_positions[:, 2].max() - joint_positions[:, 2].min()
        ]) * 0.5

        mid_x = (joint_positions[:, 0].max() + joint_positions[:, 0].min()) * 0.5
        mid_y = (joint_positions[:, 1].max() + joint_positions[:, 1].min()) * 0.5
        mid_z = (joint_positions[:, 2].max() + joint_positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if show:
            plt.show()

        return ax

    @property
    def ik_solver(self):
        """
        Вернуть текущий IK-решатель как экземпляр.
        Если хранится класс — создаём экземпляр (и кешируем).
        """
        if isinstance(self._ik_solver, type):
            self._ik_solver = self._ik_solver(self)
        return self._ik_solver