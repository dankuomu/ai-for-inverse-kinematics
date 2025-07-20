import numpy as np
from utils import Coords

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