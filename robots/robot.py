import numpy as np
from robots.utils import Coords
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import time

class Robot:
    def __init__(self, dh_parameters):
        """
        Инициализация робота с параметрами D-H.

        :param dh_parameters: Список кортежей с параметрами D-H для каждого звена в формате (a, alpha, d)
        """
        self.dh_params = dh_parameters
        self.ik_solver = None

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

    @property
    def ik_solver(self):
        """
        Вернуть текущий IK-решатель как экземпляр.
        Если хранится класс — создаём экземпляр (и кешируем).
        """
        if isinstance(self._ik_solver, type):
            self._ik_solver = self._ik_solver(self)
        return self._ik_solver

    @ik_solver.setter
    def ik_solver(self, value):
        self._ik_solver = value

    def set_inverse(self, ik_class, **kwargs):
        """
        Зарегистрировать решатель обратной задачи.
        Параметр ik может быть:
         - класс (наследник InverseKinematics) — тогда при solve будет создаваться экземпляр
         - или экземпляр решателя (уже проинициализированный).
        """
        self.ik_solver = ik_class(self, **kwargs)

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

    def visualize(self, angles, target=None, ax=None, show=True):
        """
        Визуализация робота, целевой точки и систем координат.

        :param angles: Список углов суставов
        :param target_point: Опциональная целевая точка (array-like форма (3,))
        :param ax: Существующие оси matplotlib (если None, создаются новые)
        :param show: Флаг немедленного показа графика
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        target_point = target.pos
        target_orientation = target.rot_matrix

        # Получаем позиции всех суставов
        joint_positions = self.get_joint_positions(angles)

        # Получаем текущую позицию и ориентацию энд-эффектора
        end_effector_coords = self.forward_kinematics(angles)
        end_effector_pos = end_effector_coords.pos
        end_effector_orientation = end_effector_coords.rot_matrix

        # Рисуем звенья робота
        ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                'o-', linewidth=2, markersize=5, label='Робот', color='orange')

        # Система координат энд-эффектора (сплошные линии)
        axis_length = 0.1  # Длина осей
        colors = ['red', 'green', 'blue']  # X, Y, Z оси
        axis_labels = ['X', 'Y', 'Z']

        for i, color in enumerate(colors):
            axis_vector = end_effector_orientation[:, i] * axis_length
            axis_end = end_effector_pos + axis_vector
            ax.plot([end_effector_pos[0], axis_end[0]],
                    [end_effector_pos[1], axis_end[1]],
                    [end_effector_pos[2], axis_end[2]],
                    color=color, linewidth=2,
                    label=f'Ось {axis_labels[i]} (Робота)')

        # Рисуем целевую точку если предоставлена
        if target_point is not None:
            target_point = np.array(target_point)
            ax.scatter(*target_point, color='black', s=10, label='Целевая позиция')

            if target_orientation is not None:
                if hasattr(target_orientation, 'rot_matrix'):
                    target_rot = target_orientation.rot_matrix
                else:
                    target_rot = np.array(target_orientation)

                for i, color in enumerate(colors):
                    axis_vector = target_rot[:, i] * axis_length
                    axis_end = target_point + axis_vector
                    ax.plot([target_point[0], axis_end[0]],
                            [target_point[1], axis_end[1]],
                            [target_point[2], axis_end[2]],
                            color=color, linewidth=2, linestyle='--',
                            label=f'Ось {axis_labels[i]} (Цели)')

            # Показываем ошибку позиционирования и ориентации
            error_pos = np.linalg.norm(end_effector_pos - target_point)

            error_text = f'Ошибка позиции: {error_pos:.9f}'
            if target_orientation is not None:
                if hasattr(target_orientation, 'rot_matrix'):
                    target_rot = target_orientation.rot_matrix
                else:
                    target_rot = np.array(target_orientation)

                orientation_error = np.arccos(
                    np.clip(0.5 * (np.trace(end_effector_orientation.T @ target_rot) - 1), -1, 1))
                error_text += f'\nОшибка ориентации: {orientation_error:.9f} рад'

            ax.text2D(0.05, 0.95, error_text, transform=ax.transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Настройка графика
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Кинематическая схема робота с системами координат')
        ax.legend()
        ax.grid(True)

        # Автоматическое масштабирование с учетом систем координат
        all_points = joint_positions.copy()
        if target_point is not None:
            all_points = np.vstack([all_points, target_point.reshape(1, -1)])
            # Добавляем точки осей для правильного масштабирования
            for i in range(3):
                axis_vector = end_effector_orientation[:, i] * axis_length
                all_points = np.vstack([all_points, (end_effector_pos + axis_vector).reshape(1, -1)])
                if target_orientation is not None:
                    axis_vector_target = target_rot[:, i] * axis_length
                    all_points = np.vstack([all_points, (target_point + axis_vector_target).reshape(1, -1)])

        max_range = max([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]) * 1.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if show:
            plt.show()

        return ax

    def op_solve(self,
                 initial_angles: np.ndarray,
                 target: Coords,
                 max_iter: int = 40,
                 pos_tol: float = 1e-5,
                 rot_tol: float = 1e-4,
                 pos_weight: float = 1.0,
                 rot_weight: float = 1.0,
                 eps_jac: float = 1e-6,
                 max_step_norm: float = 0.5,
                 lambda0: float = 1e-3,
                 lambda_increase: float = 10.0,
                 lambda_decrease: float = 0.1):
        """
        Уточнение решения методом Levenberg-Marquardt (дроссельный Gauss-Newton) с проверкой улучшения.
        Возвращает: (angles, metrics)
        Параметры:
          - pos_weight, rot_weight: веса для комбинированной стоимости
          - eps_jac: дельта для численного якобиана
          - max_step_norm: максимально допустимая норма одного шага (предотвращает большие "прыжки")
          - lambda0: начальное демпфирование
          - lambda_increase / lambda_decrease: факторы адаптации lambda
        """
        import time
        start_time = time.time()

        angles = initial_angles.astype(float).copy()
        n = len(angles)
        target_pos = np.asarray(target.pos)
        target_rot = np.asarray(target.rot_matrix)

        def rot_angle_from_R(R):
            # угол по матрице R (trace)
            val = 0.5 * (np.trace(R) - 1.0)
            return np.arccos(np.clip(val, -1.0, 1.0))

        def orientation_error_vector(R_target, R_curr):
            # R_err = R_target * R_curr^T
            R_err = R_target @ R_curr.T
            # осевой вектор (приближённый) = 0.5*(R_err - R_err^T) as vector part
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ]) * 0.5
            angle = rot_angle_from_R(R_err)
            norm_axis = np.linalg.norm(axis)
            if norm_axis > 1e-12:
                return axis / norm_axis * angle
            else:
                return np.zeros(3)

        def cost_from(angles_vec):
            fk = self.forward_kinematics(angles_vec)
            p = fk.pos
            R = fk.rot_matrix
            pos_err = np.linalg.norm(target_pos - p)
            rot_err = np.arccos(np.clip(0.5 * (np.trace(R.T @ target_rot) - 1), -1, 1))
            return pos_weight * pos_err + rot_weight * rot_err, pos_err, rot_err

        def numerical_jacobian(angles_vec, eps=eps_jac):
            J = np.zeros((6, n))
            fk0 = self.forward_kinematics(angles_vec)
            p0 = fk0.pos
            R0 = fk0.rot_matrix

            for i in range(n):
                a2 = angles_vec.copy()
                a2[i] += eps
                fk = self.forward_kinematics(a2)
                dp = (fk.pos - p0) / eps

                # ориентационная часть: (fk.R * R0^T) -> axis-angle approx
                dR = fk.rot_matrix @ R0.T
                axis = np.array([
                    dR[2, 1] - dR[1, 2],
                    dR[0, 2] - dR[2, 0],
                    dR[1, 0] - dR[0, 1]
                ]) * 0.5
                # делим на eps, даём аппроксимацию d(angle*axis)/dθ
                J[:, i] = np.concatenate([dp, axis / eps])
            return J

        # Начальная стоимость
        best_cost, best_pos_err, best_rot_err = cost_from(angles)
        best_angles = angles.copy()

        lam = lambda0

        iter_count = 0
        for it in range(max_iter):
            iter_count = it
            fk = self.forward_kinematics(angles)
            p = fk.pos
            R = fk.rot_matrix

            pos_err_vec = (target_pos - p)
            pos_err = np.linalg.norm(pos_err_vec)
            rot_err_vec = orientation_error_vector(target_rot, R)
            rot_err = np.linalg.norm(rot_err_vec)

            # Проверка выхода
            if pos_err < pos_tol and rot_err < rot_tol:
                break

            # Вектор ошибки (6,)
            err_vec = np.concatenate([pos_err_vec, rot_err_vec])

            # Якобиан (6 x n)
            J = numerical_jacobian(angles)

            # Составим взвешенную систему: масштабируем строки чтобы учесть веса
            W = np.ones(6)
            W[:3] *= pos_weight
            W[3:] *= rot_weight
            W_mat = np.sqrt(np.diag(W))
            Jw = W_mat @ J
            errw = W_mat @ err_vec

            # LM: (J^T J + lam I) delta = J^T err
            A = Jw.T @ Jw
            g = Jw.T @ errw

            # Добавляем демпфирование к диагонали
            A_damped = A + lam * np.eye(A.shape[0])

            # Решаем (с SVD fallback если плохо обусловлено)
            try:
                delta = np.linalg.solve(A_damped, g)
            except np.linalg.LinAlgError:
                # fallback: псевдо-инверс
                delta = np.linalg.pinv(A_damped) @ g

            # Ограничим шаг (чтобы не перепрыгивать)
            step_norm = np.linalg.norm(delta)
            if step_norm > max_step_norm:
                delta = delta / step_norm * max_step_norm

            candidate = angles + delta

            cand_cost, cand_pos_err, cand_rot_err = cost_from(candidate)

            # после J = numerical_jacobian(angles)
            sv = np.linalg.svd(J, compute_uv=False)
            cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
            print(
                f"[NR debug] it={it}, ||err||={np.linalg.norm(err_vec):.6g}, sv0={sv[0]:.3g}, sv_last={sv[-1]:.3g}, cond={cond:.3g}")

            # перед применением шага
            print(
                f"[NR debug] step_norm={np.linalg.norm(delta):.6g}, lambda={lam:.3g}, cand_cost={cand_cost:.6g}, best_cost={best_cost:.6g}")

            # Если улучшилось — принимаем и уменьшаем lam, иначе увеличиваем lam и не принимаем
            if cand_cost < best_cost - 1e-12:
                angles = candidate
                best_cost = cand_cost
                best_angles = angles.copy()
                best_pos_err = cand_pos_err
                best_rot_err = cand_rot_err
                lam = max(lam * lambda_decrease, 1e-12)
            else:
                # не улучшилось — увеличиваем демпфирование и повторяем попытку
                lam *= lambda_increase
                # если lam слишком велик, остановиться — дальнейшие шаги бессмысленны
                if lam > 1e12:
                    break

        # финальная FK на best_angles
        final_fk = self.forward_kinematics(best_angles)
        final_pos = final_fk.pos
        final_rot = final_fk.rot_matrix

        final_pos_err = np.linalg.norm(final_pos - target_pos)
        final_rot_err = np.arccos(
            np.clip(0.5 * (np.trace(final_rot.T @ target_rot) - 1), -1, 1)
        )

        total_time = time.time() - start_time

        metrics = {
            "total_time": total_time,
            "best_fitness": -(pos_weight * final_pos_err + rot_weight * final_rot_err),  # отрицательная стоимость
            "target_position": target_pos,
            "achieved_position": final_pos,
            "position_error": final_pos_err,
            "target_orientation": target_rot,
            "achieved_orientation": final_rot,
            "orientation_error": final_rot_err,
            "iterations": iter_count + 1,
            "method": "Levenberg-Marquardt (refinement)"
        }

        return best_angles, metrics