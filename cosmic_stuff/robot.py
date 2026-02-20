import numpy as np
import time
import matplotlib.pyplot as plt

from robots.utils import Coords
from IK.genetic import GeneticIK


def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def T(dx=0, dy=0, dz=0):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])


class FloatingTranslationRobot:
    def __init__(self, base_mass=3.0, link_lengths=None, link_masses=None, base_size=1.0):
        self.base_mass = base_mass
        self.base_size = base_size

        if link_lengths is None:
            link_lengths = [0.5, 0.5, 0.5]
        if link_masses is None:
            link_masses = [1.0, 1.0, 1.0]

        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.total_mass = self.base_mass + sum(link_masses)

        self.dh_params = []
        self.num_joints = 6

        self.base_orientation = np.eye(3)

        self.base_attach_point = np.array([base_size / 2, base_size / 2, base_size / 2])

        self.base_position = np.zeros(3)

    def _forward_kinematics_relative(self, q):

        theta1, theta2, theta3 = q

        T0 = T(dx=self.base_size / 2, dy=self.base_size / 2, dz=self.base_size / 2)

        T1 = T0 @ Rz(theta1) @ T(dx=self.link_lengths[0])

        T2 = T1 @ Ry(theta2) @ T(dx=self.link_lengths[1])

        T3 = T2 @ Rz(theta3) @ T(dx=self.link_lengths[2])

        positions = []
        positions.append(T0[:3, 3])
        positions.append(T1[:3, 3])
        positions.append(T2[:3, 3])
        positions.append(T3[:3, 3])

        end_effector_matrix = T3

        return np.array(positions), end_effector_matrix

    def compute_center_of_mass(self, full_state):
        base_position = np.array(full_state[:3])
        angles = np.array(full_state[3:])

        positions_relative, _ = self._forward_kinematics_relative(angles)

        com_links_relative = []
        for i in range(3):
            com_link = (positions_relative[i] + positions_relative[i + 1]) / 2
            com_links_relative.append(com_link)

        com_links_absolute = [com + base_position for com in com_links_relative]

        com_base_absolute = base_position

        com_total = np.zeros(3)
        com_total += self.base_mass * com_base_absolute

        for i in range(3):
            com_total += self.link_masses[i] * com_links_absolute[i]

        com_total /= self.total_mass

        return com_total

    def forward_kinematics(self, full_state):
        base_position = np.array(full_state[:3])
        angles = np.array(full_state[3:])

        self.base_position = base_position

        positions_relative, end_effector_matrix = self._forward_kinematics_relative(angles)

        end_position_absolute = end_effector_matrix[:3, 3] + base_position

        end_rotation_absolute = self.base_orientation @ end_effector_matrix[:3, :3]

        end_effector_coords = Coords(end_position_absolute, end_rotation_absolute)

        return end_effector_coords

    def get_joint_positions(self, full_state):
        base_position = np.array(full_state[:3])
        angles = np.array(full_state[3:])

        positions_relative, _ = self._forward_kinematics_relative(angles)
        positions_absolute = positions_relative + base_position

        return positions_absolute

    def visualize(self, full_state, target=None, ax=None, show=True):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False

        base_position = np.array(full_state[:3])
        angles = np.array(full_state[3:])

        positions_relative, _ = self._forward_kinematics_relative(angles)
        positions_absolute = positions_relative + base_position

        end_effector = self.forward_kinematics(full_state)
        com_total = self.compute_center_of_mass(full_state)

        s = self.base_size
        half_s = s / 2

        vertices_relative = np.array([
            [-half_s, -half_s, -half_s],
            [half_s, -half_s, -half_s],
            [half_s, half_s, -half_s],
            [-half_s, half_s, -half_s],
            [-half_s, -half_s, half_s],
            [half_s, -half_s, half_s],
            [half_s, half_s, half_s],
            [-half_s, half_s, half_s]
        ])

        vertices_absolute = vertices_relative + base_position

        faces = [
            [vertices_absolute[0], vertices_absolute[1], vertices_absolute[2], vertices_absolute[3]],
            [vertices_absolute[4], vertices_absolute[5], vertices_absolute[6], vertices_absolute[7]],
            [vertices_absolute[0], vertices_absolute[1], vertices_absolute[5], vertices_absolute[4]],
            [vertices_absolute[2], vertices_absolute[3], vertices_absolute[7], vertices_absolute[6]],
            [vertices_absolute[1], vertices_absolute[2], vertices_absolute[6], vertices_absolute[5]],
            [vertices_absolute[0], vertices_absolute[3], vertices_absolute[7], vertices_absolute[4]]
        ]

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        cube = Poly3DCollection(faces, alpha=0.3, facecolor='lightgray',
                                edgecolor='black', linewidth=1.5)
        ax.add_collection3d(cube)

        for i in range(len(positions_absolute) - 1):
            ax.plot([positions_absolute[i][0], positions_absolute[i + 1][0]],
                    [positions_absolute[i][1], positions_absolute[i + 1][1]],
                    [positions_absolute[i][2], positions_absolute[i + 1][2]],
                    'o-', linewidth=2.5, markersize=6, color='blue')

        ax.scatter(positions_absolute[:, 0], positions_absolute[:, 1], positions_absolute[:, 2],
                   s=40, color='darkblue', zorder=5)

        ax.scatter(com_total[0], com_total[1], com_total[2],
                   s=100, color='orange', label='Center of Mass', zorder=7)

        axis_length = 0.1
        axes_local = np.eye(3) * axis_length
        axes_world = end_effector.rot_matrix @ axes_local
        origin = end_effector.pos

        ax.quiver(origin[0], origin[1], origin[2],
                  axes_world[0, 0], axes_world[1, 0], axes_world[2, 0],
                  color='red', arrow_length_ratio=0.15, linewidth=1.5)
        ax.quiver(origin[0], origin[1], origin[2],
                  axes_world[0, 1], axes_world[1, 1], axes_world[2, 1],
                  color='green', arrow_length_ratio=0.15, linewidth=1.5)
        ax.quiver(origin[0], origin[1], origin[2],
                  axes_world[0, 2], axes_world[1, 2], axes_world[2, 2],
                  color='blue', arrow_length_ratio=0.15, linewidth=1.5)

        ax.scatter(0, 0, 0, s=50, color='black', marker='x', label='(0,0,0)')

        if target is not None:

            ax.scatter(target.pos[0], target.pos[1], target.pos[2],
                       s=1, color='yellow', marker='o', label='Target', zorder=8)

            target_axis_length = 0.1
            target_axes_local = np.eye(3) * target_axis_length
            target_axes_world = target.rot_matrix @ target_axes_local
            target_origin = target.pos

            ax.plot([target_origin[0], target_origin[0] + target_axes_world[0, 0]],
                    [target_origin[1], target_origin[1] + target_axes_world[1, 0]],
                    [target_origin[2], target_origin[2] + target_axes_world[2, 0]],
                    color='red', linewidth=1.5, linestyle='-')

            ax.plot([target_origin[0], target_origin[0] + target_axes_world[0, 1]],
                    [target_origin[1], target_origin[1] + target_axes_world[1, 1]],
                    [target_origin[2], target_origin[2] + target_axes_world[2, 1]],
                    color='green', linewidth=1.5, linestyle='-')

            ax.plot([target_origin[0], target_origin[0] + target_axes_world[0, 2]],
                    [target_origin[1], target_origin[1] + target_axes_world[1, 2]],
                    [target_origin[2], target_origin[2] + target_axes_world[2, 2]],
                    color='blue', linewidth=1.5, linestyle='-')

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_zlabel('Z', fontsize=11)

        all_points = np.vstack([vertices_absolute, positions_absolute, com_total.reshape(1, -1)])
        all_points = np.vstack([all_points, np.zeros((1, 3))])

        if target is not None:
            all_points = np.vstack([all_points, target.pos.reshape(1, -1)])

        max_abs = np.max(np.abs(all_points))
        limit = max_abs * 1.2

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        ax.set_box_aspect([1, 1, 1])
        num_ticks = 12
        ticks = np.linspace(-limit, limit, num_ticks)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)

        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=9)

        if created_fig and show:
            plt.tight_layout()
            plt.show()

        return ax


def demonstrate_genetic_ik():

    robot = FloatingTranslationRobot(
        base_mass=3.0,
        link_lengths=[0.5, 0.5, 0.5],
        link_masses=[1.0, 1.0, 1.0],
        base_size=1.0
    )

    target_position = np.array([0.8, 0.5, 1.0])
    target_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = Coords(target_position, target_rotation)
    full_state_initial = np.array([0, 0, 0, 0, 0, 0])
    initial_pose = robot.forward_kinematics(full_state_initial)

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    robot.visualize(full_state_initial, ax=ax1, show=False)
    plt.tight_layout()
    plt.show()

    ik_solver = GeneticIK(
        robot=robot,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0),
                (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
        population_size=200,
        generations=100,
        position_tolerance=1e-3
    )

    ik_solver.set_target(target)


    start_time = time.time()
    best_state, metrics = ik_solver.run()
    total_time = time.time() - start_time

    print(f"Optimization time: {total_time:.2f} seconds")



    fig2 = plt.figure(figsize=(16, 7))

    ax2 = fig2.add_subplot(121, projection='3d')
    robot.visualize(best_state, target=target, ax=ax2, show=False)
    ax2.set_title(
        f"Кубик: [{best_state[0]:.3f}, {best_state[1]:.3f}, {best_state[2]:.3f}]\nУглы: [{best_state[3]:.3f}, {best_state[4]:.3f}, {best_state[5]:.3f}]\nОшибка: {metrics['position_error']:.6f}")

    ax3 = fig2.add_subplot(222)
    ax3.plot(metrics['position_error_history'], 'b-', linewidth=2, label='Ошибка позиции')
    ax3.set_xlabel('Поколения')
    ax3.set_ylabel('Ошибка')
    ax3.set_title('Ошибка по позиции')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')

    ax4 = fig2.add_subplot(224)
    ax4.plot(metrics['fitness_history'], 'g-', linewidth=2, label='Функционал')
    ax4.set_xlabel('Поколения')
    ax4.set_ylabel('Функционал')
    ax4.set_title('Функционал по поколениям')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    achieved = robot.forward_kinematics(best_state)
    com_total = robot.compute_center_of_mass(best_state)

    return robot, ik_solver, best_state, metrics


if __name__ == "__main__":
    robot, ik_solver, best_state, metrics = demonstrate_genetic_ik()
