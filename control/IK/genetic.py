import numpy as np
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from robots.utils import Coords, Obstacle
from control.IK.ik_base import InverseKinematics
from control.core.genetic_base import GeneticOptimizer
import logging

class GeneticIK(InverseKinematics):
    def __init__(self,
                 robot: 'Robot',
                 population_size: int = 500,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 elite_size: int = 20,
                 position_tolerance: float = 1e-2,
                 orientation_tolerance: float = 1e-2,
                 max_no_improvement: int = 50,
                 early_stopping_delta: float = 1e-2,
                 early_stopping_patience: int = 50,
                 error_weight_mode: str = "exp",
                 constant_orientation_weight: float = 0.5,
                 exp_weight_alpha: float = 5.0,
                 max_orientation_weight: float = 0.95,
                 save_generation_images: bool = True,
                 image_dir: str = "genetic_ik_frames",
                 obstacles=None):
        super().__init__(robot)

        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

        self.target: Optional[Coords] = None
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = bounds if bounds else [(-np.pi, np.pi)] * len(robot.dh_params)
        self.elite_size = elite_size
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_no_improvement = max_no_improvement
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.error_weight_mode = error_weight_mode
        self.constant_orientation_weight = float(np.clip(constant_orientation_weight, 0.0, 1.0))
        self.exp_weight_alpha = exp_weight_alpha
        self.max_orientation_weight = max_orientation_weight
        self.save_generation_images = save_generation_images
        self.image_dir = image_dir
        if save_generation_images:
            os.makedirs(image_dir, exist_ok=True)

        self.obstacles: list[Obstacle] = obstacles or []

        self.fitness_history = []
        self.position_error_history = []
        self.orientation_error_history = []
        self.best_individual_history = []
        self.time_history = []

        self.best_params = None
        self.optimizer: Optional[GeneticOptimizer] = None

    def set_target(self, target: Coords):
        self.target = target

    def wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def angular_difference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff

    def calculate_errors(self, angles: np.ndarray) -> Tuple[float, float]:
        angles_wrapped = self.wrap_angles(angles)
        current = self.robot.forward_kinematics(angles_wrapped)
        pos_error = np.linalg.norm(current.pos - self.target.pos)
        R_rel = current.rot_matrix.T @ self.target.rot_matrix
        trace_val = 0.5 * (np.trace(R_rel) - 1.0)
        trace_clipped = np.clip(trace_val, -1.0, 1.0)
        rot_error = float(np.arccos(trace_clipped))
        return pos_error, rot_error

    def segment_obstacle_penalty(self, p1: Coords, p2: Coords, sigma=0.01, weight=1.0):
        """
        Штраф для каждого препятствия: exp(-d / sigma), где d = dist_to_me.
        """
        penalty = 0.0
        for obs in self.obstacles:
            d = obs.dist_to_me(p1, p2)
            penalty += np.exp(-d / sigma)
        return weight * penalty

    def _create_individual(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high) for low, high in self.bounds])

    def _fitness(self, angles: np.ndarray) -> float:
        pos_error, rot_error = self.calculate_errors(angles)

        if self.error_weight_mode == "constant":
            w_rot = self.constant_orientation_weight
        else:
            # Exponential/sigmoid schedule: orientation gets higher weight near target.
            w_rot_raw = 1.0 / (1.0 + np.exp(self.exp_weight_alpha * (pos_error - self.position_tolerance)))
            w_rot = min(w_rot_raw, self.max_orientation_weight)
        w_pos = 1.0 - w_rot

        combined_error = w_pos * pos_error + w_rot * rot_error

        angular_penalty = 0.0
        if self.best_individual_history:
            last_best = self.best_individual_history[-1]
            d_angles = self.angular_difference(angles, last_best)
            angular_penalty = 0.01 * (np.linalg.norm(d_angles) / np.sqrt(len(d_angles)))

        obstacle_penalty = 0.0
        points_positions = self.robot.get_joint_positions(angles)
        points = [Coords(points_positions[i]) for i in range(len(points_positions))]
        for i in range(len(points)-1):
            obstacle_penalty += self.segment_obstacle_penalty(points[i], points[i+1])

        return - (combined_error + angular_penalty + obstacle_penalty)

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        alpha = 0.3
        if np.random.rand() < self.crossover_rate:
            t = np.random.uniform(-alpha, 1 + alpha, size=p1.shape)
            child = (1 - t) * p1 + t * p2
            return child
        return p1.copy()

    def _mutate(self, ind: np.ndarray, mutation_rate: float) -> np.ndarray:
        out = ind.copy()
        for i in range(len(out)):
            if np.random.rand() < mutation_rate:
                low, high = self.bounds[i]
                range_i = high - low
                sigma = 0.02 * range_i
                out[i] += np.random.randn() * sigma
        out = self.wrap_angles(out)
        return out

    def _create_optimizer(self):
        return GeneticOptimizer(
            pop_size=self.pop_size,
            create_individual=self._create_individual,
            fitness_func=self._fitness,
            crossover_func=self._crossover,
            mutate_func=self._mutate,
            bounds=self.bounds,
            elite_size=self.elite_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            early_stopping_delta=self.early_stopping_delta,
            early_stopping_patience=self.early_stopping_patience,
            max_no_improvement=self.max_no_improvement,
            logger=self.logger
        )

    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not kwargs and self.best_params:
            kwargs = self.best_params
        self.set_target(target)
        return self.run(**kwargs)

    def run(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.target is None:
            raise ValueError("Target not set. Use set_target or solve().")

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.fitness_history = []
        self.position_error_history = []
        self.orientation_error_history = []
        self.best_individual_history = []
        self.time_history = []

        optimizer = self._create_optimizer()

        def callback(gen, best_individual, best_fitness, opt):
            if best_individual is not None:
                pos_err, rot_err = self.calculate_errors(best_individual)
                self.position_error_history.append(pos_err)
                self.orientation_error_history.append(rot_err)
                self.best_individual_history.append(best_individual.copy())
                self.fitness_history.append(best_fitness)

                if self.save_generation_images and (
                        gen % max(1, self.generations // 100) == 0 or gen == 0):
                    self.visualize_generation(gen, best_individual,
                                              self.position_error_history,
                                              self.orientation_error_history)
            self.time_history.append(opt.generation_times[-1] if opt.generation_times else 0)

        best_individual, metrics = optimizer.run(
            generations=self.generations,
            verbose=True,
            callback=callback
        )

        achieved_coords = self.robot.forward_kinematics(best_individual) if best_individual is not None else None

        final_pos_err = None
        final_rot_err = None
        if achieved_coords is not None:
            final_pos_err = np.linalg.norm(self.target.pos - achieved_coords.pos)
            R_rel = achieved_coords.rot_matrix.T @ self.target.rot_matrix
            trace_val = 0.5 * (np.trace(R_rel) - 1.0)
            trace_clipped = np.clip(trace_val, -1.0, 1.0)
            final_rot_err = np.arccos(trace_clipped)

        metrics.update({
            'position_error_history': self.position_error_history,
            'orientation_error_history': self.orientation_error_history,
            'time_history': self.time_history,
            'total_time': metrics['total_time'],
            'best_fitness': metrics['best_fitness'],
            'target_position': self.target.pos,
            'target_orientation': self.target.rot_matrix,
            'achieved_position': achieved_coords.pos if achieved_coords is not None else None,
            'achieved_orientation': achieved_coords.rot_matrix if achieved_coords is not None else None,
            'position_error': final_pos_err,
            'orientation_error': final_rot_err,
            'generations_completed': metrics['generations_completed']
        })
        return best_individual, metrics

    def visualize_generation(self, generation: int, best_individual: np.ndarray,
                             position_errors: List[float], orientation_errors: List[float]):
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        self.robot.visualize(best_individual, target=self.target, ax=ax1, show=False)
        if self.obstacles:
            for obs in self.obstacles:
                obs.visualize(ax1)
        ax1.set_title(f'Generation {generation}\n'
                      f'Position Error: {position_errors[-1]:.6f}\n'
                      f'Orientation Error: {orientation_errors[-1]:.6f}')

        ax2 = fig.add_subplot(132)
        gens = list(range(1, len(position_errors) + 1))
        ax2.plot(gens, position_errors, 'b-', label='Position Error', linewidth=2)
        ax2.plot(gens, orientation_errors, 'r-', label='Orientation Error', linewidth=2)
        ax2.axhline(y=self.position_tolerance, color='b', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.orientation_tolerance, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Error')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        if self.save_generation_images:
            filename = os.path.join(self.image_dir, f'generation_{generation:04d}.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            return filename
        else:
            plt.show()
            return None

    def create_animation(self, output_path: str = "genetic_ik_animation.gif", frame_interval: int = 100):
        if not self.save_generation_images:
            print("save_generation_images=False → нет кадров")
            return
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        images = [Image.open(os.path.join(self.image_dir, f)) for f in image_files]
        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:],
                           duration=frame_interval, loop=0)
            print(f"Анимация сохранена как {output_path}")
