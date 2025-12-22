import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Dict, Any
from robots.utils import Coords, Obstacle
from IK.ik_base import InverseKinematics
import itertools
from tqdm import tqdm
import os
from PIL import Image

import logging


class GeneticIK(InverseKinematics):
    def __init__(self,
                 robot: 'Robot',
                 population_size: int = 500,
                 generations: int = 100,
                 mutation_rate: float = 0.0,
                 crossover_rate: float = 0.8,
                 w_pos: float = 0.1,
                 w_rot: float = 0.9,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 elite_size: int = 20,
                 save_animation: bool = False,
                 position_tolerance: float = 1e-3,
                 orientation_tolerance: float = 1e-3,
                 max_no_improvement: int = 50,
                 save_generation_images: bool = False,
                 image_dir: str = "genetic_ik_frames",
                 early_stopping_delta: float = 1e-2,
                 early_stopping_patience: int = 50,
                 # Новые параметры для адаптивной мутации
                 adaptive_mutation: bool = True,
                 min_mutation_rate: float = 1e-3,
                 max_mutation_rate: float = 1e-3,
                 mutation_steepness: float = 10.0,
                 mutation_transition_point: float = 0.1,
                 obstacles = None):
        super().__init__(robot)

        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

        self.target: Optional[Coords] = None
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.w_pos = w_pos
        self.w_rot = w_rot
        self.bounds = bounds if bounds else [(-np.pi, np.pi)] * len(robot.dh_params)
        self.elite_size = elite_size

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_no_improvement = max_no_improvement

        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience

        self.save_generation_images = save_generation_images
        self.image_dir = image_dir
        if save_generation_images:
            os.makedirs(image_dir, exist_ok=True)

        # Новые параметры адаптивной мутации
        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.mutation_steepness = mutation_steepness
        self.mutation_transition_point = mutation_transition_point

        self.best_params = None
        self.fitness_history = []
        self.time_history = []
        self.position_error_history = []
        self.orientation_error_history = []
        self.best_individual_history = []
        self.mutation_rate_history = []  # Для отслеживания изменения коэффициента мутации

        self.obstacles: list[Obstacle] = obstacles or []

    def get_adaptive_mutation_rate(self, pos_error: float, rot_error: float) -> float:
        """
        Вычисляет адаптивный коэффициент мутации на основе текущей ошибки.
        Используется сигмоида: высокая мутация при большой ошибке, низкая при малой.
        """
        if not self.adaptive_mutation:
            return self.mutation_rate

        # Комбинированная ошибка (можно настроить веса)
        combined_error = self.w_pos * pos_error + self.w_rot * rot_error

        # Нормализуем ошибку относительно точки перехода
        normalized_error = combined_error / self.mutation_transition_point

        # Сигмоида: от min_mutation_rate при малой ошибке до max_mutation_rate при большой
        sigmoid = 1.0 / (1.0 + np.exp(-self.mutation_steepness * (normalized_error - 1.0)))

        adaptive_rate = self.min_mutation_rate + (self.max_mutation_rate - self.min_mutation_rate) * sigmoid

        return float(np.clip(adaptive_rate, self.min_mutation_rate, self.max_mutation_rate))

    def select_tournament(self, population: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        def tournament_one():
            idx = np.random.choice(len(population), size=k, replace=False)
            best = idx[np.argmax(fitnesses[idx])]
            return population[best].copy()

        return np.array([tournament_one(), tournament_one()])

    def crossover_blend(self, parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        if np.random.rand() < self.crossover_rate:
            t = np.random.uniform(-alpha, 1 + alpha, size=parent1.shape)
            child = (1 - t) * parent1 + t * parent2
            for i in range(len(child)):
                low, high = self.bounds[i]
                child[i] = np.clip(child[i], low, high)
            return child
        return parent1.copy()

    def mutate_gaussian(self, individual: np.ndarray, current_mutation_rate: float,
                        base_sigma: float = 0.001) -> np.ndarray:
        """
        Модифицированная функция мутации с передачей текущего коэффициента мутации.
        """
        out = individual.copy()
        for i in range(len(out)):
            if np.random.rand() < current_mutation_rate:
                low, high = self.bounds[i]
                range_i = high - low
                sigma = base_sigma * range_i
                out[i] += np.random.randn() * sigma
                out[i] = np.clip(out[i], low, high)
        out = self.wrap_angles(out)
        return out

    def population_variance(self, population: np.ndarray) -> float:
        cos_mean = np.mean(np.cos(population), axis=0)
        sin_mean = np.mean(np.sin(population), axis=0)
        R = np.sqrt(cos_mean ** 2 + sin_mean ** 2)
        circ_var = 1.0 - R
        return float(np.mean(circ_var))

    def wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def angular_difference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff

    def set_target(self, target: Coords):
        self.target = target

    def calculate_errors(self, angles: np.ndarray) -> Tuple[float, float]:
        angles_wrapped = self.wrap_angles(angles)
        current = self.robot.forward_kinematics(angles_wrapped)
        pos_error = np.linalg.norm(current.pos - self.target.pos)
        R_rel = current.rot_matrix.T @ self.target.rot_matrix
        trace_val = 0.5 * (np.trace(R_rel) - 1.0)
        trace_clipped = np.clip(trace_val, -1.0, 1.0)
        rot_error = float(np.arccos(trace_clipped))
        return pos_error, rot_error

    def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = np.abs(a - b)
        return np.minimum(diff, 2 * np.pi - diff)

    def fitness(self, angles: np.ndarray) -> float:
        pos_error, rot_error = self.calculate_errors(angles)

        pos_s = pos_error
        rot_s = rot_error
        alpha = 20.0
        w_rot_raw = 1.0 / (1.0 + np.exp(alpha * (pos_s - self.position_tolerance)))
        max_w_rot = 0.9
        w_rot = min(w_rot_raw, max_w_rot)
        w_pos = 1.0 - w_rot

        combined_error = w_pos * pos_s + w_rot * rot_s

        angular_penalty = 0.0
        if self.best_individual_history:
            last_best = self.best_individual_history[-1]
            d_angles = self.angular_difference(angles, last_best)
            angular_penalty = 0.01 * (np.linalg.norm(d_angles) / np.sqrt(len(d_angles)))

        obstacle_penalty = 0.0

        def segment_obstacle_penalty(p1, p2, obstacles, weight=1.0):
            """
            p1, p2: np.array(3) - концы сегмента
            obstacles: список объектов Obstacle
            """
            penalty = 0.0
            for obs in obstacles:
                # Найти ближайшую точку на отрезке к центру / объекту
                # стандартный метод: проекция точки на сегмент
                seg_vec = p2 - p1
                seg_len2 = np.dot(seg_vec, seg_vec)
                if seg_len2 < 1e-12:
                    closest = p1
                else:
                    t = np.dot(obs.center.pos - p1, seg_vec) / seg_len2
                    t = np.clip(t, 0, 1)
                    closest = p1 + t * seg_vec
                d = obs.distance_to_point(closest)
                if d < 0:
                    penalty += -d
            return weight * penalty

        points = self.robot.get_joint_positions(angles)
        obs_penalty = 0.0
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            obstacle_penalty += segment_obstacle_penalty(p1, p2, self.obstacles)

        fitness_value = - (combined_error + angular_penalty + obstacle_penalty)
        return float(fitness_value)

    def init_population(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high, self.pop_size)
                         for (low, high) in self.bounds]).T

    def visualize_generation(self, generation: int, best_individual: np.ndarray,
                             position_errors: List[float], orientation_errors: List[float],
                             mutation_rates: List[float] = None):
        fig = plt.figure(figsize=(18, 6))

        # 3D визуализация робота
        ax1 = fig.add_subplot(131, projection='3d')
        self.robot.visualize(best_individual, target=self.target, ax=ax1, show=False)

        # --- Визуализация препятствий ---
        if self.obstacles:
            for obs in self.obstacles:
                obs.visualize(ax1)
        # -------------------------------

        ax1.set_title(f'Generation {generation}\n'
                      f'Position Error: {position_errors[-1]:.6f}\n'
                      f'Orientation Error: {orientation_errors[-1]:.6f}')

        # График ошибок
        ax2 = fig.add_subplot(132)
        generations_range = list(range(1, len(position_errors) + 1))
        ax2.plot(generations_range, position_errors, 'b-', label='Position Error', linewidth=2)
        ax2.plot(generations_range, orientation_errors, 'r-', label='Orientation Error', linewidth=2)
        ax2.axhline(y=self.position_tolerance, color='b', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.orientation_tolerance, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Error')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # График коэффициента мутации (если есть данные)
        if mutation_rates and len(mutation_rates) == len(position_errors):
            ax3 = fig.add_subplot(133)
            ax3.plot(generations_range, mutation_rates, 'g-', label='Mutation Rate', linewidth=2)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Mutation Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, max(mutation_rates) * 1.1)

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

    def run(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.target is None:
            raise ValueError("Target not set. Use set_target or solve().")

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        start_time = time.time()
        population = self.init_population()
        best_fitness = -np.inf
        best_individual = None
        no_improvement_count = 0
        converged = False

        best_error = np.inf
        stagnation_counter = 0

        self.position_error_history = []
        self.orientation_error_history = []
        self.mutation_rate_history = []  # Сбрасываем историю мутаций

        for gen in range(self.generations):
            gen_start_time = time.time()

            # Вычисляем текущие ошибки для лучшей особи предыдущего поколения
            current_pos_error, current_rot_error = 0.0, 0.0
            if best_individual is not None:
                current_pos_error, current_rot_error = self.calculate_errors(best_individual)

            # Вычисляем адаптивный коэффициент мутации
            current_mutation_rate = self.get_adaptive_mutation_rate(current_pos_error, current_rot_error)
            self.mutation_rate_history.append(current_mutation_rate)

            fitnesses = np.array([self.fitness(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[best_idx]
            current_best_individual = population[best_idx].copy()

            pos_error, orient_error = self.calculate_errors(current_best_individual)
            self.position_error_history.append(pos_error)
            self.orientation_error_history.append(orient_error)

            combined_err = self.w_pos * pos_error + self.w_rot * orient_error

            if pos_error < self.position_tolerance and orient_error < self.orientation_tolerance:
                print(f"[OK] Сходимость на поколении {gen}: pos={pos_error:.6f}, rot={orient_error:.6f}")
                converged = True
                break

            if combined_err + self.early_stopping_delta < best_error:
                best_error = combined_err
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > self.early_stopping_patience:
                print(f"[STOP] Стагнация {self.early_stopping_patience} поколений, выход "
                      f"(pos={pos_error:.4e}, rot={orient_error:.4e})")
                break

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            self.fitness_history.append(best_fitness)
            self.best_individual_history.append(best_individual.copy())

            if self.save_generation_images and (
                    gen % max(1, self.generations // 100) == 0 or gen == 0 or gen == self.generations - 1):
                self.visualize_generation(gen, best_individual,
                                          self.position_error_history,
                                          self.orientation_error_history,
                                          self.mutation_rate_history)

            if no_improvement_count >= self.max_no_improvement:
                print(f"[STOP] Нет улучшений {self.max_no_improvement} поколений, выход на {gen}")
                break

            elite_size = min(self.elite_size, self.pop_size // 5)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            new_population = [population[i].copy() for i in elite_indices]

            while len(new_population) < self.pop_size:
                parents = self.select_tournament(population, fitnesses, k=3)
                child = self.crossover_blend(parents[0], parents[1], alpha=0.3)
                # Используем адаптивный коэффициент мутации
                child = self.mutate_gaussian(child, current_mutation_rate, base_sigma=0.02)
                new_population.append(child)

            population = np.array(new_population)

            self.time_history.append(time.time() - gen_start_time)

            if gen % max(1, self.generations // 20) == 0:
                print(
                    f"Generation {gen}: pos={pos_error:.6f}, rot={orient_error:.6f}, mut_rate={current_mutation_rate:.4f}")

        total_time = time.time() - start_time
        achieved_coords = self.robot.forward_kinematics(best_individual) if best_individual is not None else None

        metrics = {
            'fitness_history': self.fitness_history,
            'position_error_history': self.position_error_history,
            'orientation_error_history': self.orientation_error_history,
            'mutation_rate_history': self.mutation_rate_history,  # Добавляем историю мутаций
            'time_history': self.time_history,
            'total_time': total_time,
            'best_fitness': best_fitness,
            'target_position': self.target.pos,
            'target_orientation': self.target.rot_matrix,
            'achieved_position': achieved_coords.pos if achieved_coords is not None else None,
            'achieved_orientation': achieved_coords.rot_matrix if achieved_coords is not None else None,
            'position_error': np.linalg.norm(
                self.target.pos - achieved_coords.pos) if achieved_coords is not None else None,
            'orientation_error': np.arccos(
                np.clip(0.5 * (np.trace(achieved_coords.rot_matrix.T @ self.target.rot_matrix) - 1), -1, 1)
            ) if achieved_coords is not None else None,
            'converged': converged,
            'generations_completed': len(self.fitness_history),
        }
        return best_individual, metrics

    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not kwargs and self.best_params:
            kwargs = self.best_params
        self.set_target(target)
        return self.run(**kwargs)

    def tune(self, targets: List[Coords], param_grid: Optional[Dict[str, List]] = None, **kwargs):
        default_param_grid = {
            "population_size": [50, 100, 200],
            "generations": [200],
            "mutation_rate": [0.05, 0.1, 0.2],
            "crossover_rate": [0.5, 0.8, 0.95],
            "w_pos": [0.9, 0.5, 0.2],
            "w_rot": [0.1, 0.5, 0.8],
            "elite_size": [5, 10, 20],
            "position_tolerance": [1e-2],
            "orientation_tolerance": [1e-2],
            "max_no_improvement": [10, 20],
            # Новые параметры для настройки
            "adaptive_mutation": [True, False],
            "min_mutation_rate": [0.01, 0.05],
            "max_mutation_rate": [0.3, 0.5, 0.7],
            "mutation_steepness": [5.0, 10.0, 20.0],
            "mutation_transition_point": [0.05, 0.1, 0.2],
        }

        grid = default_param_grid if param_grid is None else {**default_param_grid, **param_grid}

        valid_params = {k: v for k, v in grid.items() if hasattr(self, k)}
        param_names = list(valid_params.keys())
        param_values = list(valid_params.values())
        param_combinations = list(itertools.product(*param_values))

        results = []
        print(f"[TUNE] Комбинаций: {len(param_combinations)}")

        for param_vals in tqdm(param_combinations, desc="Grid search"):
            params_dict = dict(zip(param_names, param_vals))
            metrics_list = []
            for target in targets:
                self.set_target(target)
                _, metrics = self.run(**params_dict)
                metrics_list.append(metrics)

            avg_pos_err = np.mean([m["position_error"] for m in metrics_list])
            avg_orient_err = np.mean([m["orientation_error"] for m in metrics_list])
            avg_fit = np.mean([m["best_fitness"] for m in metrics_list])
            avg_time = np.mean([m["total_time"] for m in metrics_list])
            success_rate = np.mean([1 if m["position_error"] < 0.01 else 0 for m in metrics_list])
            conv_rate = np.mean([1 if m["converged"] else 0 for m in metrics_list])
            avg_gens = np.mean([m["generations_completed"] for m in metrics_list])

            result = {
                "params": params_dict,
                "avg_fitness": avg_fit,
                "avg_pos_err": avg_pos_err,
                "avg_orient_err": avg_orient_err,
                "avg_time": avg_time,
                "success_rate": success_rate,
                "convergence_rate": conv_rate,
                "avg_generations": avg_gens,
            }
            results.append(result)
            print(f"[TUNE] {params_dict} → pos={avg_pos_err:.6f}, orient={avg_orient_err:.6f}")

        results.sort(key=lambda x: x["avg_pos_err"])
        self.best_params = results[0]["params"]

        print(f"[TUNE] Лучшие параметры: {self.best_params}")
        return self.best_params, results