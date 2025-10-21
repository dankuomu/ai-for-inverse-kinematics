import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Dict, Any
from robots.utils import Coords
from IK.ik_base import InverseKinematics
import itertools
from tqdm import tqdm

class GeneticIK(InverseKinematics):
    def __init__(self,
                 robot: 'Robot',
                 population_size: int = 200,
                 generations: int = 1000,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.8,
                 w_pos: float = 0.7,
                 w_rot: float = 0.3,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 elite_size: int = 2,
                 save_animation: bool = False):
        super().__init__(robot)
        self.target: Optional[Coords] = None
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.w_pos = w_pos
        self.w_rot = w_rot
        self.bounds = bounds if bounds else [(-np.pi, np.pi)] * len(robot.dh_params)
        self.elite_size = elite_size

        self.best_params = None

        self.fitness_history = []
        self.time_history = []
        self.position_history = []
        self.best_individual_history = []

    def set_target(self, target: Coords):
        self.target = target

    def fitness(self, angles: np.ndarray) -> float:
        if self.target is None:
            raise ValueError("Target is not set. Call set_target(target) or use solve(target).")
        current = self.robot.forward_kinematics(angles)
        pos_error = np.linalg.norm(current.pos - self.target.pos)
        rot_error = np.arccos(np.clip(0.5 * (np.trace(current.rot_matrix.T @ self.target.rot_matrix) - 1), -1, 1))
        return - (self.w_pos * pos_error + self.w_rot * rot_error)

    def init_population(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high, self.pop_size)
                         for (low, high) in self.bounds]).T

    def select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        probabilities = fitnesses - np.min(fitnesses) + 1e-10
        probabilities = probabilities / np.sum(probabilities)
        indices = np.random.choice(np.arange(self.pop_size), size=2, p=probabilities)
        return population[indices]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            return np.concatenate([parent1[:point], parent2[point:]])
        return parent1

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                low, high = self.bounds[i]
                individual[i] = np.random.uniform(low, high)
        return individual

    def run(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.target is None:
            raise ValueError("Target is not set. Use set_target(target) or call solve(target).")

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        start_time = time.time()
        population = self.init_population()
        best_fitness = -np.inf
        best_individual = None

        for gen in range(self.generations):
            gen_start_time = time.time()
            fitnesses = np.array([self.fitness(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[best_idx].copy()

            self.fitness_history.append(best_fitness)

            new_population = []
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population.extend(population[elite_indices])

            while len(new_population) < self.pop_size:
                parents = self.select(population, fitnesses)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)

            population = np.array(new_population)
            self.time_history.append(time.time() - gen_start_time)

        total_time = time.time() - start_time

        achieved_coords = self.robot.forward_kinematics(best_individual) if best_individual is not None else None

        metrics = {
            'fitness_history': self.fitness_history,
            'time_history': self.time_history,
            'total_time': total_time,
            'best_fitness': best_fitness,
            'target_position': self.target.pos,
            'target_orientation': self.target.rot_matrix,
            'achieved_position': achieved_coords.pos if achieved_coords is not None else None,
            'achieved_orientation': achieved_coords.rot_matrix if achieved_coords is not None else None,
            'position_error': np.linalg.norm(self.target.pos - achieved_coords.pos) if achieved_coords is not None else None,
            'orientation_error': np.arccos(np.clip(0.5 * (np.trace(achieved_coords.rot_matrix.T @ self.target.rot_matrix) - 1), -1, 1)) if achieved_coords is not None else None,
        }

        return best_individual, metrics

    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not kwargs and self.best_params:
            kwargs = self.best_params
        self.set_target(target)
        return self.run(**kwargs)

    def tune(self, targets: List[Coords], param_grid: Optional[Dict[str, List]] = None, **kwargs):
        # Сетка гиперпараметров по умолчанию (можно расширить)
        default_param_grid = {
            "population_size": [50, 100, 200],
            "generations": [50, 100, 200],
            "mutation_rate": [0.01, 0.05, 0.1],
            "crossover_rate": [0.5, 0.8, 0.95],
            "w_pos": [0.5, 1.0, 2.0],
            "w_rot": [0.0, 0.1, 0.5],
            "elite_size": [0, 1, 2, 5],
        }

        # Используем переданную сетку или сетку по умолчанию
        if param_grid is not None:
            # Объединяем с дефолтной сеткой, приоритет у переданной
            grid = default_param_grid.copy()
            grid.update(param_grid)
        else:
            grid = default_param_grid

        # Фильтруем только те параметры, которые есть в классе
        valid_params = {}
        for param_name, values in grid.items():
            if hasattr(self, param_name):
                valid_params[param_name] = values
            else:
                print(f"[TUNE] Предупреждение: параметр '{param_name}' не найден в классе, пропускаем")

        # Создаем все комбинации параметров
        param_names = list(valid_params.keys())
        param_values = list(valid_params.values())
        param_combinations = list(itertools.product(*param_values))

        results = []

        print(f"[TUNE] Запуск подбора гиперпараметров, всего комбинаций: {len(param_combinations)}")
        print(f"[TUNE] Параметры для тюнинга: {param_names}")

        for param_values in tqdm(param_combinations, desc="Grid search", leave=True):
            # Создаем словарь параметров для текущей комбинации
            params_dict = dict(zip(param_names, param_values))

            metrics_list = []
            for target in targets:
                self.set_target(target)
                _, metrics = self.run(**params_dict)
                metrics_list.append(metrics)

            # Агрегируем метрики по всем таргетам
            avg_pos_err = np.mean([m["position_error"] for m in metrics_list])
            avg_orient_err = np.mean([m["orientation_error"] for m in metrics_list])
            avg_fit = np.mean([m["best_fitness"] for m in metrics_list])
            avg_time = np.mean([m["total_time"] for m in metrics_list])
            success_rate = np.mean([1 if m["position_error"] < 0.01 else 0 for m in metrics_list])

            result = {
                "params": params_dict,
                "avg_fitness": avg_fit,
                "avg_pos_err": avg_pos_err,
                "avg_orient_err": avg_orient_err,
                "avg_time": avg_time,
                "success_rate": success_rate,
            }
            results.append(result)

            print(f"[TUNE] {params_dict} → "
                  f"pos_err={avg_pos_err:.6f}, orient_err={avg_orient_err:.6f}, "
                  f"success={success_rate:.2%}, time={avg_time:.4f}s")

        # Сортируем результаты по ошибке позиции (можно изменить критерий)
        results.sort(key=lambda x: x["avg_pos_err"])

        # Выбираем лучшие параметры
        self.best_params = results[0]["params"]

        print(f"[TUNE] Лучшие параметры: {self.best_params}")
        print(f"[TUNE] Лучшая позиционная ошибка: {results[0]['avg_pos_err']:.6f}")
        print(f"[TUNE] Лучшая ошибка ориентации: {results[0]['avg_orient_err']:.6f}")
        print(f"[TUNE] Успешность: {results[0]['success_rate']:.2%}")

        return self.best_params, results


