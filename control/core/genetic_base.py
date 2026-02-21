import numpy as np
import time
import logging
from typing import Callable, Optional, List, Tuple, Dict, Any

class GeneticOptimizer:
    """
    Параметризуется функциями:
        - create_individual: () -> np.ndarray (создание случайной особи)
        - fitness_func: (individual) -> float (оценка приспособленности, максимизируется)
        - crossover_func: (parent1, parent2) -> child
        - mutate_func: (individual, mutation_rate) -> mutated_individual
    """
    def __init__(self,
                 pop_size: int,
                 create_individual: Callable[[], np.ndarray],
                 fitness_func: Callable[[np.ndarray], float],
                 crossover_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 mutate_func: Callable[[np.ndarray, float], np.ndarray],
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 elite_size: int = 5,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 early_stopping_delta: float = 1e-4,
                 early_stopping_patience: int = 50,
                 max_no_improvement: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        self.pop_size = pop_size
        self.create_individual = create_individual
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.bounds = bounds
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.max_no_improvement = max_no_improvement
        self.logger = logger or logging.getLogger(__name__)


        self.best_individuals = []
        self.best_fitnesses = []
        self.generation_times = []

        self._best_fitness_ever = -np.inf
        self._best_individual_ever = None
        self._no_improvement_counter = 0
        self._stagnation_counter = 0
        self._best_error_for_stopping = np.inf

    def _init_population(self) -> np.ndarray:
        return np.array([self.create_individual() for _ in range(self.pop_size)])

    def _select_tournament(self, population: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        idx = np.random.choice(len(population), size=k, replace=False)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx].copy()

    def run(self,
            generations: int,
            verbose: bool = False,
            callback: Optional[Callable] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Запуск эволюции.
        generations: максимальное число поколений.
        callback: функция, вызываемая после каждого поколения с аргументами
                  (generation, best_individual, best_fitness, self)
        """
        population = self._init_population()
        self.best_individuals = []
        self.best_fitnesses = []
        self.generation_times = []

        self._best_fitness_ever = -np.inf
        self._best_individual_ever = None
        self._no_improvement_counter = 0
        self._stagnation_counter = 0
        self._best_error_for_stopping = np.inf

        start_time = time.time()
        stop_reason = None

        for gen in range(generations):
            gen_start = time.time()

            fitnesses = np.array([self.fitness_func(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[best_idx]
            current_best_individual = population[best_idx].copy()

            self.best_fitnesses.append(current_best_fitness)
            self.best_individuals.append(current_best_individual)

            if current_best_fitness > self._best_fitness_ever + 1e-12:
                self._best_fitness_ever = current_best_fitness
                self._best_individual_ever = current_best_individual.copy()
                self._no_improvement_counter = 0
            else:
                self._no_improvement_counter += 1

            if current_best_fitness > self._best_error_for_stopping - self.early_stopping_delta:
                self._best_error_for_stopping = current_best_fitness
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1

            if self._stagnation_counter > self.early_stopping_patience:
                stop_reason = f"Stagnation for {self.early_stopping_patience} generations"
            if self.max_no_improvement and self._no_improvement_counter >= self.max_no_improvement:
                stop_reason = f"No improvement for {self.max_no_improvement} generations"

            if stop_reason:
                if verbose:
                    self.logger.info(f"Stopping at generation {gen}: {stop_reason}")
                break

            # Элитизм
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_indices]

            while len(new_population) < self.pop_size:
                p1 = self._select_tournament(population, fitnesses)
                p2 = self._select_tournament(population, fitnesses)
                if np.random.rand() < self.crossover_rate:
                    child = self.crossover_func(p1, p2)
                else:
                    child = p1.copy()
                child = self.mutate_func(child, self.mutation_rate)
                if self.bounds is not None:
                    for i, (low, high) in enumerate(self.bounds):
                        child[i] = np.clip(child[i], low, high)
                new_population.append(child)

            population = np.array(new_population)

            gen_time = time.time() - gen_start
            self.generation_times.append(gen_time)

            if verbose and (gen % max(1, generations // 20) == 0 or gen == generations - 1):
                self.logger.info(f"Gen {gen}: best fitness = {current_best_fitness:.4f}, "
                                 f"time = {gen_time:.3f}s")

            if callback is not None:
                callback(gen, self._best_individual_ever, self._best_fitness_ever, self)

        total_time = time.time() - start_time

        metrics = {
            'best_fitness': self._best_fitness_ever,
            'best_individual': self._best_individual_ever,
            'fitness_history': self.best_fitnesses,
            'generation_times': self.generation_times,
            'total_time': total_time,
            'generations_completed': len(self.best_fitnesses),
            'stop_reason': stop_reason
        }
        return self._best_individual_ever, metrics