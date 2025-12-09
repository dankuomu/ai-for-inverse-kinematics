from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from robots.utils import Coords

class InverseKinematics(ABC):
    def __init__(self, robot):
        self.robot = robot
        self.best_params: Dict[str, Any] | None = None
        self.logging = None

    def set_target(self, target: Coords):
        self.target = target

    def tune(self, targets: List[Coords], param_grid: Dict[str, List[Any]]):
        raise NotImplementedError("Этот метод нужно реализовать в конкретном IK алгоритме.")

    @abstractmethod
    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError("Этот метод нужно реализовать в конкретном IK алгоритме.")

    def _check_angle_limits(self, angle_limits: List[Tuple[float, float]]):
        if len(angle_limits) > 6 and any(a != b for a, b in angle_limits):
            self.logging.warning(
                "Данный метод не поддерживает решения IK для излишнего числа степеней свободы"
                "Ограничьте angle_limits до 6 степеней свободы, иначе алгоритм может плохо отрабатывать"
            )