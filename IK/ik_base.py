from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from robots.utils import Coords

class InverseKinematics(ABC):
    def __init__(self, robot):
        self.robot = robot
        self.best_params: Dict[str, Any] | None = None

    def set_target(self, target: Coords):
        self.target = target

    def tune(self, targets: List[Coords], param_grid: Dict[str, List[Any]]):
        raise NotImplementedError("Этот метод нужно реализовать в конкретном IK алгоритме.")

    @abstractmethod
    def solve(self, target: Coords, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError("Этот метод нужно реализовать в конкретном IK алгоритме.")