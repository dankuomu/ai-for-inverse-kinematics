import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from robots.utils import Coords
from control.IK.ik_base import InverseKinematics
from control.IK.ml_dataset import (
    build_xy_from_robot,
    joint_grid_to_configurations,
    orientation_error_radians,
    position_error,
    sample_random_joint_configs,
)

logger = logging.getLogger(__name__)


def _metrics_for_prediction(robot, pred: np.ndarray, target: Coords) -> Dict:
    achieved = robot.forward_kinematics(pred)
    pos_err = position_error(achieved, target)
    rot_err = orientation_error_radians(achieved, target)
    return {
        "target_position": target.pos,
        "achieved_position": achieved.pos,
        "position_error": pos_err,
        "target_orientation": target.rot_matrix,
        "achieved_orientation": achieved.rot_matrix,
        "orientation_error": rot_err,
        "best_fitness": pos_err + rot_err,
    }


class RandomForestIK(InverseKinematics):
    def __init__(self,
                 robot,
                 n_estimators: int = 100,
                 max_depth: int = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 auto_train: bool = False,
                 dataset_size: int = 100000,
                 angle_limits: Optional[List[Tuple[float, float]]] = None):

        super().__init__(robot)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )

        self.test_size = test_size
        self.random_state = random_state
        self.trained = False

        self.angle_limits = angle_limits
        self.dataset_size = dataset_size

        self._log = logger.getChild(self.__class__.__name__)

        if auto_train:
            self.generate_dataset_random(angle_limits, dataset_size)
            self.train(plot_learning_curve=False)

    def generate_dataset_random(
            self,
            angle_limits: Optional[List[Tuple[float, float]]] = None,
            n_samples: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        limits = angle_limits if angle_limits is not None else self.angle_limits
        if limits is None:
            raise ValueError("Задайте angle_limits в конструкторе или в generate_dataset_random.")
        n = n_samples if n_samples is not None else self.dataset_size
        A = sample_random_joint_configs(limits, n, rng=rng)
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float64, dtype_y=np.float64)
        self._log.info("Dataset (random): X=%s, y=%s", self.X.shape, self.y.shape)
        return self.X, self.y

    def generate_dataset_grid(
            self,
            angle_limits: List[Tuple[float, float]],
            joint_value_grid: Dict[int, Sequence[float]],
            default_angles: Optional[np.ndarray] = None,
            max_combinations: int = 50_000,
    ):
        n_j = len(angle_limits)
        A = joint_grid_to_configurations(
            n_j, angle_limits, joint_value_grid, default_angles, max_combinations=max_combinations
        )
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float64, dtype_y=np.float64)
        self._log.info("Dataset (grid): X=%s, y=%s", self.X.shape, self.y.shape)
        return self.X, self.y

    def generate_dataset(self, angle_limits: List[Tuple[float, float]], n_samples: int = 5000):
        return self.generate_dataset_random(angle_limits, n_samples)

    def train(self, plot_learning_curve=True):

        try:
            self.X
        except AttributeError:
            self.generate_dataset_random(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        train_mse = np.mean((self.model.predict(X_train) - y_train) ** 2)
        test_mse = np.mean((self.model.predict(X_test) - y_test) ** 2)

        self._log.info("Training complete: train MSE=%.6e, test MSE=%.6e", train_mse, test_mse)

        if plot_learning_curve:
            self._plot_learning_curve(X_train, y_train, X_test, y_test)

    def _plot_learning_curve(self, X_train, y_train, X_test, y_test, steps=10):

        sizes = np.linspace(0.1, 1.0, steps)
        train_err, test_err = [], []

        for frac in sizes:
            m = int(frac * len(X_train))
            Xp = X_train[:m]
            yp = y_train[:m]

            model = RandomForestRegressor(
                n_estimators=self.model.n_estimators,
                max_depth=self.model.max_depth,
                n_jobs=-1,
                random_state=0
            )
            model.fit(Xp, yp)

            train_err.append(np.mean((model.predict(Xp) - yp) ** 2))
            test_err.append(np.mean((model.predict(X_test) - y_test) ** 2))

        plt.figure(figsize=(8, 5))
        plt.plot(sizes, train_err, label="Train MSE")
        plt.plot(sizes, test_err, label="Test MSE")
        plt.xlabel("Dataset fraction")
        plt.ylabel("MSE")
        plt.title("Learning Curve — Random Forest IK")
        plt.grid(True)
        plt.legend()
        plt.show()

    def solve(self, target: Coords, **kwargs):

        if not self.trained:
            raise RuntimeError("RandomForestIK not trained. Use train() or auto_train=True.")

        x = np.concatenate([target.pos, target.rot_matrix.reshape(-1)]).reshape(1, -1)
        pred = self.model.predict(x)[0]

        metrics = _metrics_for_prediction(self.robot, pred, target)
        return pred, metrics


class XGBoostIK(InverseKinematics):
    def __init__(self,
                 robot,
                 n_estimators: int = 500,
                 max_depth: int = 8,
                 learning_rate: float = 0.05,
                 subsample: float = 0.9,
                 colsample_bytree: float = 0.9,
                 tree_method: str = "hist",
                 n_jobs: int = -1,
                 objective: str = "reg:squarederror",
                 random_state: int = 42,
                 test_size: float = 0.2,
                 auto_train: bool = False,
                 dataset_size: int = 100000,
                 angle_limits: Optional[List[Tuple[float, float]]] = None,
                 model_path: Optional[str] = None):
        super().__init__(robot)

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            n_jobs=n_jobs,
            objective=objective,
            random_state=random_state
        )

        self._log = logger.getChild(self.__class__.__name__)

        self.test_size = test_size
        self.auto_train = auto_train
        self.dataset_size = dataset_size
        self.angle_limits = angle_limits
        self.model_path = model_path
        self.trained = False

        if self.model_path is not None:
            self._try_load(self.model_path)

        if self.auto_train and not self.trained:
            self.generate_dataset_random(self.angle_limits, self.dataset_size)
            self.train(plot_learning_curve=False)

    def _try_load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        import joblib
        self.model = joblib.load(path)
        self.trained = True
        self._log.info("Loaded XGBoost model from %s", path)

    def generate_dataset_random(
            self,
            angle_limits: Optional[List[Tuple[float, float]]] = None,
            n_samples: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        limits = angle_limits if angle_limits is not None else self.angle_limits
        if limits is None:
            raise ValueError("Задайте angle_limits в конструкторе или в generate_dataset_random.")
        n = n_samples if n_samples is not None else self.dataset_size
        A = sample_random_joint_configs(limits, n, rng=rng)
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float64, dtype_y=np.float64)
        self._log.info("Dataset (random): X=%s, y=%s", self.X.shape, self.y.shape)
        return self.X, self.y

    def generate_dataset_grid(
            self,
            angle_limits: List[Tuple[float, float]],
            joint_value_grid: Dict[int, Sequence[float]],
            default_angles: Optional[np.ndarray] = None,
            max_combinations: int = 50_000,
    ):
        n_j = len(angle_limits)
        A = joint_grid_to_configurations(
            n_j, angle_limits, joint_value_grid, default_angles, max_combinations=max_combinations
        )
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float64, dtype_y=np.float64)
        self._log.info("Dataset (grid): X=%s, y=%s", self.X.shape, self.y.shape)
        return self.X, self.y

    def generate_dataset(self, angle_limits: List[Tuple[float, float]], n_samples: int = 5000):
        return self.generate_dataset_random(angle_limits, n_samples)

    def train(self, plot_learning_curve=True):
        try:
            self.X
        except AttributeError:
            self.generate_dataset_random(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        train_mse = np.mean((self.model.predict(X_train) - y_train) ** 2)
        test_mse = np.mean((self.model.predict(X_test) - y_test) ** 2)

        self._log.info("Training complete: train MSE=%.6e, test MSE=%.6e", train_mse, test_mse)

        if self.model_path:
            import joblib
            joblib.dump(self.model, self.model_path)
            self._log.info("Model saved to %s", self.model_path)

        if plot_learning_curve:
            self._plot_learning_curve(X_train, y_train, X_test, y_test)

    def _plot_learning_curve(self, X_train, y_train, X_test, y_test, steps=10):
        sizes = np.linspace(0.1, 1.0, steps)
        train_err, test_err = [], []
        for frac in sizes:
            m = int(frac * len(X_train))
            Xp = X_train[:m]
            yp = y_train[:m]
            model = XGBRegressor(**self.model.get_params())
            model.fit(Xp, yp)
            train_err.append(np.mean((model.predict(Xp) - yp) ** 2))
            test_err.append(np.mean((model.predict(X_test) - y_test) ** 2))
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, train_err, label="Train MSE")
        plt.plot(sizes, test_err, label="Test MSE")
        plt.xlabel("Dataset fraction")
        plt.ylabel("MSE")
        plt.title("Learning Curve — XGBoost IK")
        plt.grid(True)
        plt.legend()
        plt.show()

    def solve(self, target: Coords, **kwargs):
        if not self.trained:
            raise RuntimeError("XGBoostIK not trained. Use train() or auto_train=True.")
        x = np.concatenate([target.pos, target.rot_matrix.reshape(-1)]).reshape(1, -1)
        pred = self.model.predict(x)[0]
        metrics = _metrics_for_prediction(self.robot, pred, target)
        return pred, metrics
