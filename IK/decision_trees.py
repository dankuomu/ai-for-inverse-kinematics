import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict, Any

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from robots.utils import Coords
from IK.ik_base import InverseKinematics

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

        if auto_train:
            self.generate_dataset(angle_limits, dataset_size)
            self.train(plot_learning_curve=False)

    def generate_dataset(self,
                         angle_limits: List[Tuple[float, float]],
                         n_samples: int = 5000):

        X, y = [], []

        for _ in range(n_samples):
            angles = np.array([np.random.uniform(a, b) for (a, b) in angle_limits])
            coords = self.robot.forward_kinematics(angles)

            feature = np.concatenate([coords.pos, coords.rot_matrix.reshape(-1)])
            X.append(feature)
            y.append(angles)

        self.X = np.array(X)
        self.y = np.array(y)

        print(f"[RF IK] Dataset created: X={self.X.shape}, y={self.y.shape}")

        return self.X, self.y

    def train(self, plot_learning_curve=True):

        try: self.X
        except AttributeError:
            _, _ = self.generate_dataset(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        train_mse = np.mean((self.model.predict(X_train) - y_train)**2)
        test_mse  = np.mean((self.model.predict(X_test)  - y_test)**2)

        print("[RF IK] Training complete:")
        print(f"  Train MSE = {train_mse:.6e}")
        print(f"  Test  MSE = {test_mse:.6e}")

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

            train_err.append(np.mean((model.predict(Xp) - yp)**2))
            test_err .append(np.mean((model.predict(X_test) - y_test)**2))

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

        achieved = self.robot.forward_kinematics(pred)

        pos_err = np.linalg.norm(achieved.pos - target.pos)
        rot_err = np.linalg.norm(achieved.rot_matrix - target.rot_matrix)

        metrics = {
            "target_position": target.pos,
            "achieved_position": achieved.pos,
            "position_error": pos_err,
            "target_orientation": target.rot_matrix,
            "achieved_orientation": achieved.rot_matrix,
            "orientation_error": rot_err,
            "best_fitness": pos_err + rot_err
        }

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

        self.logger = self.logging.getLogger(__name__).getChild(self.__class__.__name__)

        self.test_size = test_size
        self.auto_train = auto_train
        self.dataset_size = dataset_size
        self.angle_limits = angle_limits
        self.model_path = model_path
        self.trained = False

        if self.model_path is not None:
            self._try_load(self.model_path)

        if self.auto_train and not self.trained:
            self.generate_dataset(self.angle_limits, self.dataset_size)
            self.train(plot_learning_curve=False)

    def generate_dataset(self, angle_limits: List[Tuple[float, float]], n_samples: int = 5000):
        X, y = [], []
        for _ in range(n_samples):
            angles = np.array([np.random.uniform(a, b) for (a, b) in angle_limits])
            coords = self.robot.forward_kinematics(angles)
            feature = np.concatenate([coords.pos, coords.rot_matrix.reshape(-1)])
            X.append(feature)
            y.append(angles)
        self.X = np.array(X)
        self.y = np.array(y)
        print(f"[XGB IK] Dataset created: X={self.X.shape}, y={self.y.shape}")
        return self.X, self.y

    def train(self, plot_learning_curve=True):
        try: self.X
        except AttributeError:
            self.generate_dataset(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=1
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        train_mse = np.mean((self.model.predict(X_train) - y_train)**2)
        test_mse  = np.mean((self.model.predict(X_test)  - y_test)**2)

        self.logger
        print("[XGB IK] Training complete:")
        print(f"  Train MSE = {train_mse:.6e}")
        print(f"  Test  MSE = {test_mse:.6e}")

        if self.model_path:
            import joblib
            joblib.dump(self.model, self.model_path)
            print(f"[XGB IK] Model saved to {self.model_path}")

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
            train_err.append(np.mean((model.predict(Xp) - yp)**2))
            test_err.append(np.mean((model.predict(X_test) - y_test)**2))
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
        achieved = self.robot.forward_kinematics(pred)
        pos_err = np.linalg.norm(achieved.pos - target.pos)
        rot_err = np.linalg.norm(achieved.rot_matrix - target.rot_matrix)
        metrics = {
            "target_position": target.pos,
            "achieved_position": achieved.pos,
            "position_error": pos_err,
            "target_orientation": target.rot_matrix,
            "achieved_orientation": achieved.rot_matrix,
            "orientation_error": rot_err,
            "best_fitness": pos_err + rot_err
        }
        return pred, metrics
