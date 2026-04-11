import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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


class MLP(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class ForwardNeuralIK(InverseKinematics):
    def __init__(self,
                 robot,
                 layers: List[int],
                 lr: float = 1e-3,
                 epochs: int = 50,
                 batch_size: int = 256,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 auto_train: bool = False,
                 dataset_size: int = 50000,
                 angle_limits: Optional[List[Tuple[float, float]]] = None,
                 model_path: Optional[str] = None):

        super().__init__(robot)

        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.angle_limits = angle_limits
        self.model_path = model_path

        self.model = MLP(layers)
        self.trained = False

        self._log = logger.getChild(self.__class__.__name__)

        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location="cpu")
            except Exception:
                state = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
            self.trained = True
            self._log.debug("Loaded pretrained model from %s", model_path)
        else:
            self._log.info("No pretrained model found. Training required.")

        if auto_train and not self.trained:
            self.generate_dataset_random(angle_limits, dataset_size)
            self.train()

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
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float32, dtype_y=np.float32)
        self._log.debug("Dataset (random): X=%s, y=%s", self.X.shape, self.y.shape)
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
        self.X, self.y = build_xy_from_robot(self.robot, A, dtype_x=np.float32, dtype_y=np.float32)
        self._log.debug("Dataset (grid): X=%s, y=%s", self.X.shape, self.y.shape)
        return self.X, self.y

    def generate_dataset(self,
                         angle_limits: List[Tuple[float, float]],
                         n_samples: int = 5000):
        """Обратная совместимость: то же, что ``generate_dataset_random``."""
        return self.generate_dataset_random(angle_limits, n_samples)

    def train(self):
        try:
            self.X
        except AttributeError:
            self.generate_dataset_random(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log.debug("Using device: %s", device)
        self.model.to(device)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_train_t = torch.tensor(X_train).to(device)
        y_train_t = torch.tensor(y_train).to(device)
        X_test_t = torch.tensor(X_test).to(device)
        y_test_t = torch.tensor(y_test).to(device)

        for epoch in range(self.epochs):
            idx = torch.randperm(len(X_train_t))
            Xb = X_train_t[idx]
            yb = y_train_t[idx]

            for i in range(0, len(Xb), self.batch_size):
                bx = Xb[i:i + self.batch_size]
                by = yb[i:i + self.batch_size]

                opt.zero_grad()
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    test_loss = loss_fn(self.model(X_test_t), y_test_t).item()
                self._log.debug("Epoch %s/%s, Test MSE=%.6e", epoch + 1, self.epochs, test_loss)

        self.trained = True

        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)
            self._log.info("Saved model to %s", self.model_path)

    def solve(self, target: Coords, **kwargs):
        if not self.trained:
            raise RuntimeError("NeuralIK not trained.")

        device = next(self.model.parameters()).device

        x = np.concatenate([target.pos, target.rot_matrix.reshape(-1)]).astype(np.float32)
        x_t = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = self.model(x_t).cpu().numpy()[0]

        achieved = self.robot.forward_kinematics(pred)
        pos_err = position_error(achieved, target)
        rot_err = orientation_error_radians(achieved, target)

        return pred, {
            "position_error": pos_err,
            "orientation_error": rot_err,
            "best_fitness": pos_err + rot_err,
            "target_position": target.pos,
            "achieved_position": achieved.pos,
            "target_orientation": target.rot_matrix,
            "achieved_orientation": achieved.rot_matrix,
        }
