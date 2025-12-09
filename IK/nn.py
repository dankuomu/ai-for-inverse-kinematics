import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

from robots.utils import Coords
from IK.ik_base import InverseKinematics


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

        self.logger = self.logging.getLogger(__name__).getChild(self.__class__.__name__)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.trained = True
            self.logger.debug(f"Loaded pretrained model from {model_path}")
        else:
            self.logger.info("No pretrained model found. Training required.")

        if auto_train and not self.trained:
            self.generate_dataset(angle_limits, dataset_size)
            self.train()

    def generate_dataset(self,
                         angle_limits: List[Tuple[float, float]],
                         n_samples: int = 5000):

        X, y = [], []
        for _ in range(n_samples):
            angles = np.array([np.random.uniform(a, b) for (a, b) in angle_limits])
            coords = self.robot.forward_kinematics(angles)
            feat = np.concatenate([coords.pos, coords.rot_matrix.reshape(-1)])
            X.append(feat)
            y.append(angles)

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

        self.logger.debug(f"Dataset created: X={self.X.shape}, y={self.y.shape}")
        return self.X, self.y

    def train(self):
        try: self.X
        except AttributeError:
            self.generate_dataset(self.angle_limits, self.dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Using device: {device}")
        self.model.to(device)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_train_t = torch.tensor(X_train).to(device)
        y_train_t = torch.tensor(y_train).to(device)
        X_test_t  = torch.tensor(X_test).to(device)
        y_test_t  = torch.tensor(y_test).to(device)

        for epoch in range(self.epochs):
            idx = torch.randperm(len(X_train_t))
            Xb = X_train_t[idx]
            yb = y_train_t[idx]

            for i in range(0, len(Xb), self.batch_size):
                bx = Xb[i:i+self.batch_size]
                by = yb[i:i+self.batch_size]

                opt.zero_grad()
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()

            if (epoch+1) % 5 == 0:
                with torch.no_grad():
                    test_loss = loss_fn(self.model(X_test_t), y_test_t).item()
                self.logger.debug(f"Epoch {epoch+1}/{self.epochs}, Test MSE={test_loss:.6e}")

        self.trained = True

        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)
            self.logger.info(f"Saved model to {self.model_path}")

    def solve(self, target: Coords, **kwargs):
        if not self.trained:
            raise RuntimeError("NeuralIK not trained.")

        device = next(self.model.parameters()).device

        x = np.concatenate([target.pos, target.rot_matrix.reshape(-1)]).astype(np.float32)
        x_t = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = self.model(x_t).cpu().numpy()[0]

        achieved = self.robot.forward_kinematics(pred)

        pos_err = np.linalg.norm(achieved.pos - target.pos)
        rot_err = np.linalg.norm(achieved.rot_matrix - target.rot_matrix)

        return pred, {
            "position_error": pos_err,
            "orientation_error": rot_err,
            "best_fitness": pos_err + rot_err
        }