import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

from robots.utils import Coords, Obstacle
from control.IK.ik_base import InverseKinematics


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class OUNoise:
    """Ornstein-Uhlenbeck process — коррелированный шум для исследования."""
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state.copy()


class DDPGIK(InverseKinematics):
    """
    Решение обратной кинематики методом DDPG (Deep Deterministic Policy Gradient).

    Среда:
        - Состояние: нормализованные углы суставов + вектор ошибки позиции + вектор ошибки ориентации
        - Действие: приращения углов суставов (continuous, [-1, 1] → масштабируется на action_scale)
        - Награда: -(комбинированная_ошибка + штраф_препятствий), где комбинированная ошибка
          как в ``GeneticIK``: в режиме ``exp`` вес ориентации растёт сигмоидой при
          уменьшении ошибки позиции; в ``constant`` — фиксированный вес.
        - Эпизод завершается при достижении цели или исчерпании шагов

    Веса: при ``weights_path`` после обучения (``episodes`` > 0) чекпоинт сохраняется;
    при следующем ``solve`` веса подгружаются до обучения. ``episodes=0`` — только
    greedy rollout с уже загруженной политикой (нужен существующий файл весов).
    """

    def __init__(self,
                 robot: 'Robot',
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 hidden_dims: Optional[List[int]] = None,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100_000,
                 batch_size: int = 128,
                 episodes: int = 300,
                 max_steps: int = 100,
                 action_scale: float = 0.05,
                 position_tolerance: float = 1e-2,
                 orientation_tolerance: float = 1e-2,
                 error_weight_mode: str = "exp",
                 constant_orientation_weight: float = 0.5,
                 exp_weight_alpha: float = 5.0,
                 max_orientation_weight: float = 0.95,
                 orientation_weight: Optional[float] = None,
                 obstacle_sigma: float = 0.01,
                 obstacle_weight: float = 1.0,
                 noise_sigma: float = 0.2,
                 noise_theta: float = 0.15,
                 warmup_steps: int = 256,
                 save_episode_images: bool = False,
                 image_dir: str = "ddpg_ik_frames",
                 obstacles=None,
                 weights_path: Optional[str] = None,
                 save_weights_after_run: bool = True,
                 load_weights_if_exist: bool = True,
                 inference_starts: int = 64):
        super().__init__(robot)

        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

        self.target: Optional[Coords] = None
        self.bounds = bounds if bounds else [(-np.pi, np.pi)] * len(robot.dh_params)
        self.hidden_dims = hidden_dims or [256, 256]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episodes = episodes
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        if orientation_weight is not None:
            self.error_weight_mode = "constant"
            self.constant_orientation_weight = float(np.clip(orientation_weight, 0.0, 1.0))
        else:
            self.error_weight_mode = error_weight_mode
            self.constant_orientation_weight = float(np.clip(constant_orientation_weight, 0.0, 1.0))
        self.exp_weight_alpha = exp_weight_alpha
        self.max_orientation_weight = max_orientation_weight
        self.obstacle_sigma = obstacle_sigma
        self.obstacle_weight = obstacle_weight
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.warmup_steps = warmup_steps
        self.save_episode_images = save_episode_images
        self.image_dir = image_dir
        if save_episode_images:
            os.makedirs(image_dir, exist_ok=True)
        self.obstacles: list[Obstacle] = obstacles or []

        self.weights_path = weights_path
        self.save_weights_after_run = save_weights_after_run
        self.load_weights_if_exist = load_weights_if_exist
        self.inference_starts = inference_starts

        self.n_joints = len(robot.dh_params)
        self.state_dim = self.n_joints + 6  # angles + pos_error(3) + orient_error(3)
        self.action_dim = self.n_joints

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reward_history: List[float] = []
        self.position_error_history: List[float] = []
        self.orientation_error_history: List[float] = []
        self.best_angles: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf

        self.best_params = None

    def _build_networks(self):
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.noise = OUNoise(self.action_dim, theta=self.noise_theta, sigma=self.noise_sigma)

        if self.load_weights_if_exist and self.weights_path and os.path.isfile(self.weights_path):
            self._load_weights()

    def _checkpoint_meta(self) -> Dict[str, Any]:
        return {
            "n_joints": self.n_joints,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
        }

    def save_weights(self, path: Optional[str] = None) -> None:
        """Сохранить веса actor/critic и target-сетей (и метаданные архитектуры)."""
        out = path or self.weights_path
        if not out:
            raise ValueError("Укажите weights_path в конструкторе или path в save_weights().")
        if not hasattr(self, "actor"):
            raise RuntimeError("Сети ещё не созданы. Вызовите solve() или run() сначала.")
        parent = os.path.dirname(os.path.abspath(out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "meta": self._checkpoint_meta(),
            },
            out,
        )
        self.logger.info(f"Веса DDPG сохранены: {out}")

    def _load_weights(self) -> bool:
        if not self.weights_path or not os.path.isfile(self.weights_path):
            return False
        try:
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.weights_path, map_location=self.device)
        meta = ckpt.get("meta", {})
        if meta:
            if meta.get("n_joints") != self.n_joints:
                self.logger.warning(
                    f"Чекпоинт n_joints={meta.get('n_joints')} не совпадает с роботом ({self.n_joints}). "
                    "Загрузка пропущена."
                )
                return False
            if meta.get("hidden_dims") != list(self.hidden_dims):
                self.logger.warning(
                    f"Чекпоинт hidden_dims={meta.get('hidden_dims')} ≠ {self.hidden_dims}. "
                    "Загрузка пропущена."
                )
                return False
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.logger.info(f"Веса DDPG загружены из {self.weights_path}")
        return True

    def _greedy_rollout_best_angles(self) -> Tuple[np.ndarray, float, float]:
        """Несколько стартов, детерминированный актор без обучения — лучшие углы по комбинированной ошибке."""
        best_angles = self._random_angles()
        best_pos, best_ori = self._calculate_errors(best_angles)[2:4]
        best_score = self._combined_task_error(best_pos, best_ori)

        for _ in range(max(1, self.inference_starts)):
            angles = self._random_angles()
            state = self._get_state(angles)
            for _step in range(self.max_steps):
                action = self._select_action(state, add_noise=False)
                angles = self._clip_angles(angles + action * self.action_scale)
                state = self._get_state(angles)
            _, _, pos_err, orient_err = self._calculate_errors(angles)
            score = self._combined_task_error(pos_err, orient_err)
            if score < best_score:
                best_score = score
                best_angles = angles.copy()
                best_pos, best_ori = pos_err, orient_err

        return best_angles, best_pos, best_ori

    def _random_angles(self) -> np.ndarray:
        return np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])

    def _clip_angles(self, angles: np.ndarray) -> np.ndarray:
        return np.array([np.clip(a, lo, hi) for a, (lo, hi) in zip(angles, self.bounds)])

    def _calculate_errors(self, angles: np.ndarray):
        """Возвращает (pos_err_vec, orient_err_vec, pos_err_scalar, orient_err_scalar)."""
        fk = self.robot.forward_kinematics(angles)
        pos_err_vec = self.target.pos - fk.pos
        pos_err = float(np.linalg.norm(pos_err_vec))

        R_rel = fk.rot_matrix.T @ self.target.rot_matrix
        trace_val = np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)
        orient_err = float(np.arccos(trace_val))

        axis = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ]) * 0.5
        norm = np.linalg.norm(axis)
        orient_err_vec = (axis / norm * orient_err) if norm > 1e-12 else np.zeros(3)

        return pos_err_vec, orient_err_vec, pos_err, orient_err

    def _combined_task_error(self, pos_err: float, orient_err: float) -> float:
        """Та же логика весов, что в ``GeneticIK._fitness`` (позиция / ориентация)."""
        if self.error_weight_mode == "constant":
            w_rot = self.constant_orientation_weight
        else:
            w_rot_raw = 1.0 / (1.0 + np.exp(self.exp_weight_alpha * (pos_err - self.position_tolerance)))
            w_rot = min(w_rot_raw, self.max_orientation_weight)
        w_pos = 1.0 - w_rot
        return w_pos * pos_err + w_rot * orient_err

    def _obstacle_penalty(self, angles: np.ndarray) -> float:
        if not self.obstacles:
            return 0.0
        penalty = 0.0
        positions = self.robot.get_joint_positions(angles)
        points = [Coords(positions[i]) for i in range(len(positions))]
        for i in range(len(points) - 1):
            for obs in self.obstacles:
                d = obs.dist_to_me(points[i], points[i + 1])
                penalty += np.exp(-d / self.obstacle_sigma)
        return self.obstacle_weight * penalty

    def _get_state(self, angles: np.ndarray) -> np.ndarray:
        pos_err_vec, orient_err_vec, _, _ = self._calculate_errors(angles)
        normalized = np.zeros(self.n_joints)
        for i, (lo, hi) in enumerate(self.bounds):
            rng = hi - lo
            normalized[i] = 2.0 * (angles[i] - lo) / rng - 1.0 if rng > 1e-12 else 0.0
        return np.concatenate([normalized, pos_err_vec, orient_err_vec]).astype(np.float32)

    def _compute_reward(self, angles: np.ndarray, pos_err: float, orient_err: float) -> float:
        obstacle_pen = self._obstacle_penalty(angles)
        combined = self._combined_task_error(pos_err, orient_err)
        reward = -(combined + obstacle_pen)
        if pos_err < self.position_tolerance and orient_err < self.orientation_tolerance:
            reward += 10.0
        elif pos_err < self.position_tolerance * 5:
            reward += 1.0
        return reward

    def _select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if add_noise:
            action = action + self.noise.sample()
            action = np.clip(action, -1.0, 1.0)
        return action

    def _update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_value = rewards_t + self.gamma * (1.0 - dones_t) * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.functional.mse_loss(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def set_target(self, target: Coords):
        self.target = target

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

        self._build_networks()

        self.reward_history = []
        self.position_error_history = []
        self.orientation_error_history = []
        self.best_angles = None
        self.best_fitness = -np.inf

        start_time = time.time()
        total_steps = 0

        if self.episodes <= 0:
            if not (self.load_weights_if_exist and self.weights_path and os.path.isfile(self.weights_path)):
                self.logger.warning(
                    "episodes=0 без файла весов: rollout идёт с текущей (случайной) политикой."
                )
            self.best_angles, ep_pos_err, ep_orient_err = self._greedy_rollout_best_angles()
            self.position_error_history.append(ep_pos_err)
            self.orientation_error_history.append(ep_orient_err)
            self.reward_history.append(0.0)
            total_time = time.time() - start_time
            achieved = self.robot.forward_kinematics(self.best_angles)
            final_pos_err = float(np.linalg.norm(self.target.pos - achieved.pos))
            R_rel = achieved.rot_matrix.T @ self.target.rot_matrix
            trace_val = np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)
            final_orient_err = float(np.arccos(trace_val))
            metrics = {
                "total_time": total_time,
                "best_fitness": -self._combined_task_error(final_pos_err, final_orient_err),
                "target_position": self.target.pos,
                "achieved_position": achieved.pos,
                "position_error": final_pos_err,
                "target_orientation": self.target.rot_matrix,
                "achieved_orientation": achieved.rot_matrix,
                "orientation_error": final_orient_err,
                "episodes_completed": 0,
                "total_steps": 0,
                "reward_history": self.reward_history,
                "position_error_history": self.position_error_history,
                "orientation_error_history": self.orientation_error_history,
                "inference_only": True,
            }
            return self.best_angles, metrics

        for episode in range(self.episodes):
            angles = self._random_angles()
            state = self._get_state(angles)
            self.noise.reset()
            episode_reward = 0.0

            best_ep_fitness = -np.inf
            best_ep_angles = angles.copy()

            for step in range(self.max_steps):
                action = self._select_action(state, add_noise=True)
                delta_angles = action * self.action_scale
                new_angles = self._clip_angles(angles + delta_angles)

                _, _, pos_err, orient_err = self._calculate_errors(new_angles)
                reward = self._compute_reward(new_angles, pos_err, orient_err)

                done = (pos_err < self.position_tolerance and
                        orient_err < self.orientation_tolerance)

                next_state = self._get_state(new_angles)
                self.replay_buffer.push(state, action, reward, next_state, float(done))

                total_steps += 1
                if total_steps > self.warmup_steps:
                    self._update()

                if reward > best_ep_fitness:
                    best_ep_fitness = reward
                    best_ep_angles = new_angles.copy()

                if reward > self.best_fitness:
                    self.best_fitness = reward
                    self.best_angles = new_angles.copy()

                angles = new_angles
                state = next_state
                episode_reward += reward

                if done:
                    break

            _, _, ep_pos_err, ep_orient_err = self._calculate_errors(best_ep_angles)
            self.position_error_history.append(ep_pos_err)
            self.orientation_error_history.append(ep_orient_err)
            self.reward_history.append(episode_reward)

            if self.save_episode_images and (
                    episode % max(1, self.episodes // 50) == 0 or episode == 0):
                self._visualize_episode(episode, best_ep_angles)

            if episode % max(1, self.episodes // 20) == 0 or episode == self.episodes - 1:
                ep_comb = self._combined_task_error(ep_pos_err, ep_orient_err)
                if self.best_angles is not None:
                    _, _, bp, bo = self._calculate_errors(self.best_angles)
                    best_comb = self._combined_task_error(bp, bo)
                    self.logger.info(
                        f"Episode {episode}/{self.episodes}: "
                        f"sum_reward={episode_reward:.4f}, "
                        f"ep_best_step pos={ep_pos_err:.4f} orient={ep_orient_err:.4f} comb={ep_comb:.4f} | "
                        f"global_best pos={bp:.4f} orient={bo:.4f} comb={best_comb:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Episode {episode}/{self.episodes}: "
                        f"sum_reward={episode_reward:.4f}, "
                        f"ep_best_step pos={ep_pos_err:.4f} orient={ep_orient_err:.4f} comb={ep_comb:.4f}"
                    )

        total_time = time.time() - start_time

        if self.best_angles is None:
            self.best_angles = self._random_angles()

        achieved = self.robot.forward_kinematics(self.best_angles)
        final_pos_err = float(np.linalg.norm(self.target.pos - achieved.pos))
        R_rel = achieved.rot_matrix.T @ self.target.rot_matrix
        trace_val = np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0)
        final_orient_err = float(np.arccos(trace_val))

        metrics = {
            'total_time': total_time,
            'best_fitness': self.best_fitness,
            'target_position': self.target.pos,
            'achieved_position': achieved.pos,
            'position_error': final_pos_err,
            'target_orientation': self.target.rot_matrix,
            'achieved_orientation': achieved.rot_matrix,
            'orientation_error': final_orient_err,
            'episodes_completed': len(self.reward_history),
            'total_steps': total_steps,
            'reward_history': self.reward_history,
            'position_error_history': self.position_error_history,
            'orientation_error_history': self.orientation_error_history,
        }

        if self.save_weights_after_run and self.weights_path and self.episodes > 0:
            self.save_weights()

        return self.best_angles, metrics

    def _visualize_episode(self, episode: int, best_angles: np.ndarray):
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        self.robot.visualize(best_angles, target=self.target, ax=ax1, show=False)
        if self.obstacles:
            for obs in self.obstacles:
                obs.visualize(ax1)

        pos_err = self.position_error_history[-1] if self.position_error_history else 0
        orient_err = self.orientation_error_history[-1] if self.orientation_error_history else 0
        ax1.set_title(f'Episode {episode}\n'
                      f'Position Error: {pos_err:.6f}\n'
                      f'Orientation Error: {orient_err:.6f}')

        ax2 = fig.add_subplot(132)
        eps = list(range(1, len(self.position_error_history) + 1))
        ax2.plot(eps, self.position_error_history, 'b-', label='Position Error', linewidth=2)
        ax2.plot(eps, self.orientation_error_history, 'r-', label='Orientation Error', linewidth=2)
        ax2.axhline(y=self.position_tolerance, color='b', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.orientation_tolerance, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Error')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(eps, self.reward_history[:len(eps)], 'g-', label='Episode Reward', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        filename = os.path.join(self.image_dir, f'episode_{episode:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    def create_animation(self, output_path: str = "ddpg_ik_animation.gif", frame_interval: int = 300):
        if not self.save_episode_images:
            print("save_episode_images=False → нет кадров")
            return
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        images = [Image.open(os.path.join(self.image_dir, f)) for f in image_files]
        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:],
                           duration=frame_interval, loop=0)
            print(f"Анимация сохранена как {output_path}")
