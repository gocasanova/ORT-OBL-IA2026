"""Agente Q-Learning tabular para ``MountainCarContinuous-v0``."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


class QLearningAgent:
    """Q-Learning con discretización de estados y acciones continuas."""

    DEFAULT_OBSERVATION_LOW = np.array([-1.2, -0.07], dtype=np.float32)
    DEFAULT_OBSERVATION_HIGH = np.array([0.6, 0.07], dtype=np.float32)

    def __init__(
        self,
        position_bins: int = 40,
        velocity_bins: int = 40,
        action_bins: int = 9,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self._validate_hyperparameters(
            position_bins,
            velocity_bins,
            action_bins,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        )

        self.position_bins = int(position_bins)
        self.velocity_bins = int(velocity_bins)
        self.action_bins = int(action_bins)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.seed = seed

        self.observation_low = self.DEFAULT_OBSERVATION_LOW.copy()
        self.observation_high = self.DEFAULT_OBSERVATION_HIGH.copy()
        self._build_state_edges()

        self.actions = np.linspace(-1.0, 1.0, self.action_bins, dtype=np.float32)
        self.q_table = np.zeros(
            (self.position_bins, self.velocity_bins, self.action_bins),
            dtype=np.float64,
        )
        self.rng = np.random.default_rng(seed)
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _validate_hyperparameters(
        position_bins: int,
        velocity_bins: int,
        action_bins: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        if position_bins < 2 or velocity_bins < 2 or action_bins < 2:
            raise ValueError("Cada discretización debe tener al menos 2 bins.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha debe estar en el intervalo (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma debe estar en el intervalo [0, 1].")
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("Debe cumplirse 0 <= epsilon_min <= epsilon <= 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay debe estar en el intervalo (0, 1].")

    def _build_state_edges(self) -> None:
        self.position_edges = np.linspace(
            self.observation_low[0],
            self.observation_high[0],
            self.position_bins + 1,
            dtype=np.float32,
        )[1:-1]
        self.velocity_edges = np.linspace(
            self.observation_low[1],
            self.observation_high[1],
            self.velocity_bins + 1,
            dtype=np.float32,
        )[1:-1]

    def _use_environment_bounds(self, env: Any) -> None:
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        if low.shape != (2,) or high.shape != (2,):
            raise ValueError("El observation_space debe contener posición y velocidad.")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Los límites del observation_space deben ser finitos.")
        if np.any(high <= low):
            raise ValueError(
                "Los límites superiores deben ser mayores a los inferiores."
            )

        if not (
            np.array_equal(low, self.observation_low)
            and np.array_equal(high, self.observation_high)
        ):
            self.observation_low = low.copy()
            self.observation_high = high.copy()
            self._build_state_edges()

    def discretize_observation(self, observation: np.ndarray) -> tuple[int, int]:
        """Convierte ``[posición, velocidad]`` en un par de índices."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (2,):
            raise ValueError("La observación debe tener forma (2,).")

        position_index = int(np.digitize(obs[0], self.position_edges))
        velocity_index = int(np.digitize(obs[1], self.velocity_edges))
        position_index = int(np.clip(position_index, 0, self.position_bins - 1))
        velocity_index = int(np.clip(velocity_index, 0, self.velocity_bins - 1))
        return position_index, velocity_index

    def _learning_reward(
        self,
        env_reward: float,
        observation: np.ndarray,
        next_observation: np.ndarray,
    ) -> float:
        """Mantiene una única señal de aprendizaje: el reward del ambiente."""
        return float(env_reward)

    def _action_array(self, action_index: int) -> np.ndarray:
        return np.array([self.actions[action_index]], dtype=np.float32)

    def next_action(self, obs: np.ndarray) -> np.ndarray:
        """Devuelve la mejor acción conocida, sin exploración."""
        state = self.discretize_observation(obs)
        action_index = int(np.argmax(self.q_table[state]))
        return self._action_array(action_index)

    def epsilon_greedy_action(self, state: tuple[int, int]) -> tuple[int, np.ndarray]:
        if self.rng.random() < self.epsilon:
            action_index = int(self.rng.integers(self.action_bins))
        else:
            values = self.q_table[state]
            best_indices = np.flatnonzero(values == values.max())
            action_index = int(self.rng.choice(best_indices))
        return action_index, self._action_array(action_index)

    def _reset_environment(self, env: Any, episode: int, offset: int = 0) -> Any:
        episode_seed = None if self.seed is None else self.seed + offset + episode
        observation, _ = env.reset(seed=episode_seed)
        if episode_seed is not None:
            env.action_space.seed(episode_seed)
        return observation

    def train_agent(
        self,
        env: Any,
        episodes: int = 1000,
        max_steps: int = 999,
        verbose: bool = False,
        progress_interval: int = 500,
    ) -> dict[str, list[Any]]:
        """Entrena con Q-Learning; ``rewards`` siempre conserva reward del entorno."""
        if episodes <= 0 or max_steps <= 0 or progress_interval <= 0:
            raise ValueError(
                "episodes, max_steps y progress_interval deben ser positivos."
            )
        self._use_environment_bounds(env)

        history: dict[str, list[Any]] = {
            "rewards": [],
            "env_rewards": [],
            "learning_rewards": [],
            "steps": [],
            "successes": [],
            "epsilons": [],
            "avg_positions": [],
            "max_positions": [],
        }
        training_start = time.perf_counter()

        for episode in range(episodes):
            observation = self._reset_environment(env, episode)
            state = self.discretize_observation(observation)
            total_env_reward = 0.0
            total_learning_reward = 0.0
            success = False
            epsilon_used = self.epsilon
            position_sum = float(observation[0])
            position_count = 1
            max_position = float(observation[0])

            for step in range(1, max_steps + 1):
                action_index, action = self.epsilon_greedy_action(state)
                next_observation, env_reward, terminated, truncated, _ = env.step(
                    action
                )
                next_state = self.discretize_observation(next_observation)
                learning_reward = self._learning_reward(
                    float(env_reward), observation, next_observation
                )

                # Una truncación por límite de tiempo no es un estado terminal del MDP.
                future_value = (
                    0.0 if terminated else float(self.q_table[next_state].max())
                )
                td_target = learning_reward + self.gamma * future_value
                td_error = td_target - self.q_table[state + (action_index,)]
                self.q_table[state + (action_index,)] += self.alpha * td_error

                total_env_reward += float(env_reward)
                total_learning_reward += learning_reward
                position = float(next_observation[0])
                position_sum += position
                position_count += 1
                max_position = max(max_position, position)
                observation = next_observation
                state = next_state
                if terminated:
                    success = True
                if terminated or truncated:
                    break

            history["rewards"].append(total_env_reward)
            history["env_rewards"].append(total_env_reward)
            history["learning_rewards"].append(total_learning_reward)
            history["steps"].append(step)
            history["successes"].append(success)
            history["epsilons"].append(epsilon_used)
            history["avg_positions"].append(position_sum / position_count)
            history["max_positions"].append(max_position)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (
                (episode + 1) % progress_interval == 0 or episode + 1 == episodes
            ):
                window = min(100, len(history["rewards"]))
                recent_reward = float(np.mean(history["env_rewards"][-window:]))
                recent_position = float(np.mean(history["max_positions"][-window:]))
                recent_successes = int(sum(history["successes"][-window:]))
                elapsed = time.perf_counter() - training_start
                print(
                    f"  episodio {episode + 1}/{episodes} | "
                    f"reward_env={recent_reward:.3f} | "
                    f"max_position={recent_position:.3f} | "
                    f"epsilon={self.epsilon:.4f} | "
                    f"éxitos={recent_successes}/{window} | "
                    f"tiempo={elapsed:.1f}s",
                    flush=True,
                )

        return history

    def test_agent(
        self, env: Any, episodes: int = 10, max_steps: int = 999
    ) -> dict[str, Any]:
        """Evalúa greedy usando y reportando solo recompensa original del entorno."""
        if episodes <= 0 or max_steps <= 0:
            raise ValueError("episodes y max_steps deben ser positivos.")
        self._use_environment_bounds(env)

        rewards: list[float] = []
        steps_per_episode: list[int] = []
        successes: list[bool] = []
        max_positions: list[float] = []
        action_counts = {float(action): 0 for action in self.actions}

        for episode in range(episodes):
            observation = self._reset_environment(env, episode, offset=100_000)
            total_reward = 0.0
            success = False
            max_position = float(observation[0])

            for step in range(1, max_steps + 1):
                action = self.next_action(observation)
                action_counts[float(action[0])] += 1
                observation, env_reward, terminated, truncated, _ = env.step(action)
                total_reward += float(env_reward)
                max_position = max(max_position, float(observation[0]))
                if terminated:
                    success = True
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            steps_per_episode.append(step)
            successes.append(success)
            max_positions.append(max_position)

        success_count = int(sum(successes))
        return {
            "rewards": rewards,
            "steps": steps_per_episode,
            "successes": successes,
            "success_count": success_count,
            "success_rate": 100.0 * success_count / episodes,
            "max_positions": max_positions,
            "action_counts": action_counts,
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Guarda tabla, acciones, hiperparámetros y metadata del experimento."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 3,
            "q_table": self.q_table,
            "actions": self.actions,
            "observation_low": self.observation_low,
            "observation_high": self.observation_high,
            "final_epsilon": self.epsilon,
            "metadata": metadata or {},
            "hyperparameters": {
                "position_bins": self.position_bins,
                "velocity_bins": self.velocity_bins,
                "action_bins": self.action_bins,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "seed": self.seed,
            },
        }
        with destination.open("wb") as model_file:
            pickle.dump(data, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Reconstruye modelos actuales y conserva compatibilidad con versiones 1/2."""
        with Path(path).open("rb") as model_file:
            data = pickle.load(model_file)

        if data.get("version") not in {1, 2, 3}:
            raise ValueError("Versión de modelo no soportada.")
        supported = {
            "position_bins",
            "velocity_bins",
            "action_bins",
            "alpha",
            "gamma",
            "epsilon",
            "epsilon_min",
            "epsilon_decay",
            "seed",
        }
        hyperparameters = {
            key: value
            for key, value in data["hyperparameters"].items()
            if key in supported
        }
        agent = cls(**hyperparameters)
        agent.observation_low = np.asarray(data["observation_low"], dtype=np.float32)
        agent.observation_high = np.asarray(data["observation_high"], dtype=np.float32)
        agent._build_state_edges()
        agent.actions = np.asarray(data["actions"], dtype=np.float32)
        agent.action_bins = len(agent.actions)
        agent.q_table = np.asarray(data["q_table"], dtype=np.float64)
        agent.epsilon = float(data.get("final_epsilon", agent.epsilon))
        agent.metadata = dict(data.get("metadata", {}))

        expected_shape = (
            agent.position_bins,
            agent.velocity_bins,
            agent.action_bins,
        )
        if agent.q_table.shape != expected_shape:
            raise ValueError("El archivo contiene dimensiones incompatibles.")
        return agent
