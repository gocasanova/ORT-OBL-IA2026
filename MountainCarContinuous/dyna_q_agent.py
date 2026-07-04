"""Agente Dyna-Q tabular para ``MountainCarContinuous-v0``."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from q_learning_agent import QLearningAgent


State = tuple[int, int]
ModelKey = tuple[State, int]
ModelTransition = tuple[float, State, bool]


class DynaQAgent(QLearningAgent):
    """Q-Learning con un modelo tabular y actualizaciones de planificación."""

    def __init__(self, planning_steps: int = 10, **kwargs: Any) -> None:
        if not isinstance(planning_steps, int) or planning_steps < 0:
            raise ValueError("planning_steps debe ser un entero no negativo.")
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        self.model: dict[ModelKey, ModelTransition] = {}
        self._model_keys: list[ModelKey] = []

    def _q_update(
        self,
        state: State,
        action_index: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Aplica una actualización Q-Learning real o simulada."""
        future_value = 0.0 if done else float(self.q_table[next_state].max())
        td_target = float(reward) + self.gamma * future_value
        index = state + (action_index,)
        self.q_table[index] += self.alpha * (td_target - self.q_table[index])

    def _remember_transition(
        self,
        state: State,
        action_index: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        key = (state, int(action_index))
        if key not in self.model:
            self._model_keys.append(key)
        self.model[key] = (float(reward), next_state, bool(done))

    def plan(self) -> None:
        """Muestrea el modelo aprendido y actualiza la Q-table."""
        if not self._model_keys:
            return
        for _ in range(self.planning_steps):
            key = self._model_keys[int(self.rng.integers(len(self._model_keys)))]
            reward, next_state, done = self.model[key]
            state, action_index = key
            self._q_update(state, action_index, reward, next_state, done)

    def observe_transition(
        self,
        state: State,
        action_index: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Aprende de un paso real, actualiza el modelo y luego planifica."""
        self._q_update(state, action_index, reward, next_state, done)
        self._remember_transition(state, action_index, reward, next_state, done)
        self.plan()

    def train_agent(
        self,
        env: Any,
        episodes: int = 1000,
        max_steps: int = 999,
        verbose: bool = False,
        progress_interval: int = 500,
    ) -> dict[str, list[Any]]:
        """Entrena desde una Q-table propia y registra métricas por episodio."""
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
            "episode_times": [],
            "elapsed_times": [],
            "model_sizes": [],
            "planning_steps": [],
        }
        training_start = time.perf_counter()

        for episode in range(episodes):
            episode_start = time.perf_counter()
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
                done = bool(terminated or truncated)
                self.observe_transition(
                    state, action_index, learning_reward, next_state, done
                )

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
                if done:
                    break

            now = time.perf_counter()
            history["rewards"].append(total_env_reward)
            history["env_rewards"].append(total_env_reward)
            history["learning_rewards"].append(total_learning_reward)
            history["steps"].append(step)
            history["successes"].append(success)
            history["epsilons"].append(epsilon_used)
            history["avg_positions"].append(position_sum / position_count)
            history["max_positions"].append(max_position)
            history["episode_times"].append(now - episode_start)
            history["elapsed_times"].append(now - training_start)
            history["model_sizes"].append(len(self.model))
            history["planning_steps"].append(self.planning_steps)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (
                (episode + 1) % progress_interval == 0 or episode + 1 == episodes
            ):
                window = min(100, len(history["rewards"]))
                recent_reward = float(np.mean(history["env_rewards"][-window:]))
                recent_successes = int(sum(history["successes"][-window:]))
                print(
                    f"  episodio {episode + 1}/{episodes} | "
                    f"planning={self.planning_steps} | "
                    f"reward_env={recent_reward:.3f} | "
                    f"epsilon={self.epsilon:.4f} | "
                    f"éxitos={recent_successes}/{window} | "
                    f"modelo={len(self.model)} | "
                    f"tiempo={now - training_start:.1f}s",
                    flush=True,
                )

        return history

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Guarda el agente Dyna-Q sin depender de modelos Q-Learning."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "algorithm": "dyna_q",
            "q_table": self.q_table,
            "model": self.model,
            "actions": self.actions,
            "observation_low": self.observation_low,
            "observation_high": self.observation_high,
            "final_epsilon": self.epsilon,
            "metadata": metadata or {},
            "hyperparameters": {
                "planning_steps": self.planning_steps,
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
    def load(cls, path: str | Path) -> "DynaQAgent":
        """Reconstruye un modelo Dyna-Q guardado por esta clase."""
        with Path(path).open("rb") as model_file:
            data = pickle.load(model_file)

        if data.get("version") not in {1, 2} or data.get("algorithm") != "dyna_q":
            raise ValueError("El archivo no es un modelo Dyna-Q compatible.")
        supported = {
            "planning_steps",
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
        agent.model = dict(data.get("model", {}))
        agent._model_keys = list(agent.model)

        expected_shape = (
            agent.position_bins,
            agent.velocity_bins,
            agent.action_bins,
        )
        if agent.q_table.shape != expected_shape:
            raise ValueError("El archivo contiene dimensiones incompatibles.")
        return agent
