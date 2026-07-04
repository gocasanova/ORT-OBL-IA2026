"""Pruebas de la configuración simplificada de Q-Learning."""

from __future__ import annotations

import inspect
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q_learning_agent import QLearningAgent
from hyperparameter_search import (
    MANUAL_CONFIGS,
    OVERNIGHT_CONFIGS,
    _prepare_agent_config,
    build_grid_configs,
)


class QLearningAgentTests(unittest.TestCase):
    def test_searches_use_only_requested_hyperparameters(self) -> None:
        allowed = {
            "config_name",
            "notes",
            "position_bins",
            "velocity_bins",
            "action_bins",
            "alpha",
            "gamma",
            "epsilon",
            "epsilon_start",
            "epsilon_min",
            "epsilon_decay",
        }
        for config in [*MANUAL_CONFIGS, *OVERNIGHT_CONFIGS, *build_grid_configs()]:
            self.assertLessEqual(set(config), allowed)

    def test_search_rejects_removed_hyperparameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "Parámetros desconocidos"):
            _prepare_agent_config({"config_name": "invalid", "q_init": 5.0}, seed=42)

    def test_public_configuration_contains_only_requested_parameters(self) -> None:
        parameters = set(inspect.signature(QLearningAgent).parameters)
        self.assertEqual(
            parameters,
            {
                "position_bins",
                "velocity_bins",
                "action_bins",
                "alpha",
                "gamma",
                "epsilon",
                "epsilon_min",
                "epsilon_decay",
                "seed",
            },
        )

    def test_q_table_starts_at_zero_and_actions_are_uniform(self) -> None:
        agent = QLearningAgent(position_bins=3, velocity_bins=4, action_bins=5)

        np.testing.assert_array_equal(agent.q_table, np.zeros((3, 4, 5)))
        np.testing.assert_allclose(agent.actions, np.linspace(-1.0, 1.0, 5))

    def test_current_model_round_trip(self) -> None:
        agent = QLearningAgent(action_bins=3, seed=8)
        agent.q_table[1, 2, 0] = 4.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.pkl"
            agent.save(path, metadata={"name": "current"})
            loaded = QLearningAgent.load(path)

        np.testing.assert_array_equal(loaded.q_table, agent.q_table)
        self.assertEqual(loaded.metadata["name"], "current")

    def test_load_ignores_legacy_hyperparameters(self) -> None:
        actions = np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float32)
        q_table = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
        legacy_data = {
            "version": 2,
            "q_table": q_table,
            "actions": actions,
            "observation_low": np.array([-1.2, -0.07], dtype=np.float32),
            "observation_high": np.array([0.6, 0.07], dtype=np.float32),
            "final_epsilon": 0.1,
            "metadata": {"legacy": True},
            "hyperparameters": {
                "position_bins": 2,
                "velocity_bins": 3,
                "action_bins": 4,
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.995,
                "seed": 3,
                "reward_shaping": "potential",
                "q_init": 5.0,
                "explicit_action_values": actions.tolist(),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            with path.open("wb") as model_file:
                pickle.dump(legacy_data, model_file)
            loaded = QLearningAgent.load(path)

        np.testing.assert_array_equal(loaded.q_table, q_table)
        np.testing.assert_array_equal(loaded.actions, actions)
        self.assertTrue(loaded.metadata["legacy"])


if __name__ == "__main__":
    unittest.main()
