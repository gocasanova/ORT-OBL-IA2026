"""Pruebas unitarias del núcleo tabular de Dyna-Q."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dyna_q_agent import DynaQAgent


class _ObservationSpace:
    low = np.array([-1.2, -0.07], dtype=np.float32)
    high = np.array([0.6, 0.07], dtype=np.float32)


class _ActionSpace:
    def seed(self, seed: int) -> None:
        self.last_seed = seed


class _TruncatingEnvironment:
    observation_space = _ObservationSpace()
    action_space = _ActionSpace()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        return np.array([-0.5, 0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.array([-0.49, 0.01], dtype=np.float32), -1.0, False, True, {}


class DynaQAgentTests(unittest.TestCase):
    def build_agent(self, planning_steps: int) -> DynaQAgent:
        return DynaQAgent(
            planning_steps=planning_steps,
            position_bins=2,
            velocity_bins=2,
            action_bins=2,
            alpha=0.5,
            gamma=0.9,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            seed=7,
        )

    def test_real_update_and_model_with_no_planning(self) -> None:
        agent = self.build_agent(planning_steps=0)
        state = (0, 0)
        next_state = (1, 1)
        agent.observe_transition(state, 1, 2.0, next_state, True)

        self.assertAlmostEqual(agent.q_table[state + (1,)], 1.0)
        self.assertEqual(agent.model[(state, 1)], (2.0, next_state, True))

    def test_planning_reuses_observed_transition(self) -> None:
        agent = self.build_agent(planning_steps=2)
        agent.observe_transition((0, 0), 0, 1.0, (1, 0), True)

        # 0 -> 0.5 en el paso real; 0.75 y 0.875 en los dos simulados.
        self.assertAlmostEqual(agent.q_table[0, 0, 0], 0.875)

    def test_non_terminal_update_bootstraps_from_next_state(self) -> None:
        agent = self.build_agent(planning_steps=0)
        agent.q_table[1, 1] = [4.0, 2.0]
        agent.observe_transition((0, 0), 1, 2.0, (1, 1), False)

        self.assertAlmostEqual(agent.q_table[0, 0, 1], 2.8)

    def test_training_stores_truncation_as_done(self) -> None:
        agent = self.build_agent(planning_steps=0)
        history = agent.train_agent(_TruncatingEnvironment(), episodes=1)

        self.assertEqual(history["steps"], [1])
        self.assertEqual(len(agent.model), 1)
        self.assertTrue(next(iter(agent.model.values()))[2])

    def test_repeated_pair_overwrites_model_without_duplicate_key(self) -> None:
        agent = self.build_agent(planning_steps=0)
        agent.observe_transition((0, 0), 0, 1.0, (1, 0), False)
        agent.observe_transition((0, 0), 0, 3.0, (1, 1), True)

        self.assertEqual(len(agent.model), 1)
        self.assertEqual(len(agent._model_keys), 1)
        self.assertEqual(agent.model[((0, 0), 0)], (3.0, (1, 1), True))

    def test_save_and_load_preserves_independent_dyna_model(self) -> None:
        agent = self.build_agent(planning_steps=5)
        agent.observe_transition((0, 0), 1, 2.0, (1, 1), True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pkl"
            agent.save(path, metadata={"experiment_id": "test"})
            loaded = DynaQAgent.load(path)

        np.testing.assert_array_equal(loaded.q_table, agent.q_table)
        np.testing.assert_array_equal(loaded.actions, agent.actions)
        self.assertEqual(loaded.model, agent.model)
        self.assertEqual(loaded.planning_steps, 5)
        self.assertEqual(loaded.metadata["experiment_id"], "test")

    def test_rejects_negative_planning_steps(self) -> None:
        with self.assertRaises(ValueError):
            self.build_agent(planning_steps=-1)


if __name__ == "__main__":
    unittest.main()
