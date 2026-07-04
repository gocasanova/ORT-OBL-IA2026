"""Configuración base comparable con el Q-Learning elegido en la notebook."""

from __future__ import annotations

from typing import Any


ENV_ID = "MountainCarContinuous-v0"
Q_LEARNING_BASELINE_CONFIG_NAME = "baseline_40x40_a11"

# Esta configuración se declara explícitamente: nunca se carga una Q-table de
# Q-Learning para iniciar Dyna-Q.
BASE_DYNA_Q_CONFIG: dict[str, Any] = {
    "position_bins": 40,
    "velocity_bins": 40,
    "action_bins": 11,
    "alpha": 0.1,
    "gamma": 0.995,
    "epsilon": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.9995,
}

DEFAULT_PLANNING_STEPS = [0, 5, 10, 20, 50]
DEFAULT_EPISODES = 20_000
DEFAULT_MAX_STEPS = 999
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_SEEDS = [42]
