"""Carga un modelo Dyna-Q y lo evalúa con política greedy."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym

from dyna_q_agent import DynaQAgent
from dyna_q_config import DEFAULT_EVALUATION_EPISODES, DEFAULT_MAX_STEPS, ENV_ID
from dyna_q_results import (
    EVALUATION_COLUMNS,
    append_rows,
    evaluation_summary,
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/dyna_q_best.pkl")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/dyna_q_evaluation_results.csv",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes y max-steps deben ser positivos.")

    agent = DynaQAgent.load(args.model)
    env = gym.make(ENV_ID)
    try:
        start = time.perf_counter()
        test_results = agent.test_agent(
            env, episodes=args.episodes, max_steps=args.max_steps
        )
        evaluation_time = time.perf_counter() - start
    finally:
        env.close()

    summary = evaluation_summary(test_results, evaluation_time)
    metadata = agent.metadata
    row = {
        "run_id": metadata.get("run_id", f"reevaluation_{uuid.uuid4().hex[:8]}"),
        "experiment_id": metadata.get("experiment_id", uuid.uuid4().hex[:12]),
        "evaluated_at": timestamp(),
        "algorithm": "Dyna-Q",
        "model_path": args.model.as_posix(),
        "seed": agent.seed,
        "planning_steps": agent.planning_steps,
        "training_episodes": metadata.get("training_episodes", ""),
        "evaluation_episodes": args.episodes,
        "max_steps": args.max_steps,
        **summary,
        "training_time_seconds": metadata.get("training_time_seconds", ""),
        "position_bins": agent.position_bins,
        "velocity_bins": agent.velocity_bins,
        "action_bins": agent.action_bins,
        "action_values": json.dumps(agent.actions.tolist()),
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon_start": metadata.get("epsilon_start", ""),
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
    }
    append_rows(args.output, [row], EVALUATION_COLUMNS)
    print(
        f"reward={summary['avg_reward']:.3f} ± {summary['std_reward']:.3f} | "
        f"mín={summary['min_reward']:.3f} | máx={summary['max_reward']:.3f} | "
        f"éxito={summary['success_rate']:.1f}% | "
        f"pasos={summary['avg_episode_length']:.1f}"
    )
    print(f"Evaluación registrada en {args.output}")


if __name__ == "__main__":
    main()
