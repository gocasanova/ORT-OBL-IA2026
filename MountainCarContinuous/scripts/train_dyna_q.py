"""Entrena lotes independientes de Dyna-Q."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym

from dyna_q_agent import DynaQAgent
from dyna_q_config import (
    BASE_DYNA_Q_CONFIG,
    DEFAULT_EPISODES,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLANNING_STEPS,
    DEFAULT_SEEDS,
    ENV_ID,
    Q_LEARNING_BASELINE_CONFIG_NAME,
)
from dyna_q_results import (
    EVALUATION_COLUMNS,
    TRAINING_COLUMNS,
    append_rows,
    evaluation_summary,
    training_rows,
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planning-steps", nargs="+", type=int, default=DEFAULT_PLANNING_STEPS
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--evaluation-episodes", type=int, default=DEFAULT_EVALUATION_EPISODES
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--position-bins", type=int, default=BASE_DYNA_Q_CONFIG["position_bins"]
    )
    parser.add_argument(
        "--velocity-bins", type=int, default=BASE_DYNA_Q_CONFIG["velocity_bins"]
    )
    parser.add_argument(
        "--action-bins", type=int, default=BASE_DYNA_Q_CONFIG["action_bins"]
    )
    parser.add_argument("--alpha", type=float, default=BASE_DYNA_Q_CONFIG["alpha"])
    parser.add_argument("--gamma", type=float, default=BASE_DYNA_Q_CONFIG["gamma"])
    parser.add_argument("--epsilon", type=float, default=BASE_DYNA_Q_CONFIG["epsilon"])
    parser.add_argument(
        "--epsilon-min", type=float, default=BASE_DYNA_Q_CONFIG["epsilon_min"]
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=BASE_DYNA_Q_CONFIG["epsilon_decay"]
    )
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.max_steps <= 0 or args.evaluation_episodes <= 0:
        raise ValueError(
            "episodes, max-steps y evaluation-episodes deben ser positivos."
        )
    if args.progress_interval <= 0:
        raise ValueError("progress-interval debe ser positivo.")
    if any(value < 0 for value in args.planning_steps):
        raise ValueError("planning-steps no puede contener valores negativos.")
    if len(set(args.planning_steps)) != len(args.planning_steps):
        raise ValueError("planning-steps no puede contener duplicados.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds no puede contener duplicados.")


def agent_config(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    config = {
        "position_bins": args.position_bins,
        "velocity_bins": args.velocity_bins,
        "action_bins": args.action_bins,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "seed": seed,
    }
    return config


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    training_csv = args.results_dir / "dyna_q_training_results.csv"
    evaluation_csv = args.results_dir / "dyna_q_evaluation_results.csv"
    best_candidate: tuple[tuple[float, float, float], Path] | None = None

    print(
        f"Dyna-Q run_id={run_id} | planning={args.planning_steps} | "
        f"seeds={args.seeds} | episodios={args.episodes}",
        flush=True,
    )
    for planning_steps in args.planning_steps:
        for seed in args.seeds:
            experiment_id = uuid.uuid4().hex[:12]
            safe_run_id = re.sub(r"[^a-zA-Z0-9_-]", "_", run_id)
            model_path = args.models_dir / (
                f"dyna_q_p{planning_steps}_seed{seed}_{safe_run_id}_{experiment_id}.pkl"
            )
            config = agent_config(args, seed)
            agent = DynaQAgent(planning_steps=planning_steps, **config)
            env = gym.make(ENV_ID)
            started_at = timestamp()
            try:
                train_start = time.perf_counter()
                history = agent.train_agent(
                    env,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    verbose=not args.quiet,
                    progress_interval=args.progress_interval,
                )
                training_time = time.perf_counter() - train_start
                evaluation_start = time.perf_counter()
                test_results = agent.test_agent(
                    env,
                    episodes=args.evaluation_episodes,
                    max_steps=args.max_steps,
                )
                evaluation_time = time.perf_counter() - evaluation_start
            finally:
                env.close()

            summary = evaluation_summary(test_results, evaluation_time)
            metadata = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "algorithm": "Dyna-Q",
                "created_at": timestamp(),
                "seed": seed,
                "training_episodes": args.episodes,
                "evaluation_episodes": args.evaluation_episodes,
                "max_steps": args.max_steps,
                "epsilon_start": args.epsilon,
                "training_time_seconds": training_time,
                "evaluation_metrics": summary,
                "q_learning_baseline_config": Q_LEARNING_BASELINE_CONFIG_NAME,
            }
            agent.save(model_path, metadata=metadata)

            append_rows(
                training_csv,
                training_rows(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    timestamp=started_at,
                    seed=seed,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    agent=agent,
                    history=history,
                ),
                TRAINING_COLUMNS,
            )
            evaluation_row = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "evaluated_at": timestamp(),
                "algorithm": "Dyna-Q",
                "model_path": model_path.as_posix(),
                "seed": seed,
                "planning_steps": planning_steps,
                "training_episodes": args.episodes,
                "evaluation_episodes": args.evaluation_episodes,
                "max_steps": args.max_steps,
                **summary,
                "training_time_seconds": training_time,
                "position_bins": agent.position_bins,
                "velocity_bins": agent.velocity_bins,
                "action_bins": agent.action_bins,
                "action_values": json.dumps(agent.actions.tolist()),
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "epsilon_start": args.epsilon,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
            }
            append_rows(evaluation_csv, [evaluation_row], EVALUATION_COLUMNS)

            score = (
                float(summary["success_rate"]),
                float(summary["avg_reward"]),
                -float(summary["avg_episode_length"]),
            )
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, model_path)
            print(
                f"  p={planning_steps} seed={seed} | "
                f"reward={summary['avg_reward']:.3f} | "
                f"éxito={summary['success_rate']:.1f}% | "
                f"entrenamiento={training_time:.1f}s | {model_path}",
                flush=True,
            )

    if best_candidate is not None:
        best_path = args.models_dir / "dyna_q_best.pkl"
        shutil.copy2(best_candidate[1], best_path)
        print(f"Mejor modelo del lote: {best_path}")
    print(f"Entrenamiento por episodio: {training_csv}")
    print(f"Evaluaciones: {evaluation_csv}")


if __name__ == "__main__":
    main()
