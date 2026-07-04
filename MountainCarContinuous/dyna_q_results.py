"""Escritura y resumen de resultados de Dyna-Q."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


TRAINING_COLUMNS = [
    "run_id",
    "experiment_id",
    "timestamp",
    "algorithm",
    "seed",
    "episode",
    "episodes",
    "reward",
    "moving_average_reward",
    "episode_length",
    "epsilon",
    "planning_steps",
    "success",
    "episode_time_seconds",
    "elapsed_training_time_seconds",
    "model_size",
    "position_bins",
    "velocity_bins",
    "action_bins",
    "action_values",
    "alpha",
    "gamma",
    "epsilon_min",
    "epsilon_decay",
    "max_steps",
]

EVALUATION_COLUMNS = [
    "run_id",
    "experiment_id",
    "evaluated_at",
    "algorithm",
    "model_path",
    "seed",
    "planning_steps",
    "training_episodes",
    "evaluation_episodes",
    "max_steps",
    "avg_reward",
    "max_reward",
    "min_reward",
    "std_reward",
    "success_rate",
    "success_count",
    "avg_episode_length",
    "std_episode_length",
    "avg_episode_time_seconds",
    "evaluation_time_seconds",
    "training_time_seconds",
    "position_bins",
    "velocity_bins",
    "action_bins",
    "action_values",
    "alpha",
    "gamma",
    "epsilon_start",
    "epsilon_min",
    "epsilon_decay",
]

COMPARISON_COLUMNS = [
    "algorithm",
    "run_id",
    "experiments",
    "episodes",
    "alpha",
    "gamma",
    "epsilon_decay",
    "planning_steps",
    "avg_training_reward",
    "avg_evaluation_reward",
    "best_reward",
    "training_time_seconds",
    "avg_success_rate",
    "avg_episode_length",
    "source",
]


def append_rows(
    path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str]
) -> None:
    """Agrega filas manteniendo un esquema fijo."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    has_content = destination.exists() and destination.stat().st_size > 0
    if has_content:
        with destination.open("r", newline="", encoding="utf-8") as csv_file:
            header = next(csv.reader(csv_file), [])
        if header != columns:
            raise ValueError(f"El esquema de {destination} no es compatible.")

    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if not has_content:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def load_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    with source.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def evaluation_summary(
    test_results: dict[str, Any], elapsed_seconds: float
) -> dict[str, float | int]:
    rewards = [float(value) for value in test_results["rewards"]]
    steps = [float(value) for value in test_results["steps"]]
    return {
        "avg_reward": fmean(rewards),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "std_reward": pstdev(rewards),
        "success_rate": float(test_results["success_rate"]),
        "success_count": int(test_results["success_count"]),
        "avg_episode_length": fmean(steps),
        "std_episode_length": pstdev(steps),
        "avg_episode_time_seconds": elapsed_seconds / len(rewards),
        "evaluation_time_seconds": elapsed_seconds,
    }


def training_rows(
    *,
    run_id: str,
    experiment_id: str,
    timestamp: str,
    seed: int,
    episodes: int,
    max_steps: int,
    agent: Any,
    history: dict[str, list[Any]],
    moving_average_window: int = 100,
) -> list[dict[str, Any]]:
    rows = []
    rewards = history["env_rewards"]
    for index, reward in enumerate(rewards):
        window_start = max(0, index + 1 - moving_average_window)
        rows.append(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "algorithm": "Dyna-Q",
                "seed": seed,
                "episode": index + 1,
                "episodes": episodes,
                "reward": reward,
                "moving_average_reward": fmean(rewards[window_start : index + 1]),
                "episode_length": history["steps"][index],
                "epsilon": history["epsilons"][index],
                "planning_steps": agent.planning_steps,
                "success": int(bool(history["successes"][index])),
                "episode_time_seconds": history["episode_times"][index],
                "elapsed_training_time_seconds": history["elapsed_times"][index],
                "model_size": history["model_sizes"][index],
                "position_bins": agent.position_bins,
                "velocity_bins": agent.velocity_bins,
                "action_bins": agent.action_bins,
                "action_values": ";".join(map(str, agent.actions.tolist())),
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
                "max_steps": max_steps,
            }
        )
    return rows


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def latest_run_id(evaluation_rows: list[dict[str, str]]) -> str:
    if not evaluation_rows:
        raise ValueError("No hay evaluaciones Dyna-Q disponibles.")
    latest = max(evaluation_rows, key=lambda row: row.get("evaluated_at", ""))
    return latest["run_id"]


def build_comparison_rows(
    *,
    dyna_training_path: str | Path,
    dyna_evaluation_path: str | Path,
    q_learning_path: str | Path,
    q_learning_config_name: str,
    run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Resume un lote Dyna-Q y el baseline Q-Learning ya existente."""
    all_evaluations = load_rows(dyna_evaluation_path)
    selected_run_id = run_id or latest_run_id(all_evaluations)
    evaluations = [row for row in all_evaluations if row["run_id"] == selected_run_id]
    training = [
        row for row in load_rows(dyna_training_path) if row["run_id"] == selected_run_id
    ]
    if not evaluations or not training:
        raise ValueError(
            f"El run_id {selected_run_id!r} no tiene resultados completos."
        )

    latest_evaluation: dict[str, dict[str, str]] = {}
    for row in evaluations:
        experiment_id = row["experiment_id"]
        previous = latest_evaluation.get(experiment_id)
        if previous is None or row["evaluated_at"] > previous["evaluated_at"]:
            latest_evaluation[experiment_id] = row

    training_by_planning: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in training:
        training_by_planning[_as_int(row, "planning_steps")].append(row)
    evaluation_by_planning: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in latest_evaluation.values():
        evaluation_by_planning[_as_int(row, "planning_steps")].append(row)

    comparison: list[dict[str, Any]] = []
    for planning_steps in sorted(evaluation_by_planning):
        eval_rows = evaluation_by_planning[planning_steps]
        train_rows = training_by_planning[planning_steps]
        experiment_ids = {row["experiment_id"] for row in eval_rows}
        recent_training_rows = []
        for experiment_id in experiment_ids:
            experiment_training = sorted(
                (row for row in train_rows if row["experiment_id"] == experiment_id),
                key=lambda row: _as_int(row, "episode"),
            )
            recent_training_rows.extend(experiment_training[-100:])
        training_times = [_as_float(row, "training_time_seconds") for row in eval_rows]
        first = eval_rows[0]
        comparison.append(
            {
                "algorithm": "Dyna-Q",
                "run_id": selected_run_id,
                "experiments": len(experiment_ids),
                "episodes": _as_int(first, "training_episodes"),
                "alpha": first["alpha"],
                "gamma": first["gamma"],
                "epsilon_decay": first["epsilon_decay"],
                "planning_steps": planning_steps,
                "avg_training_reward": fmean(
                    _as_float(row, "reward") for row in recent_training_rows
                ),
                "avg_evaluation_reward": fmean(
                    _as_float(row, "avg_reward") for row in eval_rows
                ),
                "best_reward": max(_as_float(row, "avg_reward") for row in eval_rows),
                "training_time_seconds": fmean(training_times),
                "avg_success_rate": fmean(
                    _as_float(row, "success_rate") for row in eval_rows
                ),
                "avg_episode_length": fmean(
                    _as_float(row, "avg_episode_length") for row in eval_rows
                ),
                "source": Path(dyna_evaluation_path).as_posix(),
            }
        )

    q_rows = [
        row
        for row in load_rows(q_learning_path)
        if row.get("config_name") == q_learning_config_name
        and row.get("episodes")
        and row.get("test_episodes")
    ]
    if q_rows:
        max_episodes = max(_as_int(row, "episodes") for row in q_rows)
        q_rows = [row for row in q_rows if _as_int(row, "episodes") == max_episodes]
        max_test_episodes = max(_as_int(row, "test_episodes") for row in q_rows)
        q_rows = [
            row for row in q_rows if _as_int(row, "test_episodes") == max_test_episodes
        ]
        first = q_rows[0]
        comparison.insert(
            0,
            {
                "algorithm": "Q-Learning",
                "run_id": first.get("search_id", ""),
                "experiments": len(q_rows),
                "episodes": max_episodes,
                "alpha": first["alpha"],
                "gamma": first["gamma"],
                "epsilon_decay": first["epsilon_decay"],
                "planning_steps": 0,
                "avg_training_reward": fmean(
                    _as_float(row, "train_last_100_avg_env_reward") for row in q_rows
                ),
                "avg_evaluation_reward": fmean(
                    _as_float(row, "test_avg_reward") for row in q_rows
                ),
                "best_reward": max(_as_float(row, "test_avg_reward") for row in q_rows),
                "training_time_seconds": fmean(
                    _as_float(row, "training_time_seconds") for row in q_rows
                ),
                "avg_success_rate": fmean(
                    _as_float(row, "test_success_rate") for row in q_rows
                ),
                "avg_episode_length": fmean(
                    _as_float(row, "test_avg_steps") for row in q_rows
                ),
                "source": (
                    f"{Path(q_learning_path).as_posix()} | "
                    f"config={q_learning_config_name}"
                ),
            },
        )
    return selected_run_id, comparison


def write_csv(
    path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(destination)
