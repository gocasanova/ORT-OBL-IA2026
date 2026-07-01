"""Búsqueda reproducible de hiperparámetros para Q-Learning.

El script ejecuta y registra configuraciones; no las ordena ni elige una.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import gymnasium as gym

from experiment_logger import (
    DEFAULT_CSV_PATH,
    DEFAULT_OVERNIGHT_SUMMARY_CSV_PATH,
    DEFAULT_SEARCH_RUNS_CSV_PATH,
    append_experiment_result,
    append_search_run_result,
    summarize_test_results,
    summarize_training_history,
    update_overnight_summary,
)
from q_learning_agent import QLearningAgent


# Parámetros globales fáciles de modificar.
SEARCH_NAME = "qlearning_grid_search_001"
SEARCH_MODE = "grid"  # "grid" o "manual"
EPISODES = 3000
MAX_STEPS = 999
TEST_EPISODES = 20
SEEDS = [42]
DRY_RUN = False
SEARCH_NOTES = ""
VERBOSE = True
PROGRESS_INTERVAL = 500

OVERNIGHT_SEARCH_NAME = "qlearning_overnight_001"
OVERNIGHT_EPISODES = 20_000
OVERNIGHT_MAX_STEPS = 999
OVERNIGHT_TEST_EPISODES = 100
OVERNIGHT_SEEDS = [42, 123, 999, 2026, 777]

# Grid deliberadamente chico: 2 * 2 * 2 = 8 configuraciones por semilla.
POSITION_BINS_VALUES = [30, 40]
VELOCITY_BINS_VALUES = [30, 40]
ACTION_BINS_VALUES = [7, 9]
ALPHA_VALUES = [0.1]
GAMMA_VALUES = [0.99]
EPSILON_DECAY_VALUES = [0.995]

MANUAL_CONFIGS = [
    {
        "config_name": "shaped_long_exploration_10k",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "shaped_fine_actions_10k",
        "position_bins": 50,
        "velocity_bins": 50,
        "action_bins": 15,
        "alpha": 0.05,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "shaped_no_zero_action_10k",
        "position_bins": 40,
        "velocity_bins": 40,
        "explicit_action_values": [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0],
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "optimistic_no_shaping_10k",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "high_gamma_shaped_10k",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.999,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9997,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
]

OVERNIGHT_CONFIGS = [
    {
        "config_name": "optimistic_q3_a11_gamma995",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 3.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_q5_a11_gamma995",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_q10_a11_gamma995",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 10.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_q5_a15_gamma995",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 15,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_q5_fine_50x50",
        "position_bins": 50,
        "velocity_bins": 50,
        "action_bins": 15,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "shaped_q1_a11_gamma995",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "shaped_q1_a15_gamma995",
        "position_bins": 50,
        "velocity_bins": 50,
        "action_bins": 15,
        "alpha": 0.05,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "shaped_high_gamma",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.1,
        "gamma": 0.999,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9997,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "no_zero_q1_shaped",
        "position_bins": 40,
        "velocity_bins": 40,
        "explicit_action_values": [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0],
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 1.0,
        "reward_shaping": "potential",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
    },
    {
        "config_name": "no_zero_q5_no_shaping",
        "position_bins": 40,
        "velocity_bins": 40,
        "explicit_action_values": [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0],
        "alpha": 0.1,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_alpha005",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.05,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
    {
        "config_name": "optimistic_alpha015",
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 11,
        "alpha": 0.15,
        "gamma": 0.995,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "q_init": 5.0,
        "reward_shaping": "none",
    },
]

ENV_ID = "MountainCarContinuous-v0"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _label(value: Any) -> str:
    return str(value).replace(".", "p")


def build_grid_configs() -> list[dict[str, Any]]:
    """Genera el producto cartesiano del grid configurado arriba."""
    configs = []
    combinations = itertools.product(
        POSITION_BINS_VALUES,
        VELOCITY_BINS_VALUES,
        ACTION_BINS_VALUES,
        ALPHA_VALUES,
        GAMMA_VALUES,
        EPSILON_DECAY_VALUES,
    )
    for position, velocity, actions, alpha, gamma, decay in combinations:
        name = (
            f"grid_p{position}_v{velocity}_a{actions}"
            f"_alpha{_label(alpha)}_gamma{_label(gamma)}_decay{_label(decay)}"
        )
        configs.append(
            {
                "config_name": name,
                "position_bins": position,
                "velocity_bins": velocity,
                "action_bins": actions,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": 1.0,
                "epsilon_min": 0.05,
                "epsilon_decay": decay,
            }
        )
    return configs


def build_search_tasks(
    mode: str, seeds: list[int], profile: str = ""
) -> list[tuple[dict[str, Any], int]]:
    """Combina cada configuración con cada semilla solicitada."""
    if profile == "overnight":
        configs = [dict(config) for config in OVERNIGHT_CONFIGS]
    elif mode == "grid":
        configs = build_grid_configs()
    elif mode == "manual":
        configs = [dict(config) for config in MANUAL_CONFIGS]
    else:
        raise ValueError("SEARCH_MODE debe ser 'grid' o 'manual'.")
    return [(dict(config), seed) for config in configs for seed in seeds]


def build_run_key(
    config: dict[str, Any],
    seed: int,
    episodes: int,
    max_steps: int,
    test_episodes: int,
) -> str:
    """Crea una clave estable a partir de configuración y parámetros de corrida."""
    payload = {
        "config": config,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "test_episodes": test_episodes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return (
        f"{config['config_name']}|seed={seed}|episodes={episodes}|"
        f"max_steps={max_steps}|test={test_episodes}|params={digest}"
    )


def _read_experiments_without_migration() -> list[dict[str, str]]:
    """Lectura sin efectos laterales, usada por dry-run y resume."""
    if not DEFAULT_CSV_PATH.exists() or DEFAULT_CSV_PATH.stat().st_size == 0:
        return []
    with DEFAULT_CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _estimate_seconds_per_experiment(episodes: int) -> float | None:
    historical_rows = _read_experiments_without_migration()
    rates = []
    for row in historical_rows:
        try:
            previous_episodes = int(row.get("episodes", ""))
            training_time = float(row.get("training_time_seconds", ""))
        except (TypeError, ValueError):
            continue
        if previous_episodes >= 1000 and training_time > 0:
            rates.append(training_time / previous_episodes)
    if not rates:
        return None
    return median(rates) * episodes


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "sin datos previos"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {remaining_seconds:02d}s"


def _prepare_agent_config(
    config: dict[str, Any], seed: int
) -> tuple[str, str, dict[str, Any], float]:
    config = dict(config)
    try:
        config_name = str(config.pop("config_name"))
    except KeyError as error:
        raise ValueError("Cada configuración debe incluir config_name.") from error
    config_notes = str(config.pop("notes", ""))
    epsilon_start = float(config.pop("epsilon_start", config.pop("epsilon", 1.0)))

    defaults = {
        "position_bins": 40,
        "velocity_bins": 40,
        "action_bins": 9,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "reward_shaping": "none",
        "shaping_position_weight": 1.0,
        "shaping_velocity_weight": 0.5,
        "q_init": 0.0,
        "explicit_action_values": None,
    }
    unknown_keys = set(config) - set(defaults)
    if unknown_keys:
        raise ValueError(f"Parámetros desconocidos: {unknown_keys}")
    defaults.update(config)
    agent_config = {
        **defaults,
        "epsilon": epsilon_start,
        "seed": seed,
    }
    return config_name, config_notes, agent_config, epsilon_start


def run_configuration(
    config: dict[str, Any],
    seed: int,
    search_id: str,
    search_name: str,
    config_index: int,
    total_configs: int,
    episodes: int,
    max_steps: int,
    test_episodes: int,
    search_notes: str = "",
    verbose: bool = False,
    progress_interval: int = 500,
    profile: str = "",
    run_key: str = "",
    search_start_time: float | None = None,
    completed_before: int = 0,
    execution_count: int = 1,
) -> dict[str, Any]:
    """Entrena, testea, guarda y registra inmediatamente una configuración."""
    config_name, config_notes, agent_config, epsilon_start = _prepare_agent_config(
        config, seed
    )
    experiment_id = uuid.uuid4().hex[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", config_name).strip("_")
    safe_name = safe_name or "experiment"
    model_path = Path("models") / (
        f"q_learning_{safe_name}_seed{seed}_{experiment_id}.pkl"
    )

    env = gym.make(ENV_ID, render_mode="rgb_array")
    agent = QLearningAgent(**agent_config)
    try:
        training_start = time.perf_counter()
        history = agent.train_agent(
            env,
            episodes=episodes,
            max_steps=max_steps,
            verbose=verbose,
            progress_interval=progress_interval,
        )
        training_time = time.perf_counter() - training_start
        test_results = agent.test_agent(
            env, episodes=test_episodes, max_steps=max_steps
        )
    finally:
        env.close()

    notes = " | ".join(value for value in (search_notes, config_notes) if value)
    completed_at = _timestamp()
    agent.save(
        model_path,
        metadata={
            "experiment_id": experiment_id,
            "search_id": search_id,
            "search_name": search_name,
            "config_index": config_index,
            "total_configs": total_configs,
            "config_name": config_name,
            "seed": seed,
            "episodes": episodes,
            "max_steps": max_steps,
            "test_episodes": test_episodes,
            "notes": notes,
            "profile": profile,
            "run_key": run_key,
            "completed_at": completed_at,
        },
    )
    elapsed_so_far = (
        time.perf_counter() - search_start_time
        if search_start_time is not None
        else training_time
    )
    average_seconds = elapsed_so_far / (completed_before + 1)
    estimated_total_time = average_seconds * execution_count
    result = {
        "experiment_id": experiment_id,
        "timestamp": _timestamp(),
        "search_id": search_id,
        "search_name": search_name,
        "config_index": config_index,
        "total_configs": total_configs,
        "config_name": config_name,
        "seed": seed,
        "position_bins": agent_config["position_bins"],
        "velocity_bins": agent_config["velocity_bins"],
        "action_bins": agent.action_bins,
        "alpha": agent_config["alpha"],
        "gamma": agent_config["gamma"],
        "epsilon_start": epsilon_start,
        "epsilon_min": agent_config["epsilon_min"],
        "epsilon_decay": agent_config["epsilon_decay"],
        "episodes": episodes,
        "max_steps": max_steps,
        **summarize_training_history(history),
        "test_episodes": test_episodes,
        **summarize_test_results(test_results),
        "training_time_seconds": round(training_time, 3),
        "model_path": model_path.as_posix(),
        "notes": notes,
        "reward_shaping": agent.reward_shaping,
        "shaping_position_weight": agent.shaping_position_weight,
        "shaping_velocity_weight": agent.shaping_velocity_weight,
        "q_init": agent.q_init,
        "explicit_action_values": (
            json.dumps(agent.explicit_action_values)
            if agent.explicit_action_values is not None
            else ""
        ),
        "final_epsilon": agent.epsilon,
        "profile": profile,
        "run_key": run_key,
        "completed_at": completed_at,
        "estimated_total_time_seconds": round(estimated_total_time, 3),
        "average_seconds_per_experiment_so_far": round(average_seconds, 3),
    }
    return append_experiment_result(result)


def _print_dry_run(
    search_name: str,
    mode: str,
    tasks: list[tuple[dict[str, Any], int]],
    tasks_to_execute: list[tuple[dict[str, Any], int]],
    episodes: int,
    max_steps: int,
    test_episodes: int,
    profile: str,
    skipped: int,
) -> None:
    print("DRY RUN: no se entrenará ni se modificarán archivos.")
    print(f"Búsqueda: {search_name} | modo: {mode} | profile: {profile or '-'}")
    print(f"Configuraciones: {len({config['config_name'] for config, _ in tasks})}")
    print(f"Seeds: {len({seed for _, seed in tasks})}")
    print(f"Entrenamientos totales: {len(tasks)}")
    print(f"Ya completados que resume omitiría: {skipped}")
    print(f"Entrenamientos a ejecutar: {len(tasks_to_execute)}")
    print(
        f"Parámetros globales: episodes={episodes}, max_steps={max_steps}, "
        f"test_episodes={test_episodes}"
    )
    estimated_seconds = _estimate_seconds_per_experiment(episodes)
    total_estimate = (
        None if estimated_seconds is None else estimated_seconds * len(tasks_to_execute)
    )
    print(f"Estimación por entrenamiento: {_format_duration(estimated_seconds)}")
    print(f"Estimación total aproximada: {_format_duration(total_estimate)}")
    for index, (config, seed) in enumerate(tasks_to_execute, start=1):
        print(f"[{index}/{len(tasks_to_execute)}] seed={seed} | {config}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--profile", choices=("overnight",), default="")
    parser.add_argument("--mode", choices=("grid", "manual"), default=None)
    parser.add_argument("--search-name", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita tareas para pruebas cortas; no altera el grid definido.",
    )
    parser.add_argument("--notes", default=SEARCH_NOTES)
    repetition = parser.add_mutually_exclusive_group()
    repetition.add_argument("--resume", action="store_true")
    repetition.add_argument("--rerun", action="store_true")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", dest="verbose", action="store_true")
    verbosity.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=VERBOSE)
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL)
    args = parser.parse_args(argv)

    if args.profile == "overnight":
        args.mode = args.mode or "manual"
        args.search_name = args.search_name or OVERNIGHT_SEARCH_NAME
        args.episodes = OVERNIGHT_EPISODES if args.episodes is None else args.episodes
        args.max_steps = (
            OVERNIGHT_MAX_STEPS if args.max_steps is None else args.max_steps
        )
        args.test_episodes = (
            OVERNIGHT_TEST_EPISODES
            if args.test_episodes is None
            else args.test_episodes
        )
        args.seeds = OVERNIGHT_SEEDS if args.seeds is None else args.seeds
    else:
        args.mode = args.mode or SEARCH_MODE
        args.search_name = args.search_name or SEARCH_NAME
        args.episodes = EPISODES if args.episodes is None else args.episodes
        args.max_steps = MAX_STEPS if args.max_steps is None else args.max_steps
        args.test_episodes = (
            TEST_EPISODES if args.test_episodes is None else args.test_episodes
        )
        args.seeds = SEEDS if args.seeds is None else args.seeds
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.episodes <= 0
        or args.max_steps <= 0
        or args.test_episodes <= 0
        or args.progress_interval <= 0
    ):
        raise ValueError(
            "episodes, max_steps, test_episodes y progress_interval deben ser positivos."
        )
    if not args.seeds:
        raise ValueError("Debe indicarse al menos una semilla.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit debe ser positivo.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        tasks = build_search_tasks(args.mode, args.seeds, args.profile)
        if args.limit is not None:
            tasks = tasks[: args.limit]
    except ValueError as error:
        print(f"Configuración inválida: {error}", file=sys.stderr)
        return 2

    task_entries = []
    for index, (config, seed) in enumerate(tasks, start=1):
        run_key = build_run_key(
            config, seed, args.episodes, args.max_steps, args.test_episodes
        )
        task_entries.append((index, config, seed, run_key))

    completed_run_keys = set()
    if args.resume:
        for row in _read_experiments_without_migration():
            model_path = row.get("model_path", "")
            if row.get("run_key") and model_path and Path(model_path).is_file():
                completed_run_keys.add(row["run_key"])
    pending_entries = [
        entry for entry in task_entries if entry[3] not in completed_run_keys
    ]
    skipped = len(task_entries) - len(pending_entries)

    if args.dry_run:
        _print_dry_run(
            args.search_name,
            args.mode,
            tasks,
            [(config, seed) for _, config, seed, _ in pending_entries],
            args.episodes,
            args.max_steps,
            args.test_episodes,
            args.profile,
            skipped,
        )
        return 0

    search_id = uuid.uuid4().hex[:12]
    timestamp_start = _timestamp()
    search_start = time.perf_counter()
    completed = 0
    status = "completed"
    failure_note = ""
    estimated_seconds = _estimate_seconds_per_experiment(args.episodes)
    estimated_total_time = (
        estimated_seconds * len(pending_entries)
        if estimated_seconds is not None
        else 0.0
    )
    average_seconds = 0.0

    print(f"Búsqueda iniciada: {args.search_name}")
    print(f"search_id: {search_id}")
    print(f"profile: {args.profile or '-'}")
    print(f"Entrenamientos planificados: {len(tasks)}")
    print(f"Entrenamientos omitidos por resume: {skipped}")
    print(f"Entrenamientos a ejecutar: {len(pending_entries)}")
    print(f"Estimación inicial: {_format_duration(estimated_total_time or None)}")

    try:
        for execution_index, (index, config, seed, run_key) in enumerate(
            pending_entries, start=1
        ):
            print("\n" + "-" * 72)
            print(
                f"Experimento {execution_index}/{len(pending_entries)} | "
                f"config {index}/{len(tasks)} | seed={seed}"
            )
            print(config)
            result = run_configuration(
                config=config,
                seed=seed,
                search_id=search_id,
                search_name=args.search_name,
                config_index=index,
                total_configs=len(tasks),
                episodes=args.episodes,
                max_steps=args.max_steps,
                test_episodes=args.test_episodes,
                search_notes=args.notes,
                verbose=args.verbose,
                progress_interval=args.progress_interval,
                profile=args.profile,
                run_key=run_key,
                search_start_time=search_start,
                completed_before=completed,
                execution_count=len(pending_entries),
            )
            completed += 1
            average_seconds = float(result["average_seconds_per_experiment_so_far"])
            estimated_total_time = float(result["estimated_total_time_seconds"])
            elapsed = time.perf_counter() - search_start
            remaining = average_seconds * (len(pending_entries) - completed)
            if args.profile == "overnight":
                update_overnight_summary()
            print(f"Tiempo de entrenamiento: {result['training_time_seconds']} s")
            print(
                f"Test: éxito={result['test_success_rate']}% | "
                f"recompensa media={result['test_avg_reward']} | "
                f"pasos medios={result['test_avg_steps']}"
            )
            print(f"Máxima posición media de test: {result['test_avg_max_position']}")
            print(f"Modelo: {result['model_path']}")
            print(
                f"Progreso: {completed}/{len(pending_entries)} ejecutados | "
                f"acumulado={_format_duration(elapsed)} | "
                f"ETA restante={_format_duration(remaining)}"
            )
    except KeyboardInterrupt:
        status = "interrupted"
        failure_note = "Interrumpida por el usuario."
        print("\nBúsqueda interrumpida; las corridas terminadas quedaron guardadas.")
    except Exception as error:  # El resumen debe guardarse también ante un fallo.
        status = "failed"
        failure_note = f"{type(error).__name__}: {error}"
        print(f"\nLa búsqueda falló: {failure_note}", file=sys.stderr)

    timestamp_end = _timestamp()
    total_time = time.perf_counter() - search_start
    if completed:
        average_seconds = total_time / completed
        estimated_total_time = average_seconds * len(pending_entries)
    notes = " | ".join(value for value in (args.notes, failure_note) if value)
    append_search_run_result(
        {
            "search_id": search_id,
            "search_name": args.search_name,
            "search_mode": args.profile or args.mode,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "total_search_time_seconds": round(total_time, 3),
            "number_of_configs_planned": len(tasks),
            "number_of_configs_completed": completed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "test_episodes": args.test_episodes,
            "seeds": json.dumps(args.seeds),
            "status": status,
            "notes": notes,
            "profile": args.profile,
            "run_key": "",
            "completed_at": timestamp_end,
            "estimated_total_time_seconds": round(estimated_total_time, 3),
            "average_seconds_per_experiment_so_far": round(average_seconds, 3),
        }
    )
    if args.profile == "overnight":
        update_overnight_summary()

    print("\n" + "=" * 72)
    print(f"Búsqueda terminada con estado: {status}")
    print(f"Tiempo total: {total_time:.3f} s")
    print(f"Entrenamientos completados ahora: {completed}/{len(pending_entries)}")
    print(f"Omitidos por resume: {skipped}")
    print(f"Experimentos: {DEFAULT_CSV_PATH}")
    print(f"Búsquedas: {DEFAULT_SEARCH_RUNS_CSV_PATH}")
    if args.profile == "overnight":
        print(f"Resumen overnight: {DEFAULT_OVERNIGHT_SUMMARY_CSV_PATH}")
    if status == "completed":
        return 0
    if status == "interrupted":
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
