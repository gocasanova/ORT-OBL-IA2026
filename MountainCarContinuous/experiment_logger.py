"""Utilidades para registrar experimentos de Q-Learning en un CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


DEFAULT_CSV_PATH = Path("results/q_learning_experiments.csv")
DEFAULT_SEARCH_RUNS_CSV_PATH = Path("results/q_learning_search_runs.csv")
DEFAULT_OVERNIGHT_SUMMARY_CSV_PATH = Path("results/q_learning_overnight_summary.csv")

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "search_id",
    "search_name",
    "config_index",
    "total_configs",
    "config_name",
    "seed",
    "position_bins",
    "velocity_bins",
    "action_bins",
    "alpha",
    "gamma",
    "epsilon_start",
    "epsilon_min",
    "epsilon_decay",
    "episodes",
    "max_steps",
    "train_last_100_avg_reward",
    "train_last_100_avg_steps",
    "train_success_rate_last_100",
    "test_episodes",
    "test_success_rate",
    "test_avg_reward",
    "test_std_reward",
    "test_avg_steps",
    "test_std_steps",
    "training_time_seconds",
    "model_path",
    "notes",
    # Columnas legacy: se conservan para leer el CSV histórico, pero los
    # experimentos nuevos las dejan vacías.
    "reward_shaping",
    "shaping_position_weight",
    "shaping_velocity_weight",
    "q_init",
    "explicit_action_values",
    "final_epsilon",
    "train_last_100_avg_env_reward",
    "train_last_100_avg_learning_reward",
    "train_last_100_avg_position",
    "train_last_100_max_position",
    "test_avg_max_position",
    "test_success_count",
    "profile",
    "run_key",
    "completed_at",
    "estimated_total_time_seconds",
    "average_seconds_per_experiment_so_far",
]

SEARCH_RUN_COLUMNS = [
    "search_id",
    "search_name",
    "search_mode",
    "timestamp_start",
    "timestamp_end",
    "total_search_time_seconds",
    "number_of_configs_planned",
    "number_of_configs_completed",
    "episodes",
    "max_steps",
    "test_episodes",
    "seeds",
    "status",
    "notes",
    "profile",
    "run_key",
    "completed_at",
    "estimated_total_time_seconds",
    "average_seconds_per_experiment_so_far",
]

OVERNIGHT_SUMMARY_COLUMNS = [
    "search_id",
    "profile",
    "config_name",
    "seeds_completed",
    "mean_test_success_rate",
    "std_test_success_rate",
    "mean_test_avg_reward",
    "std_test_avg_reward",
    "mean_test_avg_steps",
    "std_test_avg_steps",
    "mean_test_avg_max_position",
    "mean_training_time_seconds",
    "mean_train_last_100_avg_env_reward",
    "mean_train_success_rate_last_100",
]


def ensure_results_dir(path: str | Path = "results") -> Path:
    """Crea la carpeta de resultados y devuelve su ruta."""
    results_dir = Path(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _mean_or_none(values: Iterable[Any]) -> float | None:
    numeric_values = [float(value) for value in values]
    return fmean(numeric_values) if numeric_values else None


def summarize_training_history(history: dict[str, list[Any]]) -> dict[str, Any]:
    """Resume como máximo los últimos 100 episodios de entrenamiento."""
    env_rewards = history.get("env_rewards", history.get("rewards", []))[-100:]
    learning_rewards = history.get("learning_rewards", env_rewards)[-100:]
    steps = history.get("steps", [])[-100:]
    successes = history.get("successes", [])[-100:]
    avg_positions = history.get("avg_positions", [])[-100:]
    max_positions = history.get("max_positions", [])[-100:]

    success_rate = None
    if successes:
        success_rate = 100.0 * sum(bool(value) for value in successes) / len(successes)

    return {
        "train_last_100_avg_reward": _mean_or_none(env_rewards),
        "train_last_100_avg_steps": _mean_or_none(steps),
        "train_success_rate_last_100": success_rate,
        "train_last_100_avg_env_reward": _mean_or_none(env_rewards),
        "train_last_100_avg_learning_reward": _mean_or_none(learning_rewards),
        "train_last_100_avg_position": _mean_or_none(avg_positions),
        "train_last_100_max_position": (
            max(float(value) for value in max_positions) if max_positions else None
        ),
    }


def summarize_test_results(test_results: dict[str, Any]) -> dict[str, Any]:
    """Calcula medias, desvíos poblacionales y tasa de éxito del test."""
    rewards = [float(value) for value in test_results.get("rewards", [])]
    steps = [float(value) for value in test_results.get("steps", [])]
    successes = test_results.get("successes", [])
    max_positions = [float(value) for value in test_results.get("max_positions", [])]

    success_rate = test_results.get("success_rate")
    if success_rate is None and successes:
        success_rate = 100.0 * sum(bool(value) for value in successes) / len(successes)

    return {
        "test_success_rate": success_rate,
        "test_avg_reward": fmean(rewards) if rewards else None,
        "test_std_reward": pstdev(rewards) if rewards else None,
        "test_avg_steps": fmean(steps) if steps else None,
        "test_std_steps": pstdev(steps) if steps else None,
        "test_avg_max_position": fmean(max_positions) if max_positions else None,
        "test_success_count": test_results.get(
            "success_count", sum(bool(value) for value in successes)
        ),
    }


def _ensure_csv_schema(csv_path: Path, columns: list[str]) -> None:
    """Migra de forma atómica un esquema anterior que solo carece de columnas."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames
        if header == columns:
            return
        if header is None or not set(header).issubset(columns):
            raise ValueError(
                f"El encabezado de {csv_path} contiene columnas incompatibles."
            )
        rows = list(reader)

    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for old_row in rows:
            writer.writerow({column: old_row.get(column, "") for column in columns})
    temporary_path.replace(csv_path)


def _append_result(
    result_dict: dict[str, Any], destination: Path, columns: list[str]
) -> dict[str, Any]:
    ensure_results_dir(destination.parent)
    _ensure_csv_schema(destination, columns)
    has_content = destination.exists() and destination.stat().st_size > 0

    row = {
        column: "" if result_dict.get(column) is None else result_dict.get(column, "")
        for column in columns
    }
    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if not has_content:
            writer.writeheader()
        writer.writerow(row)
    return row


def _load_results(source: Path, columns: list[str]) -> list[dict[str, str]]:
    if not source.exists() or source.stat().st_size == 0:
        return []
    _ensure_csv_schema(source, columns)
    with source.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != columns:
            # Es una defensa adicional ante modificaciones concurrentes/manuales.
            raise ValueError(
                f"El encabezado de {source} no coincide con el esquema esperado."
            )
        return list(reader)


def append_experiment_result(
    result_dict: dict[str, Any],
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Any]:
    """Agrega una fila sin reemplazar resultados ni repetir el encabezado."""
    return _append_result(result_dict, Path(csv_path), EXPERIMENT_COLUMNS)


def load_experiment_results(
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> list[dict[str, str]]:
    """Carga las filas en el mismo orden en que fueron registradas."""
    return _load_results(Path(csv_path), EXPERIMENT_COLUMNS)


def append_search_run_result(
    result_dict: dict[str, Any],
    csv_path: str | Path = DEFAULT_SEARCH_RUNS_CSV_PATH,
) -> dict[str, Any]:
    """Agrega el resumen de una búsqueda completa o interrumpida."""
    return _append_result(result_dict, Path(csv_path), SEARCH_RUN_COLUMNS)


def load_search_run_results(
    csv_path: str | Path = DEFAULT_SEARCH_RUNS_CSV_PATH,
) -> list[dict[str, str]]:
    """Carga los resúmenes de búsquedas en orden de ejecución."""
    return _load_results(Path(csv_path), SEARCH_RUN_COLUMNS)


def _numeric_column(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column) not in {None, ""}]


def _mean_column(rows: list[dict[str, str]], column: str) -> float | None:
    values = _numeric_column(rows, column)
    return fmean(values) if values else None


def _std_column(rows: list[dict[str, str]], column: str) -> float | None:
    values = _numeric_column(rows, column)
    return pstdev(values) if values else None


def update_overnight_summary(
    experiments_csv_path: str | Path = DEFAULT_CSV_PATH,
    summary_csv_path: str | Path = DEFAULT_OVERNIGHT_SUMMARY_CSV_PATH,
) -> list[dict[str, Any]]:
    """Regenera el resumen descriptivo sin ordenar ni seleccionar resultados."""
    experiments = load_experiment_results(experiments_csv_path)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in experiments:
        if row.get("profile") != "overnight":
            continue
        key = (row["search_id"], row["profile"], row["config_name"])
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (search_id, profile, config_name), rows in groups.items():
        summary_rows.append(
            {
                "search_id": search_id,
                "profile": profile,
                "config_name": config_name,
                "seeds_completed": len({row["seed"] for row in rows}),
                "mean_test_success_rate": _mean_column(rows, "test_success_rate"),
                "std_test_success_rate": _std_column(rows, "test_success_rate"),
                "mean_test_avg_reward": _mean_column(rows, "test_avg_reward"),
                "std_test_avg_reward": _std_column(rows, "test_avg_reward"),
                "mean_test_avg_steps": _mean_column(rows, "test_avg_steps"),
                "std_test_avg_steps": _std_column(rows, "test_avg_steps"),
                "mean_test_avg_max_position": _mean_column(
                    rows, "test_avg_max_position"
                ),
                "mean_training_time_seconds": _mean_column(
                    rows, "training_time_seconds"
                ),
                "mean_train_last_100_avg_env_reward": _mean_column(
                    rows, "train_last_100_avg_env_reward"
                ),
                "mean_train_success_rate_last_100": _mean_column(
                    rows, "train_success_rate_last_100"
                ),
            }
        )

    destination = Path(summary_csv_path)
    ensure_results_dir(destination.parent)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OVERNIGHT_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary_path.replace(destination)
    return summary_rows


def load_overnight_summary(
    csv_path: str | Path = DEFAULT_OVERNIGHT_SUMMARY_CSV_PATH,
) -> list[dict[str, str]]:
    """Carga el resumen overnight derivado, si ya fue generado."""
    source = Path(csv_path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    with source.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != OVERNIGHT_SUMMARY_COLUMNS:
            raise ValueError(f"El encabezado de {source} no coincide con el esperado.")
        return list(reader)
