"""Genera la comparación y los gráficos desde los CSV de experimentos."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mountaincar-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mountaincar-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dyna_q_config import Q_LEARNING_BASELINE_CONFIG_NAME
from dyna_q_results import (
    COMPARISON_COLUMNS,
    build_comparison_rows,
    load_rows,
    write_csv,
)


def group_episode_values(
    rows: list[dict[str, str]], column: str
) -> dict[int, tuple[list[int], list[float]]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        planning = int(float(row["planning_steps"]))
        episode = int(float(row["episode"]))
        grouped[(planning, episode)].append(float(row[column]))

    result: dict[int, tuple[list[int], list[float]]] = {}
    for planning in sorted({key[0] for key in grouped}):
        episodes = sorted(key[1] for key in grouped if key[0] == planning)
        result[planning] = (
            episodes,
            [fmean(grouped[(planning, episode)]) for episode in episodes],
        )
    return result


def line_plot(
    series: dict[int, tuple[list[int], list[float]]],
    *,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    for planning, (episodes, values) in series.items():
        axis.plot(episodes, values, label=f"planning={planning}", linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel("Episodio")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def comparison_plot(rows: list[dict[str, str]], output: Path) -> None:
    labels = [
        (
            "Q-Learning"
            if row["algorithm"] == "Q-Learning"
            else f"Dyna-Q\np={row['planning_steps']}"
        )
        for row in rows
    ]
    values = [float(row["avg_evaluation_reward"]) for row in rows]
    colors = [
        "#4C78A8" if row["algorithm"] == "Q-Learning" else "#59A14F" for row in rows
    ]
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, values, color=colors, alpha=0.88)
    axis.set_title("Q-Learning vs Dyna-Q: recompensa greedy")
    axis.set_ylabel("Recompensa promedio de evaluación")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def planning_plot(
    rows: list[dict[str, str]], column: str, title: str, ylabel: str, output: Path
) -> None:
    dyna = sorted(
        (row for row in rows if row["algorithm"] == "Dyna-Q"),
        key=lambda row: int(float(row["planning_steps"])),
    )
    planning = [int(float(row["planning_steps"])) for row in dyna]
    values = [float(row[column]) for row in dyna]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(planning, values, marker="o", color="#E15759", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Pasos de planificación")
    axis.set_ylabel(ylabel)
    axis.set_xticks(planning)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=ROOT / "results/dyna_q_training_results.csv",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=ROOT / "results/dyna_q_evaluation_results.csv",
    )
    parser.add_argument(
        "--q-learning-csv",
        type=Path,
        default=ROOT / "results/q_learning_experiments.csv",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=ROOT / "results/comparison_qlearning_dynaq.csv",
    )
    parser.add_argument("--plots-dir", type=Path, default=ROOT / "results/plots")
    args = parser.parse_args()

    selected_run_id, comparison = build_comparison_rows(
        dyna_training_path=args.training_csv,
        dyna_evaluation_path=args.evaluation_csv,
        q_learning_path=args.q_learning_csv,
        q_learning_config_name=Q_LEARNING_BASELINE_CONFIG_NAME,
        run_id=args.run_id,
    )
    write_csv(args.comparison_csv, comparison, COMPARISON_COLUMNS)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    if not any(row["algorithm"] == "Q-Learning" for row in comparison):
        print(
            "Aviso: no se encontró el baseline Q-Learning "
            f"{Q_LEARNING_BASELINE_CONFIG_NAME!r}; la comparación contiene "
            "solamente Dyna-Q."
        )

    training = [
        row for row in load_rows(args.training_csv) if row["run_id"] == selected_run_id
    ]
    comparison_as_strings = [
        {key: str(value) for key, value in row.items()} for row in comparison
    ]
    line_plot(
        group_episode_values(training, "reward"),
        title="Dyna-Q: recompensa por episodio",
        ylabel="Recompensa del ambiente",
        output=args.plots_dir / "dyna_q_reward_per_episode.png",
    )
    line_plot(
        group_episode_values(training, "moving_average_reward"),
        title="Dyna-Q: promedio móvil de recompensa (100 episodios)",
        ylabel="Recompensa promedio móvil",
        output=args.plots_dir / "dyna_q_moving_average_reward.png",
    )
    comparison_plot(
        comparison_as_strings,
        args.plots_dir / "qlearning_vs_dyna_q_evaluation.png",
    )
    planning_plot(
        comparison_as_strings,
        "avg_evaluation_reward",
        "Rendimiento según pasos de planificación",
        "Recompensa promedio de evaluación",
        args.plots_dir / "dyna_q_reward_vs_planning_steps.png",
    )
    planning_plot(
        comparison_as_strings,
        "training_time_seconds",
        "Costo de entrenamiento según pasos de planificación",
        "Tiempo promedio de entrenamiento (s)",
        args.plots_dir / "dyna_q_time_vs_planning_steps.png",
    )
    print(f"Run Dyna-Q resumido: {selected_run_id}")
    print(f"Comparación: {args.comparison_csv}")
    print(f"Gráficos: {args.plots_dir}")


if __name__ == "__main__":
    main()
