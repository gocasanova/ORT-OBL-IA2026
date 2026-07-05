"""Reproducible command-line experiments for the MATE Isolation agents.

Run a quick validation with::

    poetry run python experiments_mate.py --suite smoke --games 1 --depths 1

The default ``core`` suite covers algorithms, depths, and evaluation functions.
The ``full`` suite additionally runs a round-robin between all evaluations.
The ``final`` suite runs the missing matchups needed to complete the final
balanced-depth tournament, including the repository's Stratagem agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from board import Board
from mate_agents import (
    AlphaBetaAgent,
    BaseSearchAgent,
    ExpectimaxAgent,
    MinimaxAgent,
)
from mate_evaluations import EVALUATION_FUNCTIONS, get_evaluation
from random_agent import RandomAgent
from stratagem import Stratagem


@dataclass(frozen=True)
class AgentSpec:
    """Serializable description used to construct a fresh agent per game."""

    algorithm: str
    evaluation: str = "none"
    depth: int = 0

    @property
    def label(self) -> str:
        if self.algorithm in ("Random", "Stratagem"):
            return self.algorithm
        return f"{self.algorithm}[{self.evaluation},d={self.depth}]"

    def build(self, player: int, seed: int):
        if self.algorithm == "Random":
            return RandomAgent(player, seed=seed)
        if self.algorithm == "Stratagem":
            return Stratagem(player)
        classes = {
            "Minimax": MinimaxAgent,
            "AlphaBeta": AlphaBetaAgent,
            "Expectimax": ExpectimaxAgent,
        }
        return classes[self.algorithm](
            player=player,
            depth=self.depth,
            evaluation_function=get_evaluation(self.evaluation),
            seed=seed,
        )


@dataclass
class ParticipantStats:
    wins: int = 0
    decision_times: List[float] = field(default_factory=list)
    expanded_nodes: List[int] = field(default_factory=list)
    pruned_branches: List[int] = field(default_factory=list)


def _search_metrics(agent, elapsed: float) -> Tuple[float, int, int]:
    if isinstance(agent, BaseSearchAgent):
        metrics = agent.last_metrics
        return (
            metrics.execution_time,
            metrics.expanded_nodes,
            metrics.pruned_branches,
        )
    return elapsed, 0, 0


def play_game(
    first_spec: AgentSpec,
    second_spec: AgentSpec,
    first_role: str,
    second_role: str,
    seed: int,
    board_size: Tuple[int, int],
) -> Tuple[str, int, Dict[str, List[Tuple[float, int, int]]]]:
    """Play one complete game and return winning role, length, and move metrics."""
    random.seed(seed)  # Board uses Python's module-level random generator.
    board = Board(board_size)
    agents = {
        1: first_spec.build(1, seed + 1),
        2: second_spec.build(2, seed + 2),
    }
    roles = {1: first_role, 2: second_role}
    move_metrics: Dict[str, List[Tuple[float, int, int]]] = {"A": [], "B": []}
    current_player = 1
    game_length = 0

    while True:
        done, winner = board.is_end(current_player)
        if done:
            return roles[winner], game_length, move_metrics

        agent = agents[current_player]
        legal_actions = board.get_possible_actions(current_player)
        started = time.perf_counter()
        action = agent.next_action(board)
        elapsed = time.perf_counter() - started
        if action not in legal_actions:
            raise RuntimeError(
                f"{agent.__class__.__name__} returned illegal action {action}"
            )
        if not board.play(action, current_player):
            raise RuntimeError(f"Board rejected generated legal action {action}")

        move_metrics[roles[current_player]].append(_search_metrics(agent, elapsed))
        game_length += 1
        current_player = 2 if current_player == 1 else 1


def run_matchup(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    games: int,
    seed: int,
    board_size: Tuple[int, int],
) -> List[dict]:
    """Run a matchup, alternating which participant moves first."""
    stats = {"A": ParticipantStats(), "B": ParticipantStats()}
    game_lengths: List[int] = []
    for game_index in range(games):
        game_seed = seed + game_index * 1009
        if game_index % 2 == 0:
            first, second, first_role, second_role = spec_a, spec_b, "A", "B"
        else:
            first, second, first_role, second_role = spec_b, spec_a, "B", "A"
        winner_role, length, metrics = play_game(
            first, second, first_role, second_role, game_seed, board_size
        )
        stats[winner_role].wins += 1
        game_lengths.append(length)
        for role in ("A", "B"):
            for decision_time, expanded, pruned in metrics[role]:
                stats[role].decision_times.append(decision_time)
                stats[role].expanded_nodes.append(expanded)
                stats[role].pruned_branches.append(pruned)

    rows = []
    matchup_id = f"{spec_a.label}_vs_{spec_b.label}"
    for role, spec, opponent in (("A", spec_a, spec_b), ("B", spec_b, spec_a)):
        role_stats = stats[role]
        rows.append(
            {
                "matchup": matchup_id,
                "agent_name": spec.algorithm,
                "opponent_name": opponent.algorithm,
                "evaluation_function": spec.evaluation,
                "search_depth": spec.depth,
                "games": games,
                "wins": role_stats.wins,
                "losses": games - role_stats.wins,
                "win_rate": role_stats.wins / games,
                "average_game_length": mean(game_lengths),
                "average_decision_time_seconds": mean(role_stats.decision_times)
                if role_stats.decision_times
                else 0.0,
                "average_expanded_nodes": mean(role_stats.expanded_nodes)
                if role_stats.expanded_nodes
                else 0.0,
                "average_pruned_branches": mean(role_stats.pruned_branches)
                if role_stats.pruned_branches
                else 0.0,
            }
        )
    return rows


def _run_scheduled_matchup(task):
    """Run one numbered matchup; kept top-level for multiprocessing."""
    index, spec_a, spec_b, games, seed, board_size = task
    return index, run_matchup(spec_a, spec_b, games, seed, board_size)


def build_matchups(suite: str, depths: Sequence[int]) -> List[Tuple[AgentSpec, AgentSpec]]:
    """Build the requested experiment matrix without duplicate configurations."""
    random_spec = AgentSpec("Random")
    if suite == "final":
        depth = max(depths)
        minimax = AgentSpec("Minimax", "balanced", depth)
        alpha_beta = AgentSpec("AlphaBeta", "balanced", depth)
        expectimax = AgentSpec("Expectimax", "balanced", depth)
        stratagem = AgentSpec("Stratagem")
        return [
            (minimax, expectimax),
            (minimax, stratagem),
            (alpha_beta, stratagem),
            (expectimax, stratagem),
        ]

    matchups: List[Tuple[AgentSpec, AgentSpec]] = []
    active_depths = depths[:1] if suite == "smoke" else depths
    for depth in active_depths:
        minimax = AgentSpec("Minimax", "balanced", depth)
        alpha_beta = AgentSpec("AlphaBeta", "balanced", depth)
        expectimax = AgentSpec("Expectimax", "balanced", depth)
        matchups.extend(
            [
                (minimax, random_spec),
                (alpha_beta, random_spec),
                (expectimax, random_spec),
                (alpha_beta, minimax),
                (alpha_beta, expectimax),
            ]
        )

    if suite in ("core", "full"):
        evaluation_depth = max(depths)
        evaluation_specs = [
            AgentSpec("AlphaBeta", evaluation, evaluation_depth)
            for evaluation in EVALUATION_FUNCTIONS
        ]
        matchups.extend((spec, random_spec) for spec in evaluation_specs)
        if suite == "full":
            matchups.extend(combinations(evaluation_specs, 2))

    unique = []
    seen = set()
    for matchup in matchups:
        key = (matchup[0], matchup[1])
        if key not in seen:
            seen.add(key)
            unique.append(matchup)
    return unique


def save_results(rows: Iterable[dict], output_dir: Path) -> Tuple[Path, Path]:
    rows = list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mate_experiments.csv"
    json_path = output_dir / "mate_experiments.json"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(rows, output, indent=2)
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("smoke", "core", "full", "final"), default="core"
    )
    parser.add_argument("--games", type=int, default=2, help="games per matchup")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="matchups to execute concurrently",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.games < 1 or any(depth < 1 for depth in args.depths) or args.workers < 1:
        parser.error("games, workers, and all depths must be positive")
    if args.rows < 2 or args.cols < 2:
        parser.error("board dimensions must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    matchups = build_matchups(args.suite, args.depths)
    print(
        f"Running {len(matchups)} matchups x {args.games} games "
        f"on {args.rows}x{args.cols} with {args.workers} worker(s)...",
        flush=True,
    )
    tasks = [
        (
            index,
            spec_a,
            spec_b,
            args.games,
            args.seed + index * 100_003,
            (args.rows, args.cols),
        )
        for index, (spec_a, spec_b) in enumerate(matchups, start=1)
    ]
    rows_by_index = {}
    if args.workers == 1:
        for task in tasks:
            index, rows = _run_scheduled_matchup(task)
            rows_by_index[index] = rows
            print(f"[{index}/{len(matchups)}] completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_scheduled_matchup, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                index, rows = future.result()
                rows_by_index[index] = rows
                print(f"[{index}/{len(matchups)}] completed", flush=True)

    all_rows = [
        row
        for index in range(1, len(matchups) + 1)
        for row in rows_by_index[index]
    ]
    csv_path, json_path = save_results(all_rows, args.output_dir)
    print(f"Saved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
