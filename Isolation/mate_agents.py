"""Minimax, Alpha-Beta, and Expectimax agents for the Isolation environment."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

from agent import Agent
from board import Board
from mate_evaluations import EvaluationFunction, balanced_eval

Position = Tuple[int, int]
Action = Tuple[int, Position]


@dataclass
class SearchMetrics:
    """Metrics produced by the most recent call to ``next_action``."""

    expanded_nodes: int = 0
    pruned_branches: int = 0
    execution_time: float = 0.0
    chosen_action: Optional[Action] = None
    search_depth: int = 0


class BaseSearchAgent(Agent):
    """Shared configuration and safe simulation for game-tree agents."""

    def __init__(
        self,
        player: int = 1,
        depth: int = 2,
        evaluation_function: EvaluationFunction = balanced_eval,
        seed: Optional[int] = None,
        tie_breaking: str = "first",
    ) -> None:
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if tie_breaking not in ("first", "random"):
            raise ValueError("tie_breaking must be 'first' or 'random'")
        super().__init__(player)
        self.depth = depth
        self.evaluation_function = evaluation_function
        self.tie_breaking = tie_breaking
        self.random = random.Random(seed)
        self.last_metrics = SearchMetrics(search_depth=depth)

    @property
    def metrics(self) -> dict:
        """Dictionary form convenient for experiments and JSON output."""
        return asdict(self.last_metrics)

    def heuristic_utility(self, board: Board) -> float:
        return self.evaluation_function(board, self.player)

    @staticmethod
    def _opponent(player: int) -> int:
        return 2 if player == 1 else 1

    @staticmethod
    def _successor(board: Board, action: Action, player: int) -> Board:
        child = board.clone()
        if not child.play(action, player):
            raise ValueError(f"Board generated an invalid action: {action}")
        return child

    def _terminal_utility(self, board: Board, current_player: int) -> Optional[float]:
        done, winner = board.is_end(current_player)
        if not done:
            return None
        return 1.0 if winner == self.player else -1.0

    def _choose_tied(self, actions: Sequence[Action]) -> Optional[Action]:
        if not actions:
            return None
        if self.tie_breaking == "random":
            return self.random.choice(list(actions))
        return actions[0]

    @staticmethod
    def _same_value(first: float, second: float) -> bool:
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)

    def _start_search(self) -> None:
        self.last_metrics = SearchMetrics(search_depth=self.depth)

    def _finish_search(self, start: float, action: Optional[Action]) -> Optional[Action]:
        self.last_metrics.execution_time = time.perf_counter() - start
        self.last_metrics.chosen_action = action
        return action


class MinimaxAgent(BaseSearchAgent):
    """Depth-limited Minimax with the opponent modeled adversarially."""

    def next_action(self, obs: Board) -> Optional[Action]:
        self._start_search()
        start = time.perf_counter()
        _, action = self._minimax(obs, self.player, self.depth)
        return self._finish_search(start, action)

    def _minimax(
        self, board: Board, current_player: int, depth: int
    ) -> Tuple[float, Optional[Action]]:
        self.last_metrics.expanded_nodes += 1
        terminal = self._terminal_utility(board, current_player)
        if terminal is not None:
            return terminal, None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions: List[Action] = board.get_possible_actions(current_player)
        if not actions:  # Defensive fallback for custom Board implementations.
            return (-1.0 if current_player == self.player else 1.0), None

        maximizing = current_player == self.player
        best_value = -math.inf if maximizing else math.inf
        best_actions: List[Action] = []
        next_player = self._opponent(current_player)

        for action in actions:
            child = self._successor(board, action, current_player)
            value, _ = self._minimax(child, next_player, depth - 1)
            better = value > best_value if maximizing else value < best_value
            if better:
                best_value = value
                best_actions = [action]
            elif self._same_value(value, best_value):
                best_actions.append(action)
        return best_value, self._choose_tied(best_actions)


class AlphaBetaAgent(BaseSearchAgent):
    """Minimax with Alpha-Beta pruning and skipped-branch metrics."""

    def next_action(self, obs: Board) -> Optional[Action]:
        self._start_search()
        start = time.perf_counter()
        _, action = self._alpha_beta(
            obs, self.player, self.depth, -math.inf, math.inf
        )
        return self._finish_search(start, action)

    def _alpha_beta(
        self,
        board: Board,
        current_player: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Optional[Action]]:
        self.last_metrics.expanded_nodes += 1
        terminal = self._terminal_utility(board, current_player)
        if terminal is not None:
            return terminal, None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions: List[Action] = board.get_possible_actions(current_player)
        if not actions:
            return (-1.0 if current_player == self.player else 1.0), None

        maximizing = current_player == self.player
        best_value = -math.inf if maximizing else math.inf
        best_actions: List[Action] = []
        next_player = self._opponent(current_player)

        for index, action in enumerate(actions):
            child = self._successor(board, action, current_player)
            value, _ = self._alpha_beta(
                child, next_player, depth - 1, alpha, beta
            )
            better = value > best_value if maximizing else value < best_value
            if better:
                best_value = value
                best_actions = [action]
            elif self._same_value(value, best_value):
                best_actions.append(action)

            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if beta <= alpha:
                self.last_metrics.pruned_branches += len(actions) - index - 1
                break

        return best_value, self._choose_tied(best_actions)


class ExpectimaxAgent(BaseSearchAgent):
    """Expectimax with a uniform-random opponent at chance nodes.

    Alpha-Beta pruning is deliberately not used: ordinary Alpha-Beta bounds do
    not preserve the expected value at chance nodes.
    """

    def next_action(self, obs: Board) -> Optional[Action]:
        self._start_search()
        start = time.perf_counter()
        _, action = self._expectimax(obs, self.player, self.depth)
        return self._finish_search(start, action)

    def _expectimax(
        self, board: Board, current_player: int, depth: int
    ) -> Tuple[float, Optional[Action]]:
        self.last_metrics.expanded_nodes += 1
        terminal = self._terminal_utility(board, current_player)
        if terminal is not None:
            return terminal, None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions: List[Action] = board.get_possible_actions(current_player)
        if not actions:
            return (-1.0 if current_player == self.player else 1.0), None

        next_player = self._opponent(current_player)
        if current_player != self.player:
            total = 0.0
            for action in actions:
                child = self._successor(board, action, current_player)
                value, _ = self._expectimax(child, next_player, depth - 1)
                total += value
            return total / len(actions), None

        best_value = -math.inf
        best_actions: List[Action] = []
        for action in actions:
            child = self._successor(board, action, current_player)
            value, _ = self._expectimax(child, next_player, depth - 1)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif self._same_value(value, best_value):
                best_actions.append(action)
        return best_value, self._choose_tied(best_actions)
