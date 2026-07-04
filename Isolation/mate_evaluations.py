"""Simple evaluation functions for depth-limited Isolation search.

Every function returns a value from player ``agent_id``'s point of view.  Values
for non-terminal positions stay strictly between -1 and 1, so a known terminal
win (+1) is always preferred over a promising but unfinished position.
"""

from collections import deque
from typing import Callable, Dict, Optional, Tuple

from board import Board, eliminated_cell, empty_cell

EvaluationFunction = Callable[[Board, int], float]
Position = Tuple[int, int]


def _opponent(player: int) -> int:
    return 2 if player == 1 else 1


def _terminal_value(board: Board, agent_id: int) -> Optional[float]:
    """Return a utility when one player is immobile, otherwise ``None``.

    Board does not store whose turn it is.  In reachable search states only the
    player about to move needs to be tested by the search itself; this helper is
    also useful when evaluation functions are called directly.
    """
    my_moves = board.has_valid_moves(agent_id)
    opponent_moves = board.has_valid_moves(_opponent(agent_id))
    if not my_moves and opponent_moves:
        return -1.0
    if my_moves and not opponent_moves:
        return 1.0
    return None


def _bounded(raw_score: float, scale: float = 1.0) -> float:
    """Monotonically map an unbounded heuristic to the open interval (-1, 1)."""
    scaled = raw_score / scale
    return scaled / (1.0 + abs(scaled))


def legal_move_count(board: Board, player: int) -> int:
    """Count movement directions, not move/removal action combinations."""
    position = board.find_player_position(player)
    if position is None:
        return 0
    return sum(board.can_move_to(position, direction) for direction in range(8))


def mobility_eval(board: Board, agent_id: int) -> float:
    """Favor states with more movement choices than the opponent."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal
    raw = legal_move_count(board, agent_id) - legal_move_count(
        board, _opponent(agent_id)
    )
    return _bounded(raw, scale=8.0)


def aggressive_eval(board: Board, agent_id: int) -> float:
    """Favor restricting the opponent's mobility."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal
    raw = -legal_move_count(board, _opponent(agent_id))
    return _bounded(raw, scale=8.0)


def defensive_eval(board: Board, agent_id: int) -> float:
    """Favor preserving the agent's own mobility."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal
    raw = legal_move_count(board, agent_id)
    return _bounded(raw, scale=8.0)


def balanced_eval(board: Board, agent_id: int) -> float:
    """Give own mobility twice the weight of opponent mobility."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal
    own = legal_move_count(board, agent_id)
    other = legal_move_count(board, _opponent(agent_id))
    return _bounded(2 * own - other, scale=16.0)


def _reachable_cells(board: Board, player: int) -> int:
    """Count empty cells reachable using king-like movement."""
    start = board.find_player_position(player)
    if start is None:
        return 0

    queue = deque([start])
    visited = {start}
    reachable = 0
    rows, cols = board.board_size
    while queue:
        row, col = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (row + dr, col + dc)
                if neighbor in visited:
                    continue
                nr, nc = neighbor
                if 0 <= nr < rows and 0 <= nc < cols:
                    visited.add(neighbor)
                    if board.grid[neighbor] == empty_cell:
                        reachable += 1
                        queue.append(neighbor)
    return reachable


def territory_eval(board: Board, agent_id: int) -> float:
    """Compare the number of free cells reachable by each player."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal
    raw = _reachable_cells(board, agent_id) - _reachable_cells(
        board, _opponent(agent_id)
    )
    return _bounded(raw, scale=float(board.grid.size))


def _blocked_neighbors(board: Board, player: int) -> int:
    position = board.find_player_position(player)
    if position is None:
        return 8
    row, col = position
    rows, cols = board.board_size
    blocked = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                blocked += 1
            elif board.grid[nr, nc] == eliminated_cell:
                blocked += 1
    return blocked


def weighted_eval(board: Board, agent_id: int) -> float:
    """Combine mobility, pressure around the opponent, and player distance."""
    terminal = _terminal_value(board, agent_id)
    if terminal is not None:
        return terminal

    opponent = _opponent(agent_id)
    own_moves = legal_move_count(board, agent_id)
    opponent_moves = legal_move_count(board, opponent)
    pressure = _blocked_neighbors(board, opponent) - _blocked_neighbors(
        board, agent_id
    )
    my_position = board.find_player_position(agent_id)
    opponent_position = board.find_player_position(opponent)
    distance = 0
    if my_position is not None and opponent_position is not None:
        distance = max(
            abs(my_position[0] - opponent_position[0]),
            abs(my_position[1] - opponent_position[1]),
        )

    # Nearby opponents are easier to constrain, hence the negative distance.
    raw = 2 * own_moves - 2 * opponent_moves + pressure - 0.25 * distance
    return _bounded(raw, scale=24.0)


EVALUATION_FUNCTIONS: Dict[str, EvaluationFunction] = {
    "mobility": mobility_eval,
    "aggressive": aggressive_eval,
    "defensive": defensive_eval,
    "balanced": balanced_eval,
    "territory": territory_eval,
    "weighted": weighted_eval,
}


def get_evaluation(name: str) -> EvaluationFunction:
    """Resolve a command-line-friendly evaluation function name."""
    try:
        return EVALUATION_FUNCTIONS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EVALUATION_FUNCTIONS))
        raise ValueError(f"Unknown evaluation '{name}'. Choose one of: {choices}") from exc
