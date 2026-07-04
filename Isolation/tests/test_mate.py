"""Correctness tests for Part 2 (MATE)."""

import unittest

import numpy as np

from board import Board, eliminated_cell
from mate_agents import AlphaBetaAgent, ExpectimaxAgent, MinimaxAgent
from mate_evaluations import EVALUATION_FUNCTIONS, balanced_eval
from random_agent import RandomAgent


def example_board() -> Board:
    board = Board((4, 4))
    board.grid = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=int,
    )
    return board


class BoardTests(unittest.TestCase):
    def test_clone_preserves_rectangular_size_and_is_independent(self):
        board = Board((3, 5))
        clone = board.clone()
        self.assertEqual(clone.board_size, (3, 5))
        clone.grid[0, 0] = eliminated_cell
        self.assertFalse(np.array_equal(board.grid, clone.grid))

    def test_generated_actions_are_accepted_on_a_clone(self):
        board = example_board()
        original = board.grid.copy()
        for action in board.get_possible_actions(1):
            child = board.clone()
            self.assertTrue(child.play(action, 1))
        np.testing.assert_array_equal(board.grid, original)


class SearchAgentTests(unittest.TestCase):
    def test_all_agents_choose_legal_actions_without_mutating_board(self):
        for agent_class in (MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent):
            with self.subTest(agent=agent_class.__name__):
                board = example_board()
                original = board.grid.copy()
                agent = agent_class(1, depth=1, seed=7)
                action = agent.next_action(board)
                self.assertIn(action, board.get_possible_actions(1))
                np.testing.assert_array_equal(board.grid, original)
                self.assertEqual(agent.last_metrics.chosen_action, action)
                self.assertGreater(agent.last_metrics.expanded_nodes, 0)

    def test_minimax_and_alpha_beta_agree_and_alpha_beta_visits_no_more_nodes(self):
        board = example_board()
        minimax = MinimaxAgent(1, depth=2, evaluation_function=balanced_eval)
        alpha_beta = AlphaBetaAgent(1, depth=2, evaluation_function=balanced_eval)
        minimax_action = minimax.next_action(board)
        alpha_beta_action = alpha_beta.next_action(board)
        self.assertEqual(alpha_beta_action, minimax_action)
        self.assertLessEqual(
            alpha_beta.last_metrics.expanded_nodes,
            minimax.last_metrics.expanded_nodes,
        )
        self.assertGreater(alpha_beta.last_metrics.pruned_branches, 0)

    def test_agents_return_none_when_player_has_no_move(self):
        board = example_board()
        board.grid[0, 1] = eliminated_cell
        board.grid[1, 0] = eliminated_cell
        board.grid[1, 1] = eliminated_cell
        for agent_class in (MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent):
            self.assertIsNone(agent_class(1, depth=1).next_action(board))

    def test_depth_one_agent_can_finish_a_game(self):
        board = example_board()
        agents = {1: AlphaBetaAgent(1, depth=1), 2: RandomAgent(2, seed=9)}
        current = 1
        turns = 0
        while not board.is_end(current)[0]:
            action = agents[current].next_action(board)
            self.assertTrue(board.play(action, current))
            turns += 1
            current = 2 if current == 1 else 1
        self.assertGreater(turns, 0)
        self.assertLessEqual(turns, board.grid.size)


class EvaluationTests(unittest.TestCase):
    def test_all_evaluations_are_numeric_and_bounded(self):
        board = example_board()
        for name, evaluation in EVALUATION_FUNCTIONS.items():
            with self.subTest(evaluation=name):
                player_one = evaluation(board, 1)
                player_two = evaluation(board, 2)
                self.assertIsInstance(player_one, float)
                self.assertIsInstance(player_two, float)
                self.assertLessEqual(abs(player_one), 1.0)
                self.assertLessEqual(abs(player_two), 1.0)


if __name__ == "__main__":
    unittest.main()
