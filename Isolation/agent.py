from abc import ABC, abstractmethod
from board import Board

class Agent(ABC):
    
    @abstractmethod
    def __init__(self, player):
        self.player = player
        self.board = (player % 2) + 1

    @abstractmethod
    def next_action(self, obs):
        return NotImplementedError
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        return NotImplementedError
