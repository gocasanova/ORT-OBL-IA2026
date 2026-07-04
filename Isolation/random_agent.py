from agent import Agent
import random

class RandomAgent(Agent):
    def __init__(self, player=1, seed=None):
        super().__init__(player)
        self.random = random.Random(seed)
        
    def next_action(self, obs):
        return self.random_action(obs)
    
    def heuristic_utility(self, board):
        return 0

    def random_action(self, board):
        possible_actions = board.get_possible_actions(self.player)
        if not possible_actions:
            return None
        return self.random.choice(possible_actions)
