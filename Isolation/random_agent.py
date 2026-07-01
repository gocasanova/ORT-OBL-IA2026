from agent import Agent
import random

class RandomAgent(Agent):
    def __init__(self, player=1):
        super().__init__(player)
        
    def next_action(self, obs):
        return self.random_action(obs)
    
    def heuristic_utility(self, board):
        return 0
    
    def random_action(self, board):
        possible_actions = board.get_possible_actions(self.player)
        return possible_actions[random.randint(0, len(possible_actions) - 1)]
        