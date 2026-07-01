import gymnasium as gym
import numpy as np
from board import Board

player_one = 1
player_two = 2

class IsolationEnv(gym.Env):
    def __init__(self):
        super(IsolationEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,4), dtype=np.float32)
        self.reset()
        
    def reset(self):
        self.current_player = player_one
        self.grid = Board()
        return self.grid

    def step(self, action):
        action_done = self.grid.play(action, self.current_player)
        
        if action_done:
            self.next_player()
        else:
            raise ValueError("Invalid action")
            
        done, winner = self.is_done()
        return self.grid, self.get_reward(), done, winner, {}
    
    def is_done(self):        
        return self.grid.is_end(self.current_player)
    
    def get_reward(self):
        return 1 if self.current_player == player_one else -1   
        
    def next_player(self):
        self.current_player = player_one if self.current_player == player_two \
                                    else player_two
    def render(self):
        self.grid.render_grid()



