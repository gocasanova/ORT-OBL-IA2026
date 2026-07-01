from agent import Agent
from board import Board

class InputAgent(Agent):
    
    def __init__(self, player=1):
        super().__init__(player)
        
    def next_action(self, obs):
        while True:
            try:
                direction = int(input("Direction: "))
                row_to_destroy = int(input("Row to destroy: "))
                col_to_destroy = int(input("Column to destroy: "))
                input_value = (direction, (row_to_destroy, col_to_destroy))
                return input_value
            except ValueError:
                print("Please insert a valid action.")

    def heuristic_utility(self, board: Board):
        return 0