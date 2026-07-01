import numpy as np
from tabulate import tabulate
import random

empty_cell = 0
eliminated_cell = 3
int_to_string = [" ", "B", "R", "X"]

class Board:
    def __init__(self, board_size=(4, 4)):
        self.grid = np.zeros(board_size, dtype=int)
        self.board_size = board_size
        self.init_board()

    def init_board(self):
        self.place_players()

    def place_players(self):
        positions = [(i, j) for i in range(self.board_size[0]) for j in range(self.board_size[1])]
        random.shuffle(positions)

        player1_position = positions.pop()
        self.grid[player1_position] = 1

        player2_position = positions.pop()
        self.grid[player2_position] = 2

    def play(self, action, player):
        """Mueve al jugador en la dirección indicada y elimina la celda indicada"""
        direction = action[0]
        cell_to_destroy = action[1]
        
        current_player_position = self.find_player_position(player)
        
        can_move = self.can_move_to(current_player_position, direction)
        if can_move:
            temp_board = self.clone()
            temp_board.move(player, current_player_position, direction)
            temp_board.grid[current_player_position] = eliminated_cell
            can_eliminate = temp_board.can_eliminate(cell_to_destroy)
        else:
            return False
        
        if can_eliminate:
            self.move_and_destroy(player, current_player_position, direction, cell_to_destroy)
            self.grid[current_player_position] = eliminated_cell
            return True
        else:
            return False
            
    def can_move_to(self, player_position, direction):
        row, col = player_position

        if direction == 0:  # Up
            new_row, new_col = row - 1, col
        elif direction == 1:  # Down
            new_row, new_col = row + 1, col
        elif direction == 2:  # Left
            new_row, new_col = row, col - 1
        elif direction == 3:  # Right
            new_row, new_col = row, col + 1
        elif direction == 4:  # Up-Left
            new_row, new_col = row - 1, col - 1
        elif direction == 5:  # Up-Right
            new_row, new_col = row - 1, col + 1
        elif direction == 6:  # Down-Left
            new_row, new_col = row + 1, col - 1
        elif direction == 7:  # Down-Right
            new_row, new_col = row + 1, col + 1
        else:
            return False

        if (
            new_row < 0
            or new_row >= self.board_size[0]
            or new_col < 0
            or new_col >= self.board_size[1]
        ):
            return False

        if self.grid[new_row, new_col] == empty_cell:
            return True

        return False

    def can_eliminate(self, cell_to_destroy):
        return self.grid[cell_to_destroy] == empty_cell
        
    def move(self, player, current_player_position, direction):
        row, col = current_player_position
        self.grid[row, col] = empty_cell
        
        if direction == 0:
            self.grid[row - 1, col] = player
        elif direction == 1:
            self.grid[row + 1, col] = player
        elif direction == 2:
            self.grid[row, col - 1] = player
        elif direction == 3:
            self.grid[row, col + 1] = player
        elif direction == 4:
            self.grid[row - 1, col - 1] = player
        elif direction == 5:
            self.grid[row - 1, col + 1] = player
        elif direction == 6:
            self.grid[row + 1, col - 1] = player
        elif direction == 7:
            self.grid[row + 1, col + 1] = player
        else:
            return 
        
    def move_and_destroy(self, player, current_player_position, direction, cell_to_destroy):
        self.move(player, current_player_position, direction)
        self.grid[cell_to_destroy] = eliminated_cell
            
    def find_player_position(self, player):
        """Devuelve la posición del jugador en el tablero"""
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.grid[row, col] == player:
                    return (row, col)
        return None

    def render_grid(self):
        """Imprime el tablero en consola"""
        num_rows, num_cols = self.board_size
        table = [["" for _ in range(num_cols + 1)] for _ in range(num_rows)]

        for row in range(num_rows):
            for col in range(num_cols):
                cell = self.grid[row, col]
                if cell == 1:
                    table[row][col + 1] = "B"
                elif cell == 2:
                    table[row][col + 1] = "R"
                elif cell == 3:
                    table[row][col + 1] = "X"

        for i in range(num_rows):
            table[i][0] = str(i)

        headers = [str(col) for col in range(num_cols)]
        headers.insert(0, " ")
        table.insert(0, headers)

        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def clone(self):
        """Devuelve una copia del tablero"""
        board_clone = Board()
        board_clone.grid = np.copy(self.grid)
        return board_clone
    
    def is_end(self, player) -> tuple[bool, int]:
        """
        Devuelve: (True/False que indica si termino el juego, 1/2 que indica el jugador que ganó)
        """
        if not self.has_valid_moves(player):
            return True, player % 2 + 1
        else:
            return False, 0
    
    def get_possible_actions(self, player):
        """Devuelve una lista de acciones posibles para el jugador"""
        possible_actions = []
        current_position = self.find_player_position(player)

        for direction in range(8):
            
            if self.can_move_to(current_position, direction):
                temp_board = self.clone()
                temp_board.move(player, current_position, direction)
                temp_board.grid[current_position] = eliminated_cell
                for row in range(self.board_size[0]):
                    for col in range(self.board_size[1]):
                        cell_to_destroy = (row, col)
                        if temp_board.can_eliminate(cell_to_destroy):
                            possible_actions.append((direction, cell_to_destroy))
        return possible_actions
    
    def has_valid_moves(self, player):
        player_position = self.find_player_position(player)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if self.is_valid_move(player_position, (dr, dc)):
                    return True

        return False

    def is_valid_move(self, player_position, direction):
        dr, dc = direction
        row, col = player_position

        new_row, new_col = row + dr, col + dc

        if (
            new_row >= 0
            and new_row < self.board_size[0]
            and new_col >= 0
            and new_col < self.board_size[1]
        ):
            if self.grid[new_row, new_col] == empty_cell:
                return True

        return False