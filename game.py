import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.move_count = 0

    def get_available_moves(self):
        return [i * 3 + j for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def make_move(self, position, player):
        if position < 0 or position > 8:
            return False
        row, col = position // 3, position % 3
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = player
        self.move_count += 1
        return True

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                return self.board[0][j]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        return 0

    def is_board_full(self):
        return self.move_count == 9

    def get_game_result(self):
        winner = self.check_winner()
        if winner != 0:
            return winner
        if self.is_board_full():
            return 0
        return None
