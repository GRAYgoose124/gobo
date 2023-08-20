from abc import ABC, abstractmethod

import numpy as np

from .board import GoBoard, GoMoveError


class GoScorer(ABC):
    """
    To determine the winner let us first develop a scoring function which we can apply to a go game.
    We will use it to first compare with the sgf['RE'] tag, and if it matches, we will us to determine
    move quality by applying it to the board state after each move.

    The scoring function will be based on the following rules:



    """

    @abstractmethod
    def score(self, board):
        pass

    @staticmethod
    def count_free_spaces(board, x, y):
        count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 9 and 0 <= ny < 9 and board[nx, ny] == 0:
                count += 1
        return count


class FreeAreaScorer(GoScorer):
    # We'll start with a very simple scoring function that counts the free spaces around each player's stones.
    # The player with the most free spaces wins.
    def score(self, board):
        # First we'll count the number of free spaces around each player's stones
        black_score = 0
        white_score = 0
        for x in range(9):
            for y in range(9):
                if board[x, y] == 1:
                    black_score += self.count_free_spaces(board, x, y)
                elif board[x, y] == -1:
                    white_score += self.count_free_spaces(board, x, y)

        # Return the difference between the two scores
        return black_score - white_score


class TerritoryCountScorer(GoScorer):
    def score(self, board):
        # First we'll create a copy of the board to mark which spaces have been counted
        counted_board = np.zeros((9, 9))

        # Then we'll iterate over each space on the board and count the territory for each player
        black_territory = 0
        white_territory = 0
        for x in range(9):
            for y in range(9):
                if counted_board[x, y] == 0:
                    territory, player = self.count_territory(board, counted_board, x, y)
                    if player == 1:
                        black_territory += territory
                    elif player == -1:
                        white_territory += territory

        # Return the difference between the two territory counts
        return black_territory - white_territory

    def count_territory(self, board, counted_board, x, y):
        # If the space has already been counted, return 0 territory and the player who owns the stone
        if counted_board[x, y] == 1:
            return 0, board[x, y]

        # Mark the space as counted
        counted_board[x, y] = 1

        # If the space is empty, count the territory for both players
        if board[x, y] == 0:
            black_territory = 0
            white_territory = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= 9 or ny < 0 or ny >= 9:
                    continue
                territory, player = self.count_territory(board, counted_board, nx, ny)
                if player == 1:
                    black_territory += territory
                elif player == -1:
                    white_territory += territory
            if black_territory > 0 and white_territory == 0:
                return black_territory, 1
            elif white_territory > 0 and black_territory == 0:
                return white_territory, -1
            else:
                return 0, 0

        # If the space is owned by a player, count the territory for that player
        elif board[x, y] == 1:
            territory = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= 9 or ny < 0 or ny >= 9:
                    continue
                if board[nx, ny] == -1:
                    territory += 1
                elif board[nx, ny] == 0:
                    territory += self.count_territory(board, counted_board, nx, ny)[0]
            return territory, 1

        elif board[x, y] == -1:
            territory = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= 9 or ny < 0 or ny >= 9:
                    continue
                if board[nx, ny] == 1:
                    territory += 1
                elif board[nx, ny] == 0:
                    territory += self.count_territory(board, counted_board, nx, ny)[0]
            return territory, -1
