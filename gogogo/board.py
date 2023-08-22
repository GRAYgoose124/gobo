import numpy as np


class GoMoveError(Exception):
    ...


class GoGameSpaceOccupied(GoMoveError):
    ...


class GoBoard:
    def __init__(self, board: np.array = None):
        self.board = board or np.zeros((9, 9), dtype=int)

    def apply_move(self, move, player):
        if self.board[move[0]][move[1]] != 0:
            raise GoGameSpaceOccupied

        self.board[move] = player

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = move[0] + dx, move[1] + dy
            if 0 <= x < 9 and 0 <= y < 9 and self.board[x, y] == -player:
                if self.is_captured((x, y), player):
                    self.remove_group((x, y))

    def is_captured(self, start, player):
        visited = set()
        stack = [start]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 9 and 0 <= ny < 9:
                    if self.board[nx, ny] == 0:
                        return False
                    if (
                        self.board[nx, ny] == self.board[start[0], start[1]]
                        and (nx, ny) not in visited
                    ):
                        stack.append((nx, ny))
        return True

    def remove_group(self, start):
        stack = [start]
        to_remove = []
        while stack:
            x, y = stack.pop()
            to_remove.append((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < 9
                    and 0 <= ny < 9
                    and self.board[nx, ny] == self.board[start[0], start[1]]
                    and (nx, ny) not in to_remove
                ):
                    stack.append((nx, ny))
        for x, y in to_remove:
            self.board[x, y] = 0

    def get_legal_moves(self, player):
        legal_moves = []
        for x in range(9):
            for y in range(9):
                if self.board[x, y] == 0:
                    if not self.is_captured((x, y), player):
                        legal_moves.append((x, y))
        return legal_moves

    def get_legal_moves_as_onehot(self, player):
        legal_moves = self.get_legal_moves(player)
        onehot = np.zeros((9, 9))
        for x, y in legal_moves:
            onehot[x, y] = 1
        return onehot
