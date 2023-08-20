import numpy as np
from pathlib import Path

from pysgf import SGF


class GoMoveError(Exception):
    ...


class GoGameSpaceOccupied(GoMoveError):
    ...


def apply_move(board, move, player):
    """Apply a move for a player and retcurn the updated board."""
    board = board.copy()

    if board[move[0]][move[1]] != 0:
        raise GoGameSpaceOccupied

    board[move] = player

    # Check captures
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        x, y = move[0] + dx, move[1] + dy
        if 0 <= x < 9 and 0 <= y < 9 and board[x, y] == -player:
            if is_captured(board, (x, y), player):
                board = remove_group(board, (x, y))

    return board


def is_captured(board, start, player):
    """Check if a stone (and connected group) is captured by the player."""
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
                if board[nx, ny] == 0:
                    return False
                if (
                    board[nx, ny] == board[start[0], start[1]]
                    and (nx, ny) not in visited
                ):
                    stack.append((nx, ny))
    return True


def remove_group(board, start):
    """Remove a group of stones starting from the start."""
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
                and board[nx, ny] == board[start[0], start[1]]
                and (nx, ny) not in to_remove
            ):
                stack.append((nx, ny))
    for x, y in to_remove:
        board[x, y] = 0
    return board


def load_game_data_from_sgf(sgf_file_path):
    # Parse the SGF file
    root = SGF.parse_file(sgf_file_path)

    # Initialize an empty board
    board = np.zeros((9, 9), dtype=int)

    # List to store game data
    game_data = []

    # Traverse the SGF moves
    node = root
    while node.children:
        move = node.move
        if move:
            x, y = move.coords

            # Determine the current player
            current_player = 1 if move.player == "B" else -1

            # Store the game state and the chosen move
            label = np.zeros(9 * 9)
            label[x * 9 + y] = 1
            input_board = np.expand_dims(board, axis=-1)
            input_board = np.expand_dims(input_board, axis=0)
            game_data.append((input_board, label))

            # Apply move to board
            board[x, y] = current_player

        node = node.children[0]

    return game_data


def load_go_game_data(games_folder=Path(__file__).parents[1] / "data/9x9"):
    # Load game data for model

    game_files = list(games_folder.glob("**/*.sgf"))
    print(f"Total game files: {len(game_files)}, {games_folder=} ")
    loaded_gamedata = []

    total = len(game_files)
    failures = 0
    for i in range(total):
        try:
            game_data = load_game_data_from_sgf(game_files[i])
        except (TypeError, IndexError):
            failures += 1

        loaded_gamedata.extend(game_data)

    print(f"Total games: {total}")
    print(f"load failures: {failures}")

    return loaded_gamedata
