import numpy as np

from pathlib import Path
from pysgf import SGF


from gogogo.board import GoBoard
from gogogo.bot import GoBot
from gogogo.models.gru import create_go_model as gomod


class GoPlayer:
    def __init__(self, board: GoBoard):
        self.board = board


class GoGame:
    def __init__(self):
        self.board = GoBoard()
        self.winner = None
        self.history = []

        self.model = gomod()
        self.players: tuple[GoBot, GoBot] = (
            GoBot(-1, self, self.model),
            GoBot(1, self, self.model),
        )

    @staticmethod
    def load_game_data_from_sgf(sgf_file_path):
        root = SGF.parse_file(sgf_file_path)
        board = np.zeros((9, 9), dtype=int)
        game_data = {"board_states": [], "winner": root.get_property("RE")}
        node = root
        while node.children:
            move = node.move
            if move:
                x, y = move.coords
                current_player = 1 if move.player == "B" else -1
                label = np.zeros(9 * 9)
                label[x * 9 + y] = 1
                input_board = np.expand_dims(board, axis=-1)
                input_board = np.expand_dims(input_board, axis=0)
                game_data["board_states"].append((input_board, label))
                board[x, y] = current_player
            node = node.children[0]
        return game_data["board_states"]

    @staticmethod
    def load_go_gamedata(sgf_games_dir):
        game_files = list(sgf_games_dir.glob("**/*.sgf"))
        loaded_gamedata = []
        total = len(game_files)
        failures = 0
        for i in range(total):
            try:
                game_data = GoBoard.load_game_data_from_sgf(game_files[i])
            except (TypeError, IndexError):
                failures += 1
            loaded_gamedata.extend(game_data)
        print(f"Total games: {total}")
        print(f"load failures: {failures}")
        return GoBoard(board=loaded_gamedata[-1])
