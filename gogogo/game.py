import numpy as np

from pathlib import Path
from pysgf import SGF


from gogogo.board import GoBoard
from gogogo.bot import GoBot
from gogogo.models.basic import create_go_model


class GoGame:
    def __init__(self, model=None, model2=None, SAME_MODEL=True):
        self.board = GoBoard()
        self.data = {}
        self.sgf_file = None
        self.sgf_root = None

        self.model = model
        self.model2 = model if SAME_MODEL and model else model2

        if not self.model:
            self.model = create_go_model()
        if not self.model2:
            self.model2 = create_go_model()

        self.players: tuple[GoBot, GoBot] = (
            GoBot(-1, self, self.model),
            GoBot(1, self, self.model2),
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
        return game_data

    @classmethod
    def load_game_from_sgf(cls, sgf_file_path, cache_all_states=False):
        game = cls()
        game.sgf_file = sgf_file_path
        game.sgf_root = SGF.parse_file(sgf_file_path)

        board_states = []

        node = game.sgf_root
        while node.children:
            move = node.move
            if move:
                x, y = move.coords
                current_player = 1 if move.player == "B" else -1
                game.board.apply_move((x, y), current_player)
                if cache_all_states:
                    board_states.append(game.board._np_board.copy())
            node = node.children[0]

        game.data = {
            "board_states": board_states,
            "winner": game.sgf_root.get_property("RE"),
        }

        return game


class GoGameLibrary:
    def __init__(self, sgf_games_dir=None):
        self.sgf_games_dir = sgf_games_dir

        if sgf_games_dir:
            self.games = self.load_go_gamedata(sgf_games_dir)
        else:
            self.games = []

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, key):
        return self.games[key]
    
    def __iter__(self):
        return iter(self.games)
    
    def __next__(self):
        return next(self.games)

    @staticmethod
    def load_go_gamedata(sgf_games_dir):
        game_files = list(sgf_games_dir.glob("**/*.sgf"))
        loaded_gamedata = []
        total = len(game_files)
        failures = 0

        for i in range(total):
            try:
                game_data = GoGame.load_game_data_from_sgf(game_files[i])
                loaded_gamedata.append(game_data)
            except (TypeError, IndexError):
                failures += 1

        print(f"Total games: {total}")
        print(f"load failures: {failures}")
        return loaded_gamedata
