import numpy as np

from gogogo.board import GoMoveError


class GoTrainer:
    """Uses a go scorer to train a model to play go"""

    def play_game(self):
        game_over = 0
        for turn in range(81):
            if game_over >= 2:
                break

            current_player = 1 if turn % 2 == 0 else -1
            input_board = np.expand_dims(self.board, axis=-1)
            input_board = np.expand_dims(self.board, axis=0)

            move_probs = list(self.model.predict(input_board))
            move_probs = [
                np.where(move_prob > 0.1, move_prob, 0) for move_prob in move_probs
            ]
            while True:
                try:
                    move_prob = move_probs.pop()
                    game_over = 0
                except IndexError:
                    print("No more moves to try, passing...")
                    game_over += 1
                    break

                move_coords = np.unravel_index(np.argmax(move_prob), self.board.shape)

                try:
                    self.apply_move(move_coords, current_player)
                    break
                except GoMoveError as e:
                    print(e)
