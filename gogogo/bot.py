from typing import Literal
import numpy as np


class GoBot:
    def __init__(self, player: Literal[-1, 1], game, model):
        self.player = player
        self.game = game
        self.model = model

    def predict_best_move(self, board=None):
        return self.predict_top_moves(board, top=1)[0]

    def predict_top_moves(self, top=3):
        board = board or self.game.board

        # Reshape the board to match the input shape of the model
        input_board = board.reshape(1, 9, 9, 1)

        # Get model prediction
        prediction = self.model.predict(input_board)

        # Determine the best move position from the prediction
        top_move_positions = np.argsort(prediction)[0, -top:][::-1]

        return top_move_positions
