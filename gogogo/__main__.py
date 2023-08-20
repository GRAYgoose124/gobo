from datetime import datetime
import os
from pathlib import Path
import numpy as np


import tensorflow as tf


from .models import (
    load_model_state,
    save_model_state,
    train_model,
    predict_best_move,
)

from .gogame import GoMoveError, apply_move, load_go_game_data
from .visualization import plot_board_and_heatmap_side_by_side

from .models.gru import create_go_model


def main():
    if len(tf.config.list_physical_devices("GPU")) < 1:
        print("\n\nNo GPU detected, training may be slow.\n\n")

    # Training set from some game data
    loaded_gamedata = load_go_game_data()
    train_X = np.vstack([data[0] for data in loaded_gamedata])
    train_Y = np.vstack([data[1] for data in loaded_gamedata])

    # Create or load the model
    model_loaded = False
    models_dir = Path(__file__).parents[1] / "data" / "models"
    active_model_dir = models_dir / str(datetime.now())
    if models_dir.exists():
        # see if we've saved anything of the datetime format yet..
        saved_models = list(models_dir.glob("*.????-??-??T??:??:??"))
        if len(saved_models):
            active_model_dir = max(saved_models)

            if model_loaded or input("Load model? (Y/n): ") != "n":
                print("Loading...")
                try:
                    model = load_model_state(str(active_model_dir))
                    model_loaded = True
                finally:
                    pass
            print(f"Using directory: {active_model_dir}")
        else:
            print(f"No saved models found in {models_dir}")
    else:
        active_model_dir.mkdir(parents=True)

    if not model_loaded:
        model = create_go_model()

    train_model(
        model,
        train_X,
        train_Y,
        step_cb=lambda: save_model_state(model, active_model_dir),
    )

    # Initialize board
    board = np.zeros((9, 9), dtype=int)

    # Play a game:
    for turn in range(81):  # Max 81 turns
        current_player = 1 if turn % 2 == 0 else -1
        input_board = np.expand_dims(board, axis=-1)
        input_board = np.expand_dims(input_board, axis=0)

        move_probs = list(model.predict(input_board))
        move_probs = [
            np.where(move_prob > 0.1, move_prob, 0) for move_prob in move_probs
        ]
        while True:
            try:
                move_prob = move_probs.pop()
            except IndexError:
                print("No more moves to try, passing...")
                break

            move_coords = np.unravel_index(np.argmax(move_prob), board.shape)

            # Apply move
            try:
                board = apply_move(board, move_coords, current_player)
                break
            except GoMoveError as e:
                print(e)

    best_move = predict_best_move(model, board)
    plot_board_and_heatmap_side_by_side(board, model, moves=[best_move], layer="conv2d")


if __name__ == "__main__":
    main()
