from datetime import datetime
import os
from pathlib import Path
import numpy as np


import tensorflow as tf


from .board import GoBoard
from .game import GoGame, GoGameLibrary
from .models.__main__ import GoModel
from .models.gru import create_go_model

from .visualization import plot_bah_underlay


def main():
    if len(tf.config.list_physical_devices("GPU")) < 1:
        print("\n\nNo GPU detected, training may be slow.\n\n")

    # Training set from some game data
    loaded_gamedata = GoGameLibrary.load_go_gamedata(
        Path(__file__).parents[1] / "data/9x9"
    )
    train_X = np.vstack([data[0] for data in loaded_gamedata])
    train_Y = np.vstack([data[1] for data in loaded_gamedata])

    model, save_dir = GoModel.create_or_load(
        models_dir=Path(__file__).parents[1] / "data" / "models",
        model_builder=create_go_model,
    )

    model.train(
        train_X,
        train_Y,
        step_cb=lambda: model.save(save_dir),
    )

    board = play_game(model)

    best_move = predict_best_move(model, board)
    plot_bah_underlay(board, model, moves=[best_move], layer="heatmap_target")


if __name__ == "__main__":
    main()
