from datetime import datetime
import os
from pathlib import Path
import numpy as np


import tensorflow as tf


from .board import GoBoard
from .game import GoGame, GoGameLibrary
from .bot import GoBot
from .models.__main__ import GoModel
from .models.basic import create_go_model

from .visualization import plot_bah_underlay


def main():
    if len(tf.config.list_physical_devices("GPU")) < 1:
        print("\n\nNo GPU detected, training may be slow.\n\n")

    # Training set from some game data
    library = GoGameLibrary(
        Path(__file__).parents[1] / "data" / "9x9"
    )

    sample_game = library.games[0]['board_states']
    print(f"moves:{len(sample_game)}, {sample_game[0][0].shape}, {sample_game[0][1].shape}")
    # TODO: Need a better train_Y - like score per move instead of just winner
    # Rather than just train ending state and winner, train on each move/replay game and score relative domination.
    train_X = np.vstack([data[0] for data in sample_game])
    train_Y = np.vstack([data[1] for data in sample_game])

    print(train_X.shape, train_Y.shape)
    print(library.games[0].keys())

    print("Creating or loading model...")
    model, save_dir = GoModel.create_or_load(
        models_dir=Path(__file__).parents[1] / "data" / "models",
        model_builder=create_go_model,
    )

    print("Training...")
    model.train(
        train_X,
        train_Y,
        step_cb=lambda: model.save(save_dir),
    )

    print("Playing game...")
    game = GoGame(model=model)

    best_move = game.players[0].predict_best_move()
    print(best_move)
    plot_bah_underlay(game.board.raw, model, moves=[best_move], layer="heatmap_target")


if __name__ == "__main__":
    main()
