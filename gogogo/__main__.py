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

from .board import GoBoard
from .visualization import plot_bah_underlay

from .models.gru import create_go_model


def create_or_load_model(models_dir=Path(__file__).parents[1] / "data" / "models"):
    # Create or load the model
    model_loaded = False
    active_model_dir = models_dir / str(datetime.now())
    if models_dir.exists():
        # see if we've saved anything of the datetime format yet..
        saved_models = list(models_dir.glob("*????-??-??*"))
        if len(saved_models):
            active_model_dir = max(saved_models, key=os.path.getctime)

            if (
                model_loaded
                or input(f"Load model?\t {active_model_dir}\n(Y/n): ") != "n"
            ):
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

    return (model, active_model_dir)


def main():
    if len(tf.config.list_physical_devices("GPU")) < 1:
        print("\n\nNo GPU detected, training may be slow.\n\n")

    # Training set from some game data
    loaded_gamedata = GoBoard.load_go_gamedata(Path(__file__).parents[1] / "data/9x9")
    train_X = np.vstack([data[0] for data in loaded_gamedata])
    train_Y = np.vstack([data[1] for data in loaded_gamedata])

    model, save_dir = create_or_load_model()

    train_model(
        model,
        train_X,
        train_Y,
        step_cb=lambda: save_model_state(model, save_dir),
    )

    board = play_game(model)

    best_move = predict_best_move(model, board)
    plot_bah_underlay(board, model, moves=[best_move], layer="heatmap_target")


if __name__ == "__main__":
    main()
