import os
import numpy as np
from datetime import datetime

from tensorflow import keras


class GoModel:
    def __init__(self, model):
        self.model = model

    # This function saves the model's weights and architecture to a specified directory
    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_dir)

    # This function loads a previously saved model from a specified directory
    @staticmethod
    def load(save_dir):
        return GoModel(keras.models.load_model(save_dir))

    def train(self, train_X, train_Y, step_cb=None):
        total_epochs, epochs_between_cbs = 1000, 75
        eras = int(total_epochs / epochs_between_cbs)
        try:
            for i in range(eras):
                print(f"Training era {i+1}/{eras}")
                self.model.fit(
                    train_X, train_Y, epochs=epochs_between_cbs, batch_size=128
                )  # Adjust epochs and batch size as necessary
                if step_cb:
                    step_cb()
        except KeyboardInterrupt:
            pass

        return

    @staticmethod
    def create_or_load(models_dir, model_builder):
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
                        model = GoModel.load(str(active_model_dir))
                        model_loaded = True
                    finally:
                        pass
                print(f"Using directory: {active_model_dir}")
            else:
                print(f"No saved models found in {models_dir}")
        else:
            active_model_dir.mkdir(parents=True)

        if not model_loaded:
            model = GoModel(model_builder())

        return (model, active_model_dir)
