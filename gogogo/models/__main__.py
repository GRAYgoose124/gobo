import os
import numpy as np

from tensorflow import keras


# This function saves the model's weights and architecture to a specified directory
def save_model_state(model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(save_dir)


# This function loads a previously saved model from a specified directory
def load_model_state(save_dir):
    return keras.models.load_model(save_dir)


def train_model(model, train_X, train_Y, step_cb=None):
    total_epochs, epochs_between_cbs = 1000, 75
    eras = int(total_epochs / epochs_between_cbs)
    try:
        for i in range(eras):
            print(f"Training era {i+1}/{eras}")
            model.fit(
                train_X, train_Y, epochs=epochs_between_cbs, batch_size=128
            )  # Adjust epochs and batch size as necessary
            print("Saving model...")
            if step_cb:
                step_cb()
    except KeyboardInterrupt:
        pass

    return
