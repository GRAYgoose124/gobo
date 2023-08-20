import os
import numpy as np

from tensorflow import keras


def predict_best_move(model, board):
    # Reshape the board to match the input shape of the model
    input_board = board.reshape(1, 9, 9, 1)

    # Get model prediction
    prediction = model.predict(input_board)

    # Determine the best move position from the prediction
    best_move_position = np.unravel_index(np.argmax(prediction), (9, 9))

    return best_move_position


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
    try:
        for _ in range(int(total_epochs / epochs_between_cbs)):
            model.fit(
                train_X, train_Y, epochs=epochs_between_cbs, batch_size=128
            )  # Adjust epochs and batch size as necessary
            print("Saving model...")
            if step_cb:
                step_cb()
    except KeyboardInterrupt:
        pass

    return
