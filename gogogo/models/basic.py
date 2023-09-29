from tensorflow import keras


# model funcs
def create_go_model():
    # Input layer
    input_board = keras.layers.Input(shape=(9, 9, 1))

    # First set of convolutional layers
    x = keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name="conv2d"
    )(input_board)
    x = keras.layers.Conv2D(
        81, (3, 3), padding="same", activation="relu", name="conv2d1"
    )(x)

    # Recurrent layers
    # Reshape the board to (81, features) for recurrent processing
    x_recurrent = keras.layers.Reshape((81, 81))(x)
    lstm_out1 = keras.layers.LSTM(32, return_sequences=True, name="lstm")(x_recurrent)
    lstm_out2 = keras.layers.LSTM(32, return_sequences=True, name="lstm1")(x_recurrent)

    # Merge recurrent outputs
    merged_recurrent = keras.layers.concatenate([lstm_out1, lstm_out2])
    merged_recurrent_flat = keras.layers.Flatten()(merged_recurrent)

    # Fully connected layers
    x_dense = keras.layers.Dense(512, activation="relu")(merged_recurrent_flat)
    x_dense = keras.layers.Dropout(0.5)(x_dense)  # Regularization
    x_dense = keras.layers.Dense(256, activation="relu")(x_dense)

    # Final output
    output = keras.layers.Dense(9 * 9, activation="softmax", name="board_out")(x_dense)

    model = keras.models.Model(inputs=input_board, outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
