from tensorflow import keras


def create_go_model():
    input_board = keras.layers.Input(shape=(9, 9, 1))

    # Dilated convolution helps the network to capture larger patterns.
    x = keras.layers.Conv2D(
        128, (3, 3), dilation_rate=(2, 2), padding="same", activation="relu"
    )(input_board)

    # LSTM layers
    x_recurrent = keras.layers.Reshape((81, 128))(x)
    lstm_out1 = keras.layers.LSTM(64, return_sequences=True)(x_recurrent)
    lstm_out2 = keras.layers.LSTM(32, return_sequences=True)(lstm_out1)

    x_flat = keras.layers.Flatten()(lstm_out2)
    x_dense = keras.layers.Dense(512, activation="relu")(x_flat)
    x_dense = keras.layers.Dropout(0.5)(x_dense)
    x_dense = keras.layers.Dense(256, activation="relu")(x_dense)

    output = keras.layers.Dense(9 * 9, activation="softmax")(x_dense)

    model = keras.models.Model(inputs=input_board, outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

    return model
