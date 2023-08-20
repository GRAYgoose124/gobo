from tensorflow import keras

from keras.layers import *


def create_go_model():
    input_board = Input(shape=(9, 9, 1))

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(input_board)
    x_recurrent = Reshape((81, 128))(x)
    gru_out = GRU(128, return_sequences=True)(x_recurrent)
    gru_out = Add()([x_recurrent, gru_out])
    x_flat = Flatten()(gru_out)
    x_dense = Dense(512, activation="relu")(x_flat)
    x_dense = Dropout(0.5)(x_dense)
    x_dense = Dense(256, activation="relu")(x_dense)
    x_dense = Dropout(0.5)(x_dense)
    x_dense = Dense(128, activation="relu")(x_dense)
    # x_dense = Add()([gru_out, x_dense])
    output = Dense(9 * 9, activation="softmax")(x_dense)

    model = keras.models.Model(inputs=input_board, outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
