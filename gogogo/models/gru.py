from tensorflow import keras


from keras.layers import (
    Input,
    Conv2D,
    Reshape,
    GRU,
    Add,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.regularizers import l2
from keras.models import Model


def create_go_model():
    input_board = Input(shape=(9, 9, 1))

    # Initial Conv layers with increased filters and batch normalization
    x = Conv2D(256, (5, 5), padding="same", activation="relu")(input_board)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="heatmap_target")(x)
    x = BatchNormalization()(x)

    # Residual Connection with convolution
    residual = Conv2D(256, (3, 3), padding="same")(x)
    x = Add()([x, residual])
    x = BatchNormalization()(x)

    # Recurrent GRU
    x_recurrent = Reshape((81, 256))(x)
    gru_out1 = GRU(256, return_sequences=True)(x_recurrent)
    gru_out1_added = Add()([x_recurrent, gru_out1])

    # Another GRU layer
    gru_out2 = GRU(256, return_sequences=True)(gru_out1_added)
    gru_out2_added = Add()([gru_out1_added, gru_out2])

    # Convolve after GRUs
    x_after_gru = Reshape((9, 9, 256))(gru_out2_added)
    x_conv = Conv2D(256, (3, 3), padding="same", activation="relu")(x_after_gru)
    x = BatchNormalization()(x_conv)

    # Flatten and dense layers
    x_flat = Flatten()(x)
    x_dense = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(x_flat)
    x_dense = Dense(256, activation="relu", kernel_regularizer=l2(0.0001))(x_dense)
    x_dense = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))(x_dense)

    # Output layer
    output = Dense(9 * 9, activation="softmax")(x_dense)

    # Model definition
    model = Model(inputs=input_board, outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )
    return model
