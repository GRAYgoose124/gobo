from pathlib import Path
import traceback

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Ensure the img_array is 4D
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.expand_dims(img_array, -1)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def make_board_fig(board, moves=None):
    fig, ax = plt.subplots(figsize=(2, 2))

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 1:  # Black stone
                ax.add_patch(
                    patches.Circle((j, board.shape[0] - i - 1), 0.4, facecolor="black")
                )
            elif board[i, j] == -1:  # White stone
                ax.add_patch(
                    patches.Circle((j, board.shape[0] - i - 1), 0.4, facecolor="tan")
                )

    # Highlight the best move position
    while moves:
        move = moves.pop()
        ax.add_patch(
            patches.Circle((move[1], 8 - move[0]), 0.4, facecolor="red", alpha=0.3)
        )

    ax.set_aspect("equal")
    ax.set_xticks(range(board.shape[1]))
    ax.set_yticks(range(board.shape[0]))
    ax.grid(which="both")

    # Adjust the x and y axis limits to provide some margin around the grid
    ax.set_xlim(-0.5, board.shape[1] - 0.5)
    ax.set_ylim(-0.5, board.shape[0] - 0.5)

    # plt.gca().invert_yaxis()
    fig.gca().invert_yaxis()
    return fig, ax


def plot_board(board, moves=None):
    fig, ax = make_board_fig(board, moves)
    plt.show()


def plot_heatmap(board, model, layer="heatmap_target"):
    heatmap = make_gradcam_heatmap(board, model, layer)
    plt.matshow(heatmap)
    plt.gca().invert_yaxis()
    plt.show()


def plot_bah_side_by_side(board, model, moves=None, layer="heatmap_target"):
    fig, ax = make_board_fig(board, moves)
    ax2 = ax.twinx()
    try:
        heatmap = make_gradcam_heatmap(board, model, layer)
        ax2.matshow(heatmap)
        ax2.set_yticks([])
    except ValueError:
        traceback.print_exc()

    plt.show()


def plot_bah_underlay(
    board, model, moves=None, layer="heatmap_target", save: Path | None = None
):
    fig, ax = make_board_fig(board, moves)
    try:
        heatmap = make_gradcam_heatmap(board, model, layer)
        ax.imshow(heatmap, alpha=0.5)
    except ValueError:
        traceback.print_exc()

    if save:
        plt.savefig(save)

    plt.show()
