from typing import Tuple


import matplotlib.pyplot as plt
import tensorflow as tf


from histomics_detect.visualization.visualization import _plot_boxes
from histomics_detect.boxes.transforms import filter_edge_boxes


def plot_inference(rgb: tf.Tensor, boxes: tf.Tensor, nms_output: tf.Tensor, rpn_boxes: tf.Tensor,
                   save_fig: bool = False, fig_path: str = 'plot_of_inference.png', filter_edge: bool = False,
                   filter_margin: int = 32, threshold: float = 0.5, figsize: Tuple[int, int] = (20, 20),
                   gt_color: str = 'orange', pred_color: str = 'r', pred_size: int = 300):
    """
    plots the already run inference

    Returns
    -------

    """

    fig = plt.figure(figsize=figsize)
    plt.imshow(rgb)

    scores = tf.reshape(nms_output, -1)
    condition = tf.greater(scores, threshold)
    filtered_predictions = tf.squeeze(tf.gather(rpn_boxes, tf.where(condition)))

    if filter_edge:
        width, height, _ = tf.shape(rgb)
        filtered_boxes = filter_edge_boxes(boxes, width, height, filter_margin)

        edge_filtered_predictions = filter_edge_boxes(filtered_predictions, width, height, filter_margin)
    else:
        filtered_boxes = boxes

        edge_filtered_predictions = filtered_predictions

    _plot_boxes(filtered_boxes, gt_color)

    if tf.size(edge_filtered_predictions) > 0:
        filtered_prediction_dots = edge_filtered_predictions[:, :2] + edge_filtered_predictions[:, 2:]/2
        plt.scatter(filtered_prediction_dots[:, 0], filtered_prediction_dots[:, 1], color=pred_color, s=300)

    plt.show()
    if save_fig:
        fig.savefig(fig_path)

    # TODO calculate statistics for image


def run_plot():
    """
    runs the given model and plots the inference
    Returns
    -------

    """
    pass
