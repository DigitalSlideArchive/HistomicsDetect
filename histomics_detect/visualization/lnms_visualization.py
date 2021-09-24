from typing import Tuple


import matplotlib.pyplot as plt
import tensorflow as tf


from histomics_detect.visualization.visualization import _plot_boxes
from histomics_detect.boxes.transforms import filter_edge_boxes
from histomics_detect.anchors.create import create_anchors
from histomics_detect.metrics import greedy_iou
from histomics_detect.metrics.iou import iou
from histomics_detect.models.model_utils import extract_data


def plot_inference(rgb: tf.Tensor, boxes: tf.Tensor, nms_output: tf.Tensor, rpn_boxes: tf.Tensor,
                   save_fig: bool = False, fig_path: str = 'plot_of_inference.png', filter_edge: bool = False,
                   filter_margin: int = 32, threshold: float = 0.5, figsize: Tuple[int, int] = (20, 20),
                   gt_colors: Tuple[str, str] = ('g', 'r'), pred_colors: Tuple[str, str] = ('g', 'r'),
                   pred_size: int = 300, print_prediction_numbers: bool = True) -> None:
    """
    calculates the statistic of previously run output and plots it

    Parameters
    ----------
    rgb: array
        Image for display with imshow.
    boxes tensor (float32)
        ground truth boxes
        shape: G x 4
    nms_output: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    rpn_boxes: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of each regressed box and its width and height in that order.
    save_fig: bool
        true if picture should be saved
    fig_path: str
        path to save figure at
    filter_edge: bool
        true -> filters boxes and predictions too close to the border of image
    filter_margin: int
        min distance to border not to be filtered out
    threshold: float

    figsize: Tuple[int, int]
        dimensions of figure
    gt_colors: Tuple[str, str]
        color of predicted ground truth, color of not predicted ground truth (false negative)
    pred_colors: Tuple[str, str]
        color of true positive prediction, color of false positive prediction
    pred_size: int
        size of dot symbolizing a prediction
    print_prediction_numbers:
        true -> print statistic of tp, fp, fn

    Returns
    -------
    None

    """
    scores = tf.reshape(nms_output, -1)
    condition = tf.greater(scores, threshold)
    rpn_boxes = tf.squeeze(tf.gather(rpn_boxes, tf.where(condition)))

    ious, _ = iou(rpn_boxes, boxes)

    precision, recall, tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou(ious, 0.18)

    positive_pred = tf.reshape(tf.gather(rpn_boxes, tp_list[:, 0]), (-1, 4))
    negative_pred = tf.reshape(tf.gather(rpn_boxes, fp_list), (-1, 4))  # tf.where(labels == 0)

    negative_boxes = tf.reshape(tf.gather(boxes, fn_list), (-1, 4))

    if filter_edge:
        boxes = filter_edge_boxes(boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], filter_margin)
        negative_boxes = filter_edge_boxes(negative_boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], filter_margin)

        positive_pred = filter_edge_boxes(positive_pred, tf.shape(rgb)[1], tf.shape(rgb)[0], filter_margin)
        negative_pred = filter_edge_boxes(negative_pred, tf.shape(rgb)[1], tf.shape(rgb)[0], filter_margin)

    if print_prediction_numbers:
        print(f'tp: {tf.shape(positive_pred)[0]}, fp: {tf.shape(negative_pred)[0]}, fn: '
              f'{tf.shape(boxes)[0] - tf.shape(positive_pred)[0]}')

    fig = plt.figure(figsize=figsize)
    plt.imshow(tf.cast(rgb, tf.int32))

    _plot_boxes(boxes, gt_colors[0])  # '#b9ff38'
    _plot_boxes(negative_boxes, gt_colors[1])  # '#b9ff38'

    positive_pred_d = positive_pred[:, :2] + positive_pred[:, 2:] / 2
    negative_pred_d = negative_pred[:, :2] + negative_pred[:, 2:] / 2

    if tf.size(negative_pred_d) > 0:
        plt.scatter(negative_pred_d[:, 0], negative_pred_d[:, 1], color=pred_colors[1], s=pred_size)
    if tf.size(positive_pred_d) > 0:
        plt.scatter(positive_pred_d[:, 0], positive_pred_d[:, 1], color=pred_colors[0], s=pred_size)

    plt.show()
    if save_fig:
        fig.savefig(fig_path)


def run_plot(validation_data, model: tf.keras.Model, index: int = 0, save_fig: bool = False,
             fig_path: str = 'plot_of_inference.png', filter_edge: bool = False, filter_margin: int = 32,
             threshold: float = 0.5, figsize: Tuple[int, int] = (20, 20),gt_colors: Tuple[str, str] = ('g', 'r'),
             pred_colors: Tuple[str, str] = ('g', 'r'), pred_size: int = 300, print_prediction_numbers: bool = True)\
        -> None:
    """
    runs the given model and plots the inference
    Returns
    -------

    """
    validation_data = list(validation_data.as_numpy_iterator())
    data = validation_data[index]

    rgb, boxes, _ = data
    print(_)

    width, height, _ = tf.shape(rgb)
    model.anchors = create_anchors(model.anchor_px, model.field, height, width)

    norm, boxes, sample_weight = extract_data(data)
    features, rpn_boxes, scores = model.extract_boxes_n_scores(norm)

    comp_features = model.compression_net(features)
    interpolated = model._interpolate_features(comp_features, rpn_boxes)

    interpolated = tf.concat([scores, tf.reshape(interpolated, [tf.shape(interpolated)[0], -1])], axis=1)
    nms_output2 = model.net((interpolated, rpn_boxes), training=True)
    nms_output2 = tf.expand_dims(nms_output2[:, 0], axis=1)

    plot_inference(rgb, boxes, nms_output2, rpn_boxes, save_fig, fig_path, filter_edge, filter_margin, threshold,
                   figsize, gt_colors, pred_colors, pred_size, print_prediction_numbers)
