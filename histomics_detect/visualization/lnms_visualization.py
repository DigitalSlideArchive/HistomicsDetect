from typing import Tuple, List

import matplotlib.pyplot as plt
import tensorflow as tf

from histomics_detect.boxes.transforms import filter_edge_boxes
from histomics_detect.anchors.create import create_anchors
from histomics_detect.metrics import greedy_iou_mapping
from histomics_detect.metrics.iou import iou
from histomics_detect.models.model_utils import extract_data
from histomics_detect.models.lnms_model import LearningNMS
from histomics_detect.models.compression_network import CompressionNetwork


def plot_inference(rgb: tf.Tensor, boxes: tf.Tensor, nms_output: tf.Tensor, rpn_boxes: tf.Tensor,
                   save_fig: bool = False, fig_path: str = 'plot_of_inference.png', filter_edge: bool = False,
                   filter_margin: int = 32, threshold: float = 0.5, figsize: Tuple[int, int] = (20, 20),
                   gt_colors: Tuple[str, str] = ('g', 'r'), pred_colors: Tuple[str, str] = ('g', 'r'),
                   pred_size: int = 300, print_prediction_numbers: bool = True, show_axis: bool = True,
                   ax=plt, is_multi: bool = False) -> None:
    """
    calculates the statistic of previously run output and plots it

    Parameters
    ----------
    rgb: array
        Image for display with imshow.
    boxes: tensor (float32)
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
    print_prediction_numbers: bool
        true -> print statistic of tp, fp, fn
    show_axis: bool

    Returns
    -------
    None

    """
    scores = tf.reshape(nms_output, -1)
    condition = tf.greater(scores, threshold)
    rpn_boxes = tf.squeeze(tf.gather(rpn_boxes, tf.where(condition)))

    if filter_edge:
        boxes, _ = filter_edge_boxes(boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], filter_margin)

    try:
        ious = iou(rpn_boxes, boxes)

        tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou_mapping(ious, 0.18)

        positive_pred = tf.reshape(tf.gather(rpn_boxes, tp_list[:, 0]), (-1, 4))
        negative_pred = tf.reshape(tf.gather(rpn_boxes, fp_list), (-1, 4))  # tf.where(labels == 0)
        negative_boxes = tf.reshape(tf.gather(boxes, fn_list), (-1, 4))

        if filter_edge:
            negative_boxes, _ = filter_edge_boxes(negative_boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], 
                                                  filter_margin, tf.constant(False, tf.bool))
            positive_pred, _ = filter_edge_boxes(positive_pred, tf.shape(rgb)[1], tf.shape(rgb)[0], 
                                                 filter_margin, tf.constant(False, tf.bool))
            negative_pred, _ = filter_edge_boxes(negative_pred, tf.shape(rgb)[1], tf.shape(rgb)[0], 
                                                 filter_margin, tf.constant(False, tf.bool))

    except:
        positive_pred = tf.constant([])
        negative_pred = tf.constant([])
        negative_boxes = boxes

    if print_prediction_numbers:
        print(f'tp: {tf.shape(positive_pred)[0]}, fp: {tf.shape(negative_pred)[0]}, fn: '
              f'{tf.shape(boxes)[0] - tf.shape(positive_pred)[0]}')

    if not is_multi:
        fig = plt.figure(figsize=figsize)
    ax.imshow(tf.cast(rgb, tf.int32))

    _plot_boxes_multi_plot(boxes, gt_colors[0], ax)  # '#b9ff38'
    _plot_boxes_multi_plot(negative_boxes, gt_colors[1], ax)  # '#b9ff38'

    if tf.size(negative_pred) > 0:
        negative_pred_d = negative_pred[:, :2] + negative_pred[:, 2:] / 2
        ax.scatter(negative_pred_d[:, 0], negative_pred_d[:, 1], color=pred_colors[1], s=pred_size)
    if tf.size(positive_pred) > 0:
        positive_pred_d = positive_pred[:, :2] + positive_pred[:, 2:] / 2
        ax.scatter(positive_pred_d[:, 0], positive_pred_d[:, 1], color=pred_colors[0], s=pred_size)

    if not show_axis:
        ax.axis('off')

    if not is_multi:
        plt.show()
    if save_fig and not is_multi:
        fig.savefig(fig_path)


def _run_model(data, model):
    """
    runs the lnms model

    Parameters
    ----------
    data:
        model input
    model:
        network

    Returns
    -------
    rgb: array
        Image for display with imshow.
    boxes: tensor (float32)
        ground truth boxes
        shape: G x 4
    rpn_boxes: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of each regressed box and its width and height in that order.
    nms_output2: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    """
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

    return rgb, boxes, rpn_boxes, nms_output2


def _plot_boxes_multi_plot(boxes, color, ax):
    """
    plots boxes on specific ax

    Parameters
    ----------
    boxes:
        boxes to plot
    color:
        box color
    ax:
        axis to plot boxes on or plt

    Returns
    -------

    """
    x, y, w, h = tf.split(boxes, 4, axis=1)
    for i, (xi, yi, wi, hi) in enumerate(zip(x, y, w, h)):
        ax.plot([xi, xi + wi, xi + wi, xi, xi],
                [yi, yi, yi + hi, yi + hi, yi],
                color=color)


def run_plot(validation_data, model: tf.keras.Model, index: int = 0, save_fig: bool = False,
             fig_path: str = 'plot_of_inference.png', filter_edge: bool = False, filter_margin: int = 32,
             threshold: float = 0.5, figsize: Tuple[int, int] = (20, 20), gt_colors: Tuple[str, str] = ('g', 'r'),
             pred_colors: Tuple[str, str] = ('g', 'r'), pred_size: int = 300, print_prediction_numbers: bool = True,
             show_axis: bool = True) -> None:
    """
    runs model ond plots the inference

    Parameters
    ----------
    save_fig: bool
        true if picture should be saved
    fig_path: str
        path to save figure at
    filter_edge: bool
        true -> filters boxes and predictions too close to the border of image
    filter_margin: int
        min distance to border not to be filtered out
    threshold: float
    print_prediction_numbers: bool
        true -> print statistic of tp, fp, fn
    figsize: Tuple[int, int]
        dimensions of figure
    gt_colors: Tuple[str, str]
        color of predicted ground truth, color of not predicted ground truth (false negative)
    pred_colors: Tuple[str, str]
        color of true positive prediction, color of false positive prediction
    pred_size: int
        size of dot symbolizing a prediction
    show_axis: bool
    validation_data
    model: tf.keras.Model
        lnms model to run
    index: int
        index of validation image that should be run and plotted

    Returns
    -------

    """

    counter = 0
    for data in validation_data:
        if counter < index:
            counter += 1
            continue

        rgb, boxes, rpn_boxes, nms_output2 = _run_model(data, model)

        plot_inference(rgb, boxes, nms_output2, rpn_boxes, save_fig, fig_path, filter_edge, filter_margin, threshold,
                       figsize, gt_colors, pred_colors, pred_size, print_prediction_numbers, show_axis)
        break


def plot_multiple_outputs(configs: dict, validation_data: tf.data.Dataset, model_paths: List[str], variable: str,
                          model_variations: List, index: int, faster_model: tf.keras.Model, titles: List[str],
                          fig_path: str = 'figures/plot_of_inferences.png') -> None:
    """
    plots multiple outputs of lnms where one hyperparamter changes

    Parameters
    ----------
    configs: dict
        configurations as dictionary
    validation_data:
        validation dataset
    model_paths:
        list of paths to model weights
    variable:
        configs variable that changes
    model_variations:
        list of the values of the variable that changes
    index:
    faster_model:
        faster r-cnn model with loaded weights
    titles:
        list of titles for each plot
    fig_path:
        save fig here

    Returns
    -------

    """
    counter = 0
    for data in validation_data:
        if counter == index:
            data = data
            break
        counter += 1
    #     validation_data = list(validation_data.as_numpy_iterator())
    #     data = validation_data[index]

    rgb, boxes, _ = data

    fig, axs = plt.subplots(1, len(model_paths) + 1, sharey=True, figsize=(30, 20))

    regressions = faster_model(rgb, tau=0.5, nms_iou=0.3)

    scores = tf.ones((tf.shape(regressions)[0], 1))
    plot_inference(rgb, boxes.to_tensor(), scores, regressions, filter_edge=True, is_multi=True, ax=axs[0],
                   pred_size=50, show_axis=False)
    axs[0].title.set_text("NMS")

    compression_net = CompressionNetwork(configs['feature_size'], configs['anchor_size'], faster_model.backbone)

    for i, (path, model_variation) in enumerate(zip(model_paths, model_variations)):
        ax = axs[i + 1]
        ax.title.set_text(titles[i])

        copied_config = configs.copy()
        copied_config[variable] = model_variation

        model = LearningNMS(configs, faster_model.rpnetwork, faster_model.backbone, compression_net.compression_layers,
                            [configs['width'], configs['height']], )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

        model.load_weights(path)

        rgb, boxes, rpn_boxes, nms_output2 = _run_model(data, model)

        plot_inference(rgb, boxes, nms_output2, rpn_boxes, False, '', filter_edge=True, ax=ax, is_multi=True,
                       pred_size=50, show_axis=False)

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.show()
    fig.savefig(fig_path)
