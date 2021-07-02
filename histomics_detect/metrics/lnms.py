import tensorflow as tf
from typing import Tuple

from histomics_detect.metrics import iou


def calculate_performance_stats_lnms(boxes: tf.Tensor, rpn_boxes: tf.Tensor, scores: tf.Tensor) -> Tuple[int, int, int, int]:
    """
    calculates the number of true positives, false positives, true negatives, false negatives of the
    score prediction

    Parameters
    ----------
    boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the ground truth boxes
    rpn_boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the
        predictions.
    scores: tensor (float32)
        M x 1 tensor where each row contains the objectiveness score of the corresponding
        rpn_boxes

    Returns
    -------
    tp: int
        true positives
    tn: int
        true negatives
    fp: int
        false positives
    fn: int
        false negatives

    """

    ious, _ = iou(boxes, rpn_boxes)

    def func(i) -> tf.int32:
        index = tf.cast(i, tf.int32)
        assignment = tf.cast(tf.argmax(ious[index]), tf.int32)
        return assignment

    indeces = tf.expand_dims(tf.map_fn(lambda x: func(x), tf.range(0, tf.shape(ious)[0])), axis=1)
    labels = tf.scatter_nd(indeces, tf.ones(tf.shape(indeces)), tf.shape(scores))
    print(indeces)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores > 0.5), tf.int32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores <= 0.5), tf.int32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores > 0.5), tf.int32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores <= 0.5), tf.int32))

    return tp, tn, fp, fn
