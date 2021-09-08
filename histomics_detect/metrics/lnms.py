import tensorflow as tf
from typing import Tuple


def lnms_metrics(boxes: tf.Tensor, rpn_boxes: tf.Tensor, scores: tf.Tensor) \
        -> Tuple[int, int, int, int]:
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
    from histomics_detect.boxes.match import cluster_assignment
    from histomics_detect.models.lnms_loss import cluster_labels_indexes

    clusters = cluster_assignment(boxes, rpn_boxes)
    labels, _ = cluster_labels_indexes(scores, clusters)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores > 0.5), tf.int32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores <= 0.5), tf.int32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores > 0.5), tf.int32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores <= 0.5), tf.int32))

    return tp, tn, fp, fn
