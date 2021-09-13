import tensorflow as tf
from typing import Tuple, Union
from scipy.optimize import linear_sum_assignment

from histomics_detect.metrics.iou import iou


@tf.function
def tf_linear_sum_assignment(boxes, rpn_boxes):
    """
    tensorflow wrapper for linear sum assignment function from scipy.optimice

    Parameters
    ----------
    boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the ground truth boxes
    rpn_boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the
        predictions.

    Returns
    -------
    true_rpn_box_indexes: tensor (int32)
        returns the indexes of the rpn_boxes assigned to the corresponding gt box

    """
    ious, _ = iou(boxes, rpn_boxes)
    out = tf.numpy_function(linear_sum_assignment, [ious, tf.constant(True)], [tf.int64, tf.int64])
    row_ind, col_ind = out[0], out[1]

    return tf.expand_dims(tf.cast(col_ind, tf.int32), axis=1)


def lnms_metrics(boxes: tf.Tensor, rpn_boxes: tf.Tensor, scores: tf.Tensor, min_threshold: float = 0.18,
                 apply_threshold: bool = True, score_threshold: float = 0.5, return_indexes: bool = False) \
        -> Union[Tuple[int, int, int], Tuple[int, int, int, tf.Tensor, tf.Tensor, tf.Tensor]]:
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
    min_threshold: float
        if box has no ground truth with an iou higher than 'min_threshold' this box is considered an outlier
        and is not assigned to a cluster
    apply_threshold: bool
        requires predictions to have at least iou 'min_threshold' to be part of any cluster
    score_threshold: float
        min score such that corresponding prediction is considered positive
    return_indexes: bool
        if true indexes of the tp, fp, fn are also returned

    Returns
    -------
    tp: int
        true positives
    fp: int
        false positives
    fn: int
        false negatives
    tp_indexes: tensor
        indexes of the predictions that are true positives
    fp_indexes: tensor
        indexes of the predictions that are false positives
    fn_indexes: tensor
        indexes of the boxes that are false negatives

    """
    # in function imports needed because of circular imports
    from histomics_detect.boxes.match import cluster_assignment
    from histomics_detect.models.lnms_loss import cluster_labels_indexes

    clusters = cluster_assignment(boxes, rpn_boxes, min_threshold, apply_threshold)
    labels, _ = cluster_labels_indexes(scores, clusters)

    rpn_boxes = tf.squeeze(tf.gather(rpn_boxes, tf.where(tf.greater(tf.reshape(scores, -1), score_threshold))))

    ious, _ = iou(boxes, rpn_boxes)

    tp_condition = tf.reduce_max(ious, axis=1) > min_threshold

    tp = tf.reduce_sum(tf.cast(tp_condition, tf.int32))
    fn = tf.shape(boxes)[0] - tp
    fp = tf.shape(rpn_boxes)[0] - tp

    if return_indexes:
        tp_indexes = tf.where(labels == 1)
        fp_indexes = tf.where(labels == 0)
        fn_indexes = tf.where(tp_condition == False)

        return tp, fp, fn, tp_indexes, fp_indexes, fn_indexes

    return tp, fp, fn
