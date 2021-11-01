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
    ious = iou(boxes, rpn_boxes)
    out = tf.numpy_function(linear_sum_assignment, [ious, tf.constant(True)], [tf.int64, tf.int64])
    row_ind, col_ind = out[0], out[1]

    return tf.expand_dims(tf.cast(col_ind, tf.int32), axis=1)
