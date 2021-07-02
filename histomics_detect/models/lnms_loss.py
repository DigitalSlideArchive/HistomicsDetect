import tensorflow as tf
from typing import List

from histomics_detect.metrics import iou


def normal_loss(loss_object, boxes: tf.Tensor, rpn_boxes_positive: tf.Tensor, nms_output: tf.Tensor, positive_weight: float,
                standard: List[tf.keras.metrics] = [], weighted_loss: bool = False, neg_pos_loss: bool = False, use_pos_neg_loss: bool = False):
    """
    calculates the normal loss of a lnms output

    labels are calculated based on the largest iou, the prediction that is closest to the respective
    ground truth gets assigned a 1 label and the rest a 0

    then a loss is applied to the objectiveness score output 'nms_output' and the labels

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    G: number of ground truth boxes

    Parameters
    ----------
    loss_object:
        loss function for loss calculation between 'labels' and 'nms_output'
    boxes: tensor (float32)
        ground truth boxes
        shape: G x 4
    rpn_boxes_positive: tensor (float32)
        predicted boxes
        shape: N x 4
    nms_output: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    positive_weight: float
        weight applied to the positive labels ( == 1)
    standard: metric
        list of tensorflow metrics
        1, 2 should be positive and negative loss respectively if 'neg_pos_loss' set to true
    weighted_loss: bool
        if true, loss of positive labels is weighted by the difference in numbers of positive and negative
        labels
    neg_pos_loss: bool
        if true, the loss of the positive and the negative labels is calculated and logged in the metrics
    use_pos_neg_loss: bool
        returns the weighted sum of the pos and neg loss instead of the normal loss
        !!! only works if neg_pos_loss is also true

    Returns
    -------
    loss: float
        loss value
    indexes: tensor (float32)
        indexes of the values that correspond to positive anchors
    """
    ious, _ = iou(boxes, rpn_boxes_positive)

    def func(i) -> tf.int32:
        index = tf.cast(i, tf.int32)
        assignment = tf.cast(tf.argmax(ious[index]), tf.int32)
        return assignment

    indexes = tf.map_fn(lambda x: func(x), tf.range(0, tf.shape(ious)[0]))
    indexes = tf.expand_dims(indexes, axis=1)
    labels = tf.scatter_nd(indexes, tf.ones(tf.shape(indexes)), tf.shape(nms_output))

    if neg_pos_loss:
        positive_predictions, negative_predictions = tf.dynamic_partition(nms_output, tf.cast(labels == 0, tf.int32), 2)
        positive_labels = tf.ones(tf.shape(positive_predictions))
        negative_labels = tf.zeros(tf.shape(negative_predictions))

        pos_loss = tf.reduce_sum(loss_object(positive_predictions, positive_labels))
        neg_loss = tf.reduce_sum(loss_object(negative_predictions, negative_labels))

        standard[1].update_state(pos_loss + 1e-8)
        standard[2].update_state(neg_loss + 1e-8)

        if use_pos_neg_loss:
            return pos_loss * positive_weight + neg_loss, indexes

    if weighted_loss:
        num_pos = tf.cast(tf.size(positive_labels), tf.float32)
        num_neg = tf.cast(tf.size(negative_labels), tf.float32)
        weighted_labels = tf.cast(labels, tf.float32) * num_neg / num_pos * positive_weight
        weight = weighted_labels + (1 - labels)

        loss = loss_object(weighted_labels, nms_output*weight)
    else:
        loss = loss_object(labels, nms_output)

    return tf.reduce_sum(loss), indexes
