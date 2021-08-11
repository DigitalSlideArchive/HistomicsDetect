import tensorflow as tf

from histomics_detect.metrics import iou


def cluster_assignment(boxes: tf.Tensor, rpn_positive: tf.Tensor, min_threshold: float = 0.0) -> tf.Tensor:
    """
    calculates the cluster assignment of the predictions to the ground truth boxes
    a cluster is a group of predictions all of which are closest to the same ground truth box

    !!Assumption: each ground truth box has at least one corresponding prediction (e.g. ~ N >> G)
                  this assumption does not need to hold for this function to run, but it needs to hold
                  for the output to make sense

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    G: number of ground truth boxes

    Parameters
    ----------
    boxes: tensor (float32)
        ground truth boxes
        shape: G x 4
    rpn_positive: tensor (float32)
        predictions
        shape: N x 4
    min_threshold: float
        if box has no ground truth with an iou higher than 'min_threshold' this box is considered an outlier
        and is not assigned to a cluster

    Returns
    -------
    clusters: tensor (int32)
        a list with one element per prediction (rpn_positive)
        that element is the index of the closest ground truth box

    """
    ious, _ = iou(rpn_positive, boxes)

    def assign_single_prediction(i) -> tf.int32:
        assignment = tf.cast(tf.argmax(ious[i]), tf.int32)
        assignment = tf.cond(ious[i, assignment] > min_threshold, lambda: assignment,
                             lambda: tf.constant(-1, dtype=tf.int32))
        return assignment

    clusters = tf.map_fn(assign_single_prediction, tf.range(0, tf.shape(rpn_positive)[0]))
    # clusters = tf.cond(tf.reduce_max(clusters) == -1, lambda: clusters + 1, lambda: clusters)
    return tf.cast(clusters, tf.int32)
