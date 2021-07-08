import tensorflow as tf

from histomics_detect.metrics import iou


def calculate_cluster_assignment(boxes: tf.Tensor, rpn_positive: tf.Tensor, use_centroids: bool = False) -> tf.Tensor:
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
    use_centroids: bool
        are the the boxes only centroids

    Returns
    -------
    cluster_assignment: tensor (int32)
        a list with one element per prediction (rpn_positive)
        that element is the index of the closest ground truth box

    """
    if use_centroids:
        # TODO implement centroid distance metric
        ious, _ = ...
    else:
        ious, _ = iou(rpn_positive, boxes)

    def assign_single_prediction(i) -> tf.int32:
        assignment = tf.argmax(ious[i])
        return assignment

    cluster_assignment = tf.vectorized_map(assign_single_prediction, tf.range(0, tf.shape(rpn_positive)[0]))
    return cluster_assignment
