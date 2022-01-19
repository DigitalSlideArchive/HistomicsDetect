import tensorflow as tf
import tensorflow.keras.backend as kb
from typing import List, Tuple

from histomics_detect.metrics.iou import iou, greedy_iou_mapping


def normal_loss(loss_object: tf.keras.losses.Loss, boxes: tf.Tensor, rpn_boxes_positive: tf.Tensor,
                scores: tf.Tensor, positive_weight: float, standard: List[tf.keras.metrics.Metric] = [],
                weighted_loss: bool = False, neg_pos_loss: bool = False, use_pos_neg_loss: bool = False,
                min_iou: float = 0.18) \
        -> Tuple[tf.Tensor, tf.Tensor]:
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
    scores: tensor (float32)
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
    min_iou: float
        minimum iou such that box is considered positive prediction

    Returns
    -------
    loss: float
        loss value
    indexes: tensor (float32)
        indexes of the values that correspond to positive anchors
    """
    labels, indexes = calculate_labels(boxes, rpn_boxes_positive, tf.shape(scores), min_iou)

    # calculate negative and positive labels loss for comparing experiment
    if neg_pos_loss:
        (pos_loss, neg_loss), (positive_labels, negative_labels) = _pos_neg_loss_calculation(scores, labels,
                                                                                             loss_object, standard)
        # use negative or positive for training model
        if use_pos_neg_loss:
            return pos_loss * positive_weight + neg_loss, indexes

    # weigh loss
    if weighted_loss:
        num_pos = tf.cast(tf.size(positive_labels), tf.float32)
        num_neg = tf.cast(tf.size(negative_labels), tf.float32)
        weighted_labels = tf.cast(labels, tf.float32) * num_neg / num_pos * positive_weight
        weight = weighted_labels + (1 - labels)

        loss = loss_object(weighted_labels, scores * weight)
    else:
        loss = loss_object(labels, scores)

    return tf.reduce_sum(loss), indexes


def paper_loss(boxes: tf.Tensor, rpn_boxes_positive: tf.Tensor, nms_output: tf.Tensor,
               loss_object: tf.keras.losses.Loss, positive_weight: float, standard: List[tf.keras.metrics.Metric],
               weighted_loss: bool = False, neg_pos_loss: bool = False, min_iou: float = 0.18) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    loss calculation of the paper "Learning Non-Max Suppression"

    the loss is calculated with:
    - the labels vector l with 1s for positive labels and -1 for negative labels
    - the score output of the network n with values btw -1 and 1

    - calculation: positive_label_weight * log(1 + exp(-l * n))

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
        if true, the loss of the positive and the negative labels is calculated and logged in the metrixes
    min_iou: float
        minimum iou such that box is considered positive prediction

    Returns
    -------
    loss: float
        loss value
    indexes: tensor (float32)
        indexes of the values that correspond to positive anchors

    """
    labels, indexes = calculate_labels(boxes, rpn_boxes_positive, tf.shape(nms_output), min_iou)

    # calculate pos and neg loss
    if weighted_loss or neg_pos_loss:
        _, (positive_labels, negative_labels) = _pos_neg_loss_calculation(nms_output, labels, loss_object, standard)

    if weighted_loss:
        num_pos = tf.cast(tf.size(positive_labels), tf.float32)
        num_neg = tf.cast(tf.size(negative_labels), tf.float32)

        weight = labels * num_neg / (num_pos + 1e-8) * positive_weight + (1 - labels)
    else:
        weight = tf.ones(tf.shape(nms_output))

    # reformat labels and output from 0, 1 space to -1, 1 space
    labels = (2 * labels - 1)
    nms_output = (2 * nms_output) - 1

    # calculate loss
    loss = weight * kb.log(1 + kb.exp(-labels * nms_output))
    loss = tf.reduce_sum(loss)
    return loss, indexes


def calculate_labels(boxes, rpn_boxes_positive, output_shape, min_iou: float = 0.18):
    """
    calculate the labels for the predictions
    each ground truth has one positive predictions (label = 1) and the other predictions are
    negative (label = 0)

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    G: number of ground truth boxes

    Parameters
    ----------
    boxes: tensor (float32)
        ground truth boxes
        shape: G x 4
    rpn_boxes_positive: tensor (float32)
        predicted boxes
        shape: N x 4
    output_shape: tensor (int32)
        shape of the label output
    min_iou: float
        minimum iou such that box is considered positive prediction

    Returns
    -------
    labels: tensor (int32)
        tensor with one entry per prediction
        1 -> if prediction is corresponding to a ground truth
        0 -> if prediction is not corresponding to a ground truth
    indexes: tensor (int32)
        indexes of the predictions that are positive
    """
    ious = iou(rpn_boxes_positive, boxes)

    tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou_mapping(ious, min_iou)

    indexes = tf.reshape(tp_list[:, 0], (-1, 1))
    labels = tf.scatter_nd(indexes, tf.ones(tf.shape(indexes)), output_shape)

    # ious, _ = iou(boxes, rpn_boxes_positive)

    # function that finds prediction with highest overlap with ground truth
    # def assignment_func(i) -> tf.int32:
    #     index = tf.cast(i, tf.int32)
    #     assignment = tf.cast(tf.argmax(ious[index]), tf.int32)
    #     return assignment
    #
    # indexes = tf.map_fn(lambda x: assignment_func(x), tf.range(0, tf.shape(ious)[0]))
    # indexes = tf.expand_dims(indexes, axis=1)
    # labels = tf.scatter_nd(indexes, tf.ones(tf.shape(indexes)), output_shape)

    return labels, indexes


def _pos_neg_loss_calculation(nms_output: tf.Tensor, labels: tf.Tensor, loss_object: tf.keras.losses.Loss,
                              standard: List[tf.keras.metrics.Metric]) \
        -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    G: number of ground truth boxes

    Parameters
    ----------
    nms_output: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    labels: tensor (int32)
        ground truth labels of corresponding s
        shape: N x 1
    loss_object:
        loss function for loss calculation between 'labels' and 'nms_output'
    standard: metric
        list of tensorflow metrics
        1, 2 should be positive and negative loss respectively if 'neg_pos_loss' set to true

    Returns
    -------
    pos_loss: tensor (float32)
        scalar value
    neg_loss: tensor (float32)
        scalar value
    positive_labels: tensor (int32)
        ones for the number of positive ground truth samples
    negative_labels:
        zeros for the number of positive ground truth samples

    """

    positive_predictions, negative_predictions = tf.dynamic_partition(nms_output, tf.cast(labels == 0, tf.int32), 2)
    positive_labels = tf.ones(tf.shape(positive_predictions))
    negative_labels = tf.zeros(tf.shape(negative_predictions))

    # calculate loss
    pos_loss = tf.reduce_sum(loss_object(positive_predictions, positive_labels))
    neg_loss = tf.reduce_sum(loss_object(negative_predictions, negative_labels))

    zero_func = lambda: 0.0
    pos_loss = tf.cond(tf.size(positive_labels) > 0, lambda: pos_loss, zero_func)
    neg_loss = tf.cond(tf.size(negative_labels) > 0, lambda: neg_loss, zero_func)

    # update metrics
    standard[1].update_state(pos_loss + 1e-8)
    standard[2].update_state(neg_loss + 1e-8)

    return (pos_loss, neg_loss), (positive_labels, negative_labels)


def cluster_labels_indexes(scores, cluster_assignment) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    calculate the labels for the predictions based on clusters and scores

    the 'cluster_assignment' relates predictions to a cluster for each ground truth
    for each cluster the prediction with the highest score is assigned a positive label (label = 1)
    the rest is assigned a negative label (label = 0)

    N: number of predictions

    Parameters
    ----------
    scores: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    cluster_assignment: tensor (int32)
        cluster labels for each prediction
        shape: N x 1

    Returns
    -------

    """
    cluster_assignment = tf.expand_dims(cluster_assignment, axis=1)

    # find prediction index with highest objectiveness in cluster
    def max_cluster_index_func(i) -> tf.int32:
        index = tf.cast(i, tf.int32)
        max_index = tf.argmax(
            tf.multiply(tf.cast(scores, tf.float32),
                        tf.cast(tf.equal(cluster_assignment, tf.cast(index, tf.int32)),
                                tf.float32)))
        return tf.cast(max_index, tf.int32)

    indexes = tf.map_fn(lambda x: max_cluster_index_func(x), tf.range(0, tf.reduce_max(cluster_assignment) + 1))

    labels = tf.scatter_nd(indexes, tf.ones(tf.shape(indexes)), tf.shape(scores))

    return labels, indexes


def clustering_loss(nms_output: tf.Tensor, cluster_assignment: tf.Tensor, loss_object: tf.keras.losses.Loss,
                    positive_weight: float, standard: List[tf.keras.metrics.Metric], boxes: tf.Tensor,
                    rpn_positive: tf.Tensor, weighted_loss: bool = False, neg_pos_loss: bool = False,
                    add_regression_param: int = 0) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    clustering loss calculation

    the loss is calculated by:
    - for each cluster the prediction with the highest objectiveness score is stored
    - the index of the stored predictions is set to one in a labels vector
    - the values of the other indexes are 0
    - the loss is calculated by calculating the difference btw. the labels and the nms_output

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    G: number of ground truth boxes

    Parameters
    ----------
    loss_object:
        loss function for loss calculation between 'labels' and 'nms_output'
    nms_output: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    cluster_assignment: tensor (int32)
        cluster labels for each prediction
        shape: N x 1
    positive_weight: float
        weight applied to the positive labels ( == 1)
    standard: metric
        list of tensorflow metrics
        1, 2 should be positive and negative loss respectively if 'neg_pos_loss' set to true
    boxes: tensor (int32)
        ground truth boxes
    rpn_positive: tensor (float32)
        predicted boxes
    weighted_loss: bool
        if true, loss of positive labels is weighted by the difference in numbers of positive and negative
        labels
    neg_pos_loss: bool
        if true, the loss of the positive and the negative labels is calculated and logged in the metrics
    add_regression_param: int
        0 -> lnms only predicts a single obj. score
        1 -> lnms also regresses the center of the boxes
        2 -> lnms regresses the full boxes
    # TODO add weighting for regression vs score loss
    Returns
    -------
    loss: float
        loss value
    indexes: tensor (float32)
        indexes of the values that correspond to positive anchors
    """
    scores = tf.expand_dims(nms_output[:, 0], axis=1)

    labels, indeces = cluster_labels_indexes(scores, cluster_assignment)

    # calculate pos and neg loss
    if neg_pos_loss:
        _pos_neg_loss_calculation(scores, labels, loss_object, standard)

    if weighted_loss:
        weight = labels * positive_weight + (1 - labels)
    else:
        weight = tf.ones(tf.shape(scores))

    if add_regression_param > 0:
        reg = nms_output[:, 1:add_regression_param * 2 + 1]

        def pos_prediction_dist_func(i) -> tf.float32:
            index = tf.cast(i, tf.int32)
            cluster_scores = tf.multiply(tf.cast(scores, tf.float32),
                                         tf.cast(tf.equal(cluster_assignment, tf.cast(index, tf.int32)),
                                                 tf.float32))
            max_index = tf.cast(tf.argmax(cluster_scores), tf.int32)[0]

            dist = tf.math.sigmoid(
                (boxes[index, :add_regression_param * 2] - rpn_positive[max_index, :add_regression_param * 2]) / 100)

            return tf.cast(dist, tf.float32)

        distances = tf.map_fn(lambda x: pos_prediction_dist_func(x),
                              tf.cast(tf.range(0, tf.reduce_max(cluster_assignment) + 1), tf.float32))

        distance_vector = tf.scatter_nd(indeces, distances, tf.shape(reg))

        loss_score = loss_object(weight * labels, weight * scores)
        loss_reg = loss_object(distance_vector, labels * reg)

        return tf.reduce_sum(loss_score + loss_reg), labels
    else:
        loss = loss_object(weight * labels, weight * scores)
        return tf.reduce_sum(loss), labels


def normal_clustering_loss(nms_output: tf.Tensor, boxes: tf.Tensor, rpn_boxes_positive: tf.Tensor,
                           cluster_assignment: tf.Tensor, loss_object: tf.keras.losses.Loss,
                           positive_weight: float, standard: List[tf.keras.metrics.Metric], weighted_loss: bool = False,
                           neg_pos_loss: bool = False, use_pos_neg_loss: bool = False, norm_loss_weight: float = 1,
                           add_regression_param: int = 0, min_iou: float = 0.18) -> Tuple[float, tf.Tensor]:
    """
    a combination between the normal and clustering loss

    loss = 'norm_loss_weight' * normal_loss + clustering_loss

    Parameters
    ----------
    nms_output: tensor (float32)
        objectiveness scores corresponding to the predicted boxes after lnms processing
        shape: N x 1
    boxes: tensor (float32)
        ground truth boxes
        shape: G x 4
    rpn_boxes_positive: tensor (float32)
        predicted boxes
        shape: N x 4
    cluster_assignment: tensor (int32)
        cluster labels for each prediction
        shape: N x 1
    loss_object:
        loss function for loss calculation between 'labels' and 'nms_output'
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
    norm_loss_weight: float
        weight of the normal loss
    add_regression_param: int
        0 -> lnms only predicts a single obj. score
        1 -> lnms also regresses the center of the boxes
        2 -> lnms regresses the full boxes
    min_iou: float
        minimum iou such that box is considered positive prediction
    # TODO add weighting for regression vs score loss

    Returns
    -------
    loss: float
        the combined loss
    indexes: tensor (float32)
        indexes of the values that correspond to positive anchors
    """
    scores = tf.expand_dims(nms_output[:, 0], axis=1)
    norm_loss, indexes = normal_loss(loss_object, boxes, rpn_boxes_positive, scores, positive_weight, standard,
                                     weighted_loss, neg_pos_loss, use_pos_neg_loss, min_iou)
    clust_loss, _ = clustering_loss(nms_output, cluster_assignment, loss_object, positive_weight, standard, boxes,
                                    rpn_boxes_positive, weighted_loss, neg_pos_loss, add_regression_param)

    loss = norm_loss_weight * norm_loss + clust_loss

    return loss, indexes


def xor_loss(nms_output: tf.Tensor, cluster_assignment: tf.Tensor):
    """
    xor loss
    the loss is minimal if only one score of each cluster is one and the others are zero

    calculation for each cluster:
    - calculate cluster sum
    - subtract one and square result

    calculate for each prediction
    - subtract 1/2 from the score
    - square the result
    - subtract from previous result

    sum over all prediction losses

    Parameters
    ----------
    nms_output: tensor (float32)
        output scores for each prediction
    cluster_assignment: tensor (int32)
        assignment of each prediction to the corresponding cluster

    Returns
    -------
    loss: float
        calculated loss
    """

    # TODO find error cause
    # TODO add optional neg pos loss calculation

    def cluster_sum(i) -> tf.float32:
        pred_indexes = tf.where(tf.equal(tf.cast(cluster_assignment, tf.float32), tf.cast(i, tf.float32)))
        predictions = tf.gather_nd(nms_output, pred_indexes)

        sum_req = (tf.reduce_sum(predictions) - 1) ** 2

        indexes = tf.cast(pred_indexes, tf.int64)
        update_shape = tf.cast(tf.shape(cluster_assignment), tf.int64)

        false_fn = lambda: tf.scatter_nd(indexes, tf.ones(tf.shape(indexes)[0]) * sum_req, update_shape)
        scattered_sum = tf.cond(tf.size(indexes) == 0, lambda: tf.zeros(update_shape), false_fn)
        return tf.squeeze(scattered_sum)

    number_clusters = tf.reduce_max(cluster_assignment) + 1
    number_predictions = tf.shape(cluster_assignment)[0]

    output_signature = tf.TensorSpec.from_tensor(tf.ones(number_predictions, dtype=tf.float32))

    cluster_sums = tf.map_fn(lambda x: cluster_sum(x), tf.range(0, number_clusters), dtype=output_signature)
    cluster_sums = tf.expand_dims(tf.reduce_sum(cluster_sums, axis=0), axis=1)

    loss = tf.reduce_sum((cluster_sums - 1) ** 2 - (nms_output - 0.5) ** 2, axis=0)
    return loss, None
