from typing import Tuple

import tensorflow as tf

from histomics_detect.models.model_utils import extract_data
from histomics_detect.models.lnms_model import LearningNMS
from histomics_detect.boxes.match import cluster_assignment
from histomics_detect.metrics import iou


def neighborhood_free_data_formatting(data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], data_model: LearningNMS) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    norm, boxes, sample_weight = extract_data(data)
    data_model.initial_prediction_threshold = -1

    features, rpn_boxes, scores = data_model.extract_boxes_n_scores(norm)

    compressed_features = data_model.compression_net(features, training=False)

    interpolated = data_model._interpolate_features(compressed_features, rpn_boxes)
    interpolated = tf.concat([scores, interpolated], axis=1)

    num_boxes = tf.shape(interpolated)[0]

    x_tiles = tf.tile(tf.expand_dims(interpolated, axis=0), [num_boxes, 1, 1])
    y_tiles = tf.tile(tf.expand_dims(interpolated, axis=1), [1, num_boxes, 1])

    x_boxes = tf.tile(tf.expand_dims(rpn_boxes, axis=0), [num_boxes, 1, 1])
    y_boxes = tf.tile(tf.expand_dims(rpn_boxes, axis=1), [1, num_boxes, 1])

    ious = iou(rpn_boxes, rpn_boxes)
    ious = tf.expand_dims(ious, axis=2)

    new_features = tf.concat((x_tiles, y_tiles, x_boxes, y_boxes, ious), axis=2)

    clusters = cluster_assignment(boxes, rpn_boxes, min_threshold=0.0, apply_threshold=True)

    x_cluster = tf.tile(tf.expand_dims(clusters, axis=0), [num_boxes, 1])
    y_cluster = tf.tile(tf.expand_dims(clusters, axis=1), [1, num_boxes])

    labels = tf.cast(tf.logical_and(x_cluster == y_cluster, x_cluster != -1), tf.float32)
    labels = tf.expand_dims(labels, axis=2)

    return new_features, labels
