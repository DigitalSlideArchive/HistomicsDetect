from typing import Union
import tensorflow as tf
import tensorflow.keras.backend as kb

from histomics_detect.metrics import iou


def assemble_single_neighborhood(anchor_id: int, interpolated: tf.Tensor, neighborhood_indeces: tf.Tensor,
                                 neighborhood_additional_info: tf.Tensor, use_image_features: bool = True) \
        -> tf.float32:
    """
    assembles the neighborhood of a single prediction

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    
    Parameters
    ----------
    anchor_id: int
        id of current prediction in [0, N-1]
    interpolated: tensor (float32)
        N x D interpolated features
    neighborhood_indeces: tensor (int32)
        S x 2 indeces of predictions in neighborhood
    neighborhood_additional_info: tensor (float32)
        S x F additional information of neighborhood
    use_image_features: bool
        boolean whether image features should be added to neighborhood

    Returns
    -------
    concatenated_neighborhood: tensor (float32)
        S x 2D ragged tensor of assembled neighborhood
    """
    anchor_id = tf.cast(anchor_id, tf.int32)

    # collect image features
    if use_image_features:
        neighborhood = tf.reshape(tf.gather(interpolated, neighborhood_indeces),
                                  [tf.size(neighborhood_indeces), tf.shape(interpolated)[1]])
        tiled_pred = tf.tile(tf.expand_dims(interpolated[anchor_id], axis=0), (tf.shape(neighborhood)[0], 1))
    elif False:
        # TODO implement interpolate joint feature representation
        pass
    else:
        neighborhood = tf.reshape(tf.gather(interpolated[:, 0], neighborhood_indeces),
                                  [tf.size(neighborhood_indeces), 1])
        tiled_pred = tf.tile(tf.reshape(interpolated[anchor_id, 0], (1, 1)), (tf.shape(neighborhood)[0], 1))

    concatenated_neighborhood = tf.concat([neighborhood, tiled_pred, neighborhood_additional_info], axis=1)

    return concatenated_neighborhood


def single_neighborhood_additional_info(anchor_id, ious, rpn_boxes_positive,
                                        normalization_factor: float,
                                        threshold: Union[float, tf.Tensor]):
    """
    assembles additional information of a certain neighborhood
    also assembles the neighborhood_indeces

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction

    Parameters
    ----------
    anchor_id:
        id of current prediction
    ious:
        N x N ious of predictions with other predictions
    rpn_boxes_positive:
        N x 4 x,y,w,h of the boxes or centroids
    normalization_factor:
        distance normalization devider
    threshold:
        threshold for neighborhood

    Returns
    -------
    additional_info: tensor (float32)
        for each pair of boxes in the neighborhood of the current anchor the following additional information
        is presented in a tensor:
        - iou
        - normalized distance
        - l2 distance
        - scale difference
    neighborhood_indexes: tensor (float32)
        indexes of the other anchors that are in the neighborhood of the current anchor
    """
    anchor_id = tf.cast(anchor_id, tf.int32)

    neighborhood_indexes = tf.where(tf.greater(ious[:, anchor_id], threshold))

    # prepare boxes for collecting additional feature values
    neighborhood_boxes = tf.gather(rpn_boxes_positive, neighborhood_indexes)
    x, y = tf.shape(neighborhood_boxes)[0], tf.shape(neighborhood_boxes)[2]
    neighborhood_boxes = tf.reshape(neighborhood_boxes, (x, y))

    # collect feature values of neighborhood
    collected_ious = tf.gather(ious[:, anchor_id], neighborhood_indexes)
    distances = (neighborhood_boxes[:, :2] - rpn_boxes_positive[anchor_id, :2])
    normalized_distance = tf.abs(distances) / (normalization_factor + 1e-8)
    l2_distance = tf.expand_dims(tf.reduce_sum(distances ** 2, axis=1) ** (1 / 2) / (normalization_factor + 1e-8),
                                 axis=1)

    # concatenate neighborhood vector representations
    scale_difference = kb.log(kb.abs(neighborhood_boxes[:, 2:] / (rpn_boxes_positive[anchor_id, 2:] + 1e-8)) + 1e-8)
    additional_info = tf.concat([collected_ious, normalized_distance,
                                 l2_distance, scale_difference], axis=1)
    return additional_info, neighborhood_indexes


def all_neighborhoods_additional_info(rpn_boxes_positive, prediction_ids,
                                      normalization_factor: float,
                                      threshold: Union[float, tf.Tensor]):
    """
    collect additional info and indexes for all neighborhoods

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction

    Parameters
    ----------
    rpn_boxes_positive: tensor (float32)
        N x 4 x,y,w,h of the boxes or centroids
    prediction_ids: tensor (int32)
        list of ids
    normalization_factor: float
        distance normalization divider
    threshold: float
        threshold for neighborhood

    Returns
    -------
    neighborhood_sizes: tensor (int32)
        the sizes of the neighborhoods of each anchor
    neighborhoods_additional_info: tensor (float32)
        the additional information of each neighborhood collected in one tensor
    neighborhoods_indexes: tensor (int32)
        the indexes of the anchors in each neighborhood
    """

    # calculate distance or ious
    ious, _ = iou(rpn_boxes_positive, rpn_boxes_positive)

    neighborhood_sizes = tf.TensorArray(tf.int32, size=tf.shape(rpn_boxes_positive)[0])
    neighborhoods_additional_info, neighborhoods_indexes = single_neighborhood_additional_info(
        prediction_ids[0], ious, rpn_boxes_positive, normalization_factor, threshold)
    neighborhood_sizes = neighborhood_sizes.write(0, tf.shape(neighborhoods_additional_info)[0])

    # assemble neighborhoods
    for x in prediction_ids[1:]:
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(neighborhoods_additional_info, tf.TensorShape([None, None])),
                              (neighborhoods_indexes, tf.TensorShape([None, None]))])
        new_neighborhood, new_indexes = single_neighborhood_additional_info(
            x, ious, rpn_boxes_positive, normalization_factor, threshold)

        neighborhood_sizes = neighborhood_sizes.write(x, tf.shape(new_neighborhood)[0])
        neighborhoods_additional_info = tf.concat([neighborhoods_additional_info, new_neighborhood], axis=0)
        neighborhoods_indexes = tf.concat([neighborhoods_indexes, new_indexes], axis=0)
    neighborhood_sizes = neighborhood_sizes.stack()

    return neighborhood_sizes, neighborhoods_additional_info, neighborhoods_indexes
