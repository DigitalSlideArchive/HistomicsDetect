from typing import Union
import tensorflow as tf
import tensorflow.keras.backend as kb

from histomics_detect.metrics.iou import iou


def assemble_single_neighborhood(anchor_id: int, interpolated: tf.Tensor, neighborhood_indeces: tf.Tensor,
                                 neighborhood_additional_info: tf.Tensor, use_image_features: bool = True,
                                 use_joint_features: bool = False, use_cross_feature: bool = False) \
        -> tf.float32:
    """
    DEPRECIATED: unused

    assembles the prediction representations for a neighborhood of a single prediction

    the neighborhood assembly of a single prediction consists of
    - collecting the neighboring prediction representation (initially the interpolated
      features, later the output of the previous block) and concatenating those to the
      single prediction representation
    - collecting additional information (iou, normalized distance, l2 distance, scale difference)
      for each pair of predictions in the current neighborhood and concatenating it to the corresponding
      prediction representations.
      This additional information is NOT calculated here but has to be passed to the function by
      'neighborhood_additional_info'

    detailed steps:
    - collect all the prediction representations from 'interpolated' corresponding to boxes
      that are in the neighborhood of the current prediction 'anchor_id'
    - then the prediction representation of the current prediction is tiled and concatenated to
      the previously collected prediction representations
    - the 'neighborhood_additional_info' (consists of iou, normalized distance, l2 distance,
      scale difference in that order) is concatenated

    This function is called for each prediction for each lnms block the output is passed through the block

    S: size of neighborhood
    N: number of predictions
    D: size of a single prediction
    
    Parameters
    ----------
    use_joint_features: bool
        creates a bounding box spanning the neighboring boxes and interpolates the features for that
    use_cross_feature: bool
        creates a cross shape out of two intersecting bounding boxes spanning from the center until specified end or
        image boundary
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
        # TODO implement joint feature, and cross feature
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
    assembles additional information of the neighborhood of a single prediction 'anchor_id'
    also assembles the neighborhood_indexes

    detailed steps:
    - find the boxes that have an iou of bigger then 'threshold' with the current box 'anchor_id'
    - store the indexes of those boxes in a list 'neighborhood_indexes'
    - collect all the boxes that are in the neighborhood
    - collect the iou of the boxes in the neighborhood with the current box 'anchor_id'
    - calculate the distances (in x and y direction separately) of the boxes and the current box
    - calculate the l2 distance from that
    - calculate the scale difference
    - concatenate all the collected information in 'additional_info'


    S: size of neighborhood
    N: number of predictionsadditional_info
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

    # find indexes of the boxes that are in the neighborhood of the current box
    neighborhood_indexes = tf.where(tf.greater(ious[:, anchor_id], threshold))

    # collect the boxes that are in the neighborhood of the current prediction
    neighborhood_boxes = tf.gather(rpn_boxes_positive, neighborhood_indexes)
    x, y = tf.shape(neighborhood_boxes)[0], tf.shape(neighborhood_boxes)[2]
    neighborhood_boxes = tf.reshape(neighborhood_boxes, (x, y))

    # calculate additional information for the neighborhood
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


def all_neighborhoods_additional_info(rpn_boxes_positive, prediction_ids, normalization_factor: float,
                                      threshold: Union[float, tf.Tensor], use_distance: bool = False):
    """
    collect additional info and indexes for all neighborhoods by running the method
    'single_neighborhood_additional_info' for each prediction and reformatting the collected data

    neighborhoods of different predictions have different sizes, but RaggedTensors are not ideal.
    Therefore, some reformatting has to done and the neighborhood size has to be stored

    detailed steps:
    - calculate iou for all predictions
    - iterate for each prediction
        + collect additional info and neighborhood indexes with 'single_neighborhood_additional_info'
        + save size of neighborhood (e.g. number of predictions in the neighborhood) in 'neighborhood_sizes'
        + concatenate additional info and indexes for current neighborhood to already collected ones

    there is no clear way to differentiate between neighborhoods in the 'neighborhoods_additional_info' or
    'neighborhoods_indexes' tensors. To find out what belongs to which neighborhood the tensor
    'neighborhood_sizes' has to be used.

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
    use_distance: bool
        use distance for neighborhood assembly instead of iou

    Returns
    -------
    neighborhood_sizes: tensor (int32)
        the sizes of the neighborhoods of each anchor
    neighborhoods_additional_info: tensor (float32)
        the additional information of each neighborhood collected in one tensor
    neighborhoods_indexes: tensor (int32)
        the indexes of the anchors in each neighborhood
    self_indexes: tensor (int32)
        for each neighborhood the index of the main box is tiled by the number of other predictions in
        the neighborhood and concatenated with the others
    """

    # calculate distance or ious
    if use_distance:
        centroids = rpn_boxes_positive[:, :2] + rpn_boxes_positive[:, 2:]/2
        dist_x = (tf.expand_dims(centroids[:, 0], axis=1) - tf.expand_dims(centroids[:, 0], axis=0))
        dist_y = (tf.expand_dims(centroids[:, 1], axis=1) - tf.expand_dims(centroids[:, 1], axis=0))
        ious = 1/(dist_x**2 + dist_y**2)
    else:
        ious = iou(rpn_boxes_positive, rpn_boxes_positive)

    # assemble initial information
    neighborhood_sizes = tf.TensorArray(tf.int32, size=tf.shape(rpn_boxes_positive)[0])
    neighborhoods_additional_info, neighborhoods_indexes = single_neighborhood_additional_info(
        prediction_ids[0], ious, rpn_boxes_positive, normalization_factor, threshold)

    current_neighborhood_size = tf.shape(neighborhoods_additional_info)[0]
    neighborhood_sizes = neighborhood_sizes.write(0, current_neighborhood_size)
    # self_indexes = tf.expand_dims(tf.tile(0, [current_neighborhood_size]), axis=1)
    self_indexes = tf.expand_dims(tf.tile(tf.constant([0]), [current_neighborhood_size]), axis=1)

    # assemble neighborhoods for each prediction
    for x in prediction_ids[1:]:
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(neighborhoods_additional_info, tf.TensorShape([None, None])),
                              (neighborhoods_indexes, tf.TensorShape([None, None])),
                              (self_indexes, tf.TensorShape([None, None]))])
        new_neighborhood, new_indexes = single_neighborhood_additional_info(
            x, ious, rpn_boxes_positive, normalization_factor, threshold)

        current_neighborhood_size = tf.shape(new_neighborhood)[0]
        neighborhood_sizes = neighborhood_sizes.write(x, current_neighborhood_size)

        neighborhoods_additional_info = tf.concat([neighborhoods_additional_info, new_neighborhood], axis=0)
        neighborhoods_indexes = tf.concat([neighborhoods_indexes, new_indexes], axis=0)

        current_self_indexes = tf.expand_dims(tf.tile(tf.expand_dims(x, axis=0), [current_neighborhood_size]), axis=1)
        self_indexes = tf.concat([self_indexes, current_self_indexes], axis=0)
    neighborhood_sizes = neighborhood_sizes.stack()

    return neighborhood_sizes, neighborhoods_additional_info, neighborhoods_indexes, self_indexes
