import tensorflow as tf

from histomics_detect.models.faster_rcnn import map_outputs
from histomics_detect.boxes.transforms import unparameterize


def extract_data(data):
    """
    extracts image and boxes from the data

    Parameters
    ----------
    data: Tuple
        the image, the ground truth boxes, and the name of the image

    Returns
    -------
    norm: tensor (float32)
        normalized image
    boxes: tensor (float32)
        shape: D x 4
        ground truth boxes for the given image
    image_name: string
        name of the image
    """
    if len(data) == 3:
        rgb, boxes, image_name = data
    else:
        rgb, boxes = data
        image_name = None

    rgb = tf.squeeze(rgb)

    # convert boxes from RaggedTensor
    expand_fn = lambda x: tf.expand_dims(x, axis=0)
    boxes = boxes.to_tensor()
    boxes = tf.squeeze(boxes)
    boxes = tf.cond(tf.size(tf.shape(boxes)) == 1, lambda: expand_fn(boxes), lambda: boxes)

    # normalize image
    norm = tf.keras.applications.resnet.preprocess_input(tf.cast(rgb, tf.float32))

    # expand dimensions
    norm = tf.expand_dims(norm, axis=0)

    return norm, boxes, image_name


def extract_boxes_n_scores(norm: tf.Tensor, backbone: tf.keras.Model, rpnetwork: tf.keras.Model, anchors: tf.Tensor,
                           anchor_px: tf.Tensor, field: int, initial_prediction_threshold: float = 0.3):
    """
    extracts rpn_boxes and corresponding objectiveness scores with the faster r-cnn network


    - N: number of predictions
    - d: feature dimensions

    Parameters
    ----------
    norm: tensor (float32)
        normalized image
    backbone: tf.keras.Model
        backbone network usually resnet-50
    rpnetwork: tf.keras.Model
        region proposal network
    anchors: tf.Tensor
        shape: M x 4
        anchors for the region proposal network
    anchor_px: tf.Tensor
        shape: K
        anchor width and height ratios
    field: int
        field size
    initial_prediction_threshold: float
        objectiveness threshold for predictions to be passed to the network
    Returns
    -------
    features: tensor (float32)
        shape: N x d
        features extracted from the image for each box
    rpn_boxes: tensor (float32)
        shape: N x 4
        each prediction in box form
    scores: tensor (float32)
        shape: N x 1
        objectiveness score for each prediction
    """
    # extract features and rpn boxes from image
    features = backbone(norm, training=False)
    outputs = rpnetwork(features, training=False)

    rpn_reg = map_outputs(outputs[1], anchors, anchor_px, field)
    rpn_boxes = unparameterize(rpn_reg, anchors)

    # get objectiveness scores
    rpn_obj = tf.nn.softmax(map_outputs(outputs[0], anchors, anchor_px, field))
    scores = rpn_obj[:, 1] / (tf.reduce_sum(rpn_obj, axis=1))

    # filter out negative predictions
    condition = tf.where(tf.greater(scores, initial_prediction_threshold))
    rpn_boxes = tf.gather_nd(rpn_boxes, condition)
    scores = tf.expand_dims(tf.gather_nd(scores, condition), axis=1)

    return features, rpn_boxes, scores
