import tensorflow as tf


from histomics_detect.augmentation.augmentation import _box_crop


def cross_from_boxes(boxes: tf.Tensor, scale: float, width: int = 10, height: int = 10, image_width: int = 224,
                     image_height: int = 224, grow: bool = False) -> tf.Tensor:
    """
    generates cross shape for each box (similar to CornerNet)

    takes the center (c_x, c_y) of the boxes (x, y, w, h) and creates two boxes from that:
    - width box (vertical line of the cross):
        spans from (c_x-'width'/2, c_y-'factor'*h) to (c_x+'width'/2, c_y+'factor'*h)
    - height box (horizontal line of the cross):
        spans from (c_x-'factor'*w, c_y-'height'/2) to (c_x+'factor'*w, c_y+'height'/2)

    the cross is capped such that it is entirely within the image boundaries

    N: number of boxes

    Parameters
    ----------
    boxes: tensor (int32)
        input boxes to be turned into crosses
        shape (N, 4)
    scale: float
        see above

        scale * box_width, scale * box_height is the width and height of the cross
    width: int
        width of the vertical line
    height: int
        height of the horizontal line
    image_width: int
        width of the image
    image_height: int
        height of the image
    grow: bool
        make cross always span the whole image

    Returns
    -------
    cross_boxes: tensor (float32)
        vertical and horizontal boxes of each cross for each box
        shape: (N, 2, 4)
    """

    image_width = tf.cast(image_width, tf.float32)
    image_height = tf.cast(image_height, tf.float32)

    centers = boxes[:, :2] + boxes[:, 2:]/2

    if grow:
        horizontal_width = image_width * tf.ones(tf.shape(boxes[:, 2]))
        vertical_width = image_height * tf.ones(tf.shape(boxes[:, 2]))
    else:
        horizontal_width = scale * boxes[:, 2]
        vertical_width = scale * boxes[:, 3]

    h_x, h_y, h_w, h_h = centers[:, 0]-horizontal_width, centers[:, 1]-height/2, horizontal_width*2, \
                         tf.ones(tf.shape(centers)[0], tf.float32)*height
    h_x, h_w = _box_crop(h_x, h_w, image_width)
    h_y, h_h = _box_crop(h_y, h_h, image_height)

    horizontal_boxes = tf.stack([h_x, h_y, h_w, h_h], axis=1)

    v_x, v_y, v_w, v_h = centers[:, 0]-width/2, centers[:, 1]-vertical_width, tf.ones(tf.shape(centers)[0])*width, \
                         vertical_width*2
    v_x, v_w = _box_crop(v_x, v_w, image_width)
    v_y, v_h = _box_crop(v_y, v_h, image_height)

    vertical_boxes = tf.stack([v_x, v_y, v_w, v_h], axis=1)

    horizontal_boxes = tf.expand_dims(horizontal_boxes, axis=1)
    vertical_boxes = tf.expand_dims(vertical_boxes, axis=1)

    return tf.concat([horizontal_boxes, vertical_boxes], axis=1)
