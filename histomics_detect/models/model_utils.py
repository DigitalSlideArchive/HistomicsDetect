import tensorflow as tf


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
