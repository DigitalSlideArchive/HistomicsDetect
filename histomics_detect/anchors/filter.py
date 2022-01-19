from histomics_detect.metrics.iou import iou
import tensorflow as tf


@tf.function
def filter_anchors(boxes, anchors, alpha=0.7, beta=0.3, gamma=0.3):
    """Compares anchors to ground truth to determine positive and negative
    anchors.
    
    This function uses multiple criteria for intersection-over-union (IoU) 
    with ground truth to label anchors as either positive or negative. 
    A positive anchor is any anchor that has over 'alpha' IoU with any ground 
    truth object, or that is the max IoU anchor for a ground truth object 
    with IoU at least 'beta'. A negative anchor is any anchor where the max 
    IoU with ground truth objects is less than 'gamma'.
        
    Parameters
    ----------
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    anchors: tensor (float32)
        M x 4 tensor anchor positions organized in a dense grid over the image
        space. Each row contains the x,y upper left corner of the anchor in pixel 
        units relative in the image coordinate frame, and the anchor width and 
        height.
    alpha: float32
        Threshold for calling an anchor positive based on maximum IoU with any
        ground truth object.
    beta: float32
        Threshold for calling an anchor positive based on being the highest IoU
        anchor for a single ground truth object.
    gamma: float32
        Threshold for calling anchors negative.
        
    Returns
    -------
    positive: tensor (float32)
        M x 4 tensor of positive anchor positions satisfying IoU criteria
        with some ground truth object. Each row contains the x,y upper left corner 
        of the anchor in pixel units relative in the image coordinate frame, and
        the anchor width and height.
    negative: tensor (float32)
        N x 4 tensor of negative anchor positions not satisfying IoU criteria with
        any ground truth object. Each row contains the x,y upper left corner of the
        anchor in pixel units relative in the image coordinate frame, and the
        anchor width and height.
    """

    #calculate IOU between ground truth and anchors
    ious = iou(boxes, anchors)

    #anchors where IOU > alpha
    matches = tf.cast(tf.where(tf.greater(ious, alpha)), dtype=tf.int32)
    alpha_id = matches[:,1]
    alpha_box = matches[:,0]

    #anchors where max match for a box and IOU > beta
    beta_max = tf.reduce_max(ious, axis=1)
    beta_id = tf.argmax(ious, axis=1, output_type=tf.int32)
    beta_box = tf.range(0,tf.shape(boxes)[0], dtype=tf.int32)

    #filter out duplicates
    keep = tf.where(tf.logical_and(tf.greater(beta_max, beta),
                                   tf.less(beta_max, alpha)))[:,0]
    beta_id = tf.gather(beta_id, keep)
    beta_box = tf.gather(beta_box, keep)

    #combine positive anchors
    alpha_anchors = tf.concat([tf.gather(anchors, alpha_id),
                               tf.expand_dims(tf.cast(alpha_box, tf.float32), axis=1)],
                              axis=1)
    beta_anchors = tf.concat([tf.gather(anchors, beta_id),
                              tf.expand_dims(tf.cast(beta_box, tf.float32), axis=1)],
                             axis=1)
    positive = tf.concat([alpha_anchors, beta_anchors], axis=0)

    #anchors with IOU < gamma
    negative = tf.gather(anchors, tf.where(tf.less(tf.reduce_max(ious, axis=0), gamma)))
    negative = tf.squeeze(negative)

    return positive, negative
