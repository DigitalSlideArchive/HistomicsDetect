import tensorflow as tf


def parameterize(positive, boxes):
    """Transforms boxes that are matched to an anchor to a parameterized format
    used by the regression loss.
        
    Parameters
    ----------
    positive: tensor (float32)
        M x 5 tensor of anchors matched to ground truth boxes. Each row contains
        the x,y center location of an anchor, its width and height, and the index
        of the box that the anchor is matched to.
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
        
    Returns
    -------
    parameterized: tensor (float32)
        N x 4 tensor of anchor-matched boxes in a parameterized format.
    """

    #gather boxes matched to each anchor
    matched = tf.gather(boxes, tf.cast(positive[:,4], tf.int32), axis=0)

    tx = tf.divide((matched[:,0] + matched[:,2]/2) - positive[:,0], positive[:,2])
    ty = tf.divide((matched[:,1] + matched[:,3]/2) - positive[:,1], positive[:,3])
    tw = tf.math.log(tf.divide(matched[:,2], positive[:,2]))
    th = tf.math.log(tf.divide(matched[:,3], positive[:,3]))

    #stack
    parameterized = tf.stack([tx, ty, tw, th], axis=1)

    return parameterized


def unparameterize(parameterized, positive):
    """Transforms parameterized boxes to an un-parameterized [x,y,w,h] format.
    
    Given the matched and row-aligned tensors containing parameterized boxes and
    corresponding positive anchors, reverts the parameterization so that boxes
    can be used in IoU calculations or visualization.
        
    Parameters
    ----------
    parameterized: tensor (float32)
        N x 4 tensor of anchor-matched boxes in a parameterized format.
    positive: tensor (float32)
        N x 5 tensor of anchors matched to ground truth boxes. Each row contains
        the x,y center location of an anchor, its width and height, and the index
        of the box that the anchor is matched to.
        
    Returns
    -------
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    """
    
    #convert from parameterized representation to a box representation

    #transform predictions back to box format
    x = tf.multiply(parameterized[:,0], positive[:,2]) + positive[:,0]
    y = tf.multiply(parameterized[:,1], positive[:,3]) + positive[:,1]
    w = tf.multiply(tf.exp(parameterized[:,2]), positive[:,2])
    h = tf.multiply(tf.exp(parameterized[:,3]), positive[:,3])

    #translate box coordinates from center to edge
    x = x - w/2
    y = y - h/2

    #stack
    boxes = tf.stack([x, y, w, h], axis=1)

    return boxes
