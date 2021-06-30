import tensorflow as tf


def parameterize(positive, boxes):
    """Transforms boxes that are matched to an anchor to a parameterized format
    used by the regression loss.
        
    Parameters
    ----------
    positive: tensor (float32)
        M x 5 tensor of anchors matched to ground truth boxes. Each row contains
        the x,y upper left corner, width, height, and matched box index for one
        anchor.
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

    #calculate parameterization using matched anchors
    tx = tf.divide(matched[:,0] - positive[:,0], positive[:,2])
    ty = tf.divide(matched[:,1] - positive[:,1], positive[:,3])
    tw = tf.math.log(tf.divide(matched[:,2], positive[:,2]))
    th = tf.math.log(tf.divide(matched[:,3], positive[:,3]))

    #stack results
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
  

def tf_box_transform(boxes):
    """Transform bounding boxes to TF format.
    
    Transforms the [x,y,w,h] format for bounding boxes used throughout this package
    to the [x,y,x+w-1,x+h-1] expected by TensorFlow.
    
    Parameters
    ----------
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
        
    Returns
    -------
    transformed: tensor (float32)
        N x 4 tensor where each row contains the x,y locations of the upper left
        and lower right corners in that order.
    """

    #unstack box columns
    x, y, w, h = _unstack_box_array(boxes)

    #join transformed columns
    transformed = tf.stack([x, y, tf.add(x, w-1), tf.add(y, h-1)], axis=1)

    return transformed
    

def clip_boxes(boxes, width, height):
    """Clips boxes that extend beyond a region of interest boundary.
    
    Transforms each box so that the minimum x/y coordinates are zero and the maximum
    coordinates are width/height.
    
    Parameters
    ----------
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    width: float32
        Width of region of interest to clip boxes to.
    height: float32
        Height of region of interest to clip boxes to.
        
    Returns
    -------
    clipped: tensor (float32)
        Same as input but with boxes clipped to region of interest.
    """    
    
    
    #clips bounding boxes in [x,y,w,h] format to image dimensions

    # unstack box columns
    x, y, w, h = _unstack_box_array(boxes)

    #clip left corner
    x = tf.maximum(x, 0.0)
    y = tf.maximum(y, 0.0)

    #clip edge lengths
    w = tf.subtract(tf.minimum(tf.cast(width, tf.float32), tf.add(x,w)), x)
    h = tf.subtract(tf.minimum(tf.cast(height, tf.float32), tf.add(y,h)), y)

    #join clipped columns
    clipped = tf.stack([x,y,w,h], axis=1)

    return clipped


def _unstack_box_array(boxes):
    """Unstacks the x,y,w,h bounding box parameters from the stacked input.
    
    Parameters
    ----------
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
        
    Returns
    -------
    x: tensor (float32)
        N-length tensor of x coordinates of upper left corner of bounding boxes.
    y: tensor (float32)
        N-length tensor of y coordinates of upper left corner of bounding boxes.    
    w: tensor (float32)
        N-length tensor of bounding boxe widths.
    h: tensor (float32)
        N-length tensor of bounding boxe heights.
    """    
    
    
    #unstacks columns from bounding box array. use gather instead of unstack, 
    #because input could have 5 columns (arrays containing 'positive' anchors that
    #have been matched to bounding boxes can contain a fifth column that is the
    #index of the matched bounding box. 
  
    #gather first four columns
    x = tf.gather(boxes, 0, axis=1)
    y = tf.gather(boxes, 1, axis=1)
    w = tf.gather(boxes, 2, axis=1)
    h = tf.gather(boxes, 3, axis=1)

    return x, y, w, h


def filter_edge_boxes(boxes, width: float, height: float, margin: float = 5.0):
    """Filters out boxes that cross the margin of the image boundary

        a box is kept if all the points are within the image boundary inset by the margin parameter.

        Example:
            given a box with top left and bottom right corners [[2, 2], [9, 9]

            for a image with size 10, 10 this box is not filtered if margin <= 1

        Parameters
        ----------
        margin: float
            offset from the border of the image border where boxes that overlap are removed
        height: float
            height of the image
        width: float
            width of the image
        boxes: tensor (float32)
            N x 4 tensor where each row contains the x,y location of the upper left
            corner of a ground truth box and its width and height in that order.

        Returns
        -------
        filtered_boxes: tensor (float32)
            N x 4 tensor where each row contains the x,y location of the upper left
            corner of a ground truth box and its width and height in that order.
            All boxes that have at least one point in the margin have been removed.
        """
    margin = tf.cast(margin, tf.float32)
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # unstack box columns
    x, y, w, h = _unstack_box_array(boxes)
    x2 = x + w
    y2 = y + h

    # condition that box will be kept
    min_cond = tf.logical_and(x >= margin, y >= margin)
    max_cond = tf.logical_and(x2 <= (width-margin), y2 <= (height-margin))
    condition = tf.logical_and(min_cond, max_cond)

    # stack columns and collect boxes that fulfill the condition
    filtered_boxes = tf.gather_nd(tf.stack([x, y, w, h], axis=1), tf.where(condition))
    return filtered_boxes
