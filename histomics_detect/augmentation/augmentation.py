import tensorflow as tf


@tf.function
def _box_crop(corner, length, window):
    """Applies a crop to a sequence of boxes to remove portions falling outside
    a defined region.
    
    This function is applied to each dimension independently. Given the region
    with edge length 'window', this crops the boxes to clip portions that lie 
    outside the top/left or bottom right of the region. Boxes with zero overlap
    will have their height/width set to zero.
        
    Parameters
    ----------
    corner: tensor (float32)
        Coordinates of left or top of boxes.
    length: tensor (float32)
        Width or height of boxes.
    window: float32
        Width or height of region, assume upper/left corner is at 0.
        
    Returns
    -------
    corner_crop: tensor (float32)
        1D tensor containing coordinate of upper/left of cropped boxes.
    intersection: tensor (float32)
        1D tensor containing height/width of cropped boxes.
    """

    corner_crop = tf.minimum(tf.maximum(corner, 0), window)
    length_crop = tf.minimum(tf.maximum(corner+length, 0), window) - corner_crop

    return corner_crop, length_crop


@tf.function
def flip(rgb, boxes):
    """Randomly flips an image and ground truth boxes along horizontal and/or 
    verical axis.
    
    This function is used for data augmentation that consistently flips the 
    image array as well as transposing the box coordinates and width/height at
    random.
        
    Parameters
    ----------
    corner: tensor (float32)
        Coordinates of left or top of boxes.
    length: tensor (float32)
        Width or height of boxes.
    window: float32
        Width or height of region, assume upper/left corner is at 0.
        
    Returns
    -------
    corner_crop: tensor (float32)
        1D tensor containing coordinate of upper/left of cropped boxes.
    intersection: tensor (float32)
        1D tensor containing height/width of cropped boxes.
    """

    #condense ragged tensor
    [x, y, w, h] = tf.unstack(boxes.to_tensor(), num=4, axis=1)

    if tf.random.uniform([1])[0] > 0.5:
    
        #flip image
        rgb = tf.image.flip_up_down(rgb)

        #flip coordinates
        y = tf.cast(tf.shape(rgb)[0], tf.float32) - y - h

    if tf.random.uniform([1])[0] > 0.5:
    
        #flip image
        rgb = tf.image.flip_left_right(rgb)

        #flip coordinates
        x = tf.cast(tf.shape(rgb)[1], tf.float32) - x - w

    #re-stack
    boxes = tf.RaggedTensor.from_tensor(tf.stack([x, y, w, h], axis=1))

    return rgb, boxes


@tf.function
def crop(rgb, boxes, width, height, min_fraction=0.5):
    """Randomly crops a portion of the input image and ground truth boxes.
    
    This function is used for data augmentation and randomly crops a
    height x width portion of the input and corresponding ground truth
    boxes. Removes cropped boxes that do not have at least 'min_fraction'
    of their area contained within the cropped region to clean up
    boundaries. To avoid producing empty output that does not contain any
    ground truth boxes, we pick a single box at random and ensure that this 
    is wholly contained within the crop.
        
    Parameters
    ----------
    rgb: tensor
        The 2D or 3D input image tensor.
    boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    width: int32
        Width of the cropped output image.
    height: int32
        Height of the cropped output image.
    min_fraction: float32
        Threshold for percent of original area used to retain cropped boxes.
        
    Returns
    -------
    crop: tensor
        Cropped height x width image.
    boxes: tensor (float32)
        N x 4 tensor containing cropped boxes where each row contains the x,y 
        location of the upper left corner of a ground truth box and its width 
        and height in that order. Each box is at least min_fraction percent
        of its original uncropped size.
    """

    #modify shapes
    width = tf.minimum(width, tf.shape(rgb)[1])
    height = tf.minimum(height, tf.shape(rgb)[0])

    #condense ragged tensor
    [x, y, w, h] = tf.unstack(boxes.to_tensor(), num=4, axis=1)

    #identify candidate cells to include in cropped roi
    c_width = tf.less(w, tf.cast(width, tf.float32))
    c_height = tf.less(h, tf.cast(height, tf.float32))
    c_left = tf.greater_equal(x, 0.0)
    c_top = tf.greater_equal(y, 0.0)
    c_right = tf.less(x+w, tf.cast(tf.shape(rgb)[1], tf.float32))
    c_bottom = tf.less(y+h, tf.cast(tf.shape(rgb)[0], tf.float32))
    conditions = tf.stack([c_width, c_height, c_left,
                           c_top, c_right, c_bottom], axis=1)
    include = tf.cast(tf.where(tf.reduce_all(conditions, axis=1)), tf.int32)
    x = tf.gather(x, include)
    y = tf.gather(y, include)
    w = tf.gather(w, include)
    h = tf.gather(h, include)

    #select box/cell at random
    selected = tf.random.uniform([1], 0, tf.size(x)-1, dtype=tf.int32)[0]
  
    #Determine range of crop limits for upper left corner of random 
    #(height, width) fields that include box. Calculate this in floats to deal
    #with edge cases like floor(x[selected])=0. 
    lower_x = tf.maximum(x[selected] + w[selected] - tf.cast(width, tf.float32), 0)
    upper_x = tf.minimum(x[selected], tf.cast(tf.shape(rgb)[1] - width,
                                              tf.float32))
    lower_y = tf.maximum(y[selected] + h[selected] - tf.cast(height, tf.float32), 0)
    upper_y = tf.minimum(y[selected], tf.cast(tf.shape(rgb)[0] - height,
                                              tf.float32))

    #sample upper left corner of roi
    x_crop = tf.random.uniform([1], lower_x[0], upper_x[0], tf.float32)[0]
    y_crop = tf.random.uniform([1], lower_y[0], upper_y[0], tf.float32)[0]
    x_crop = tf.cast(tf.round(x_crop), tf.int32)
    y_crop = tf.cast(tf.round(y_crop), tf.int32) 

    #crop 
    crop = tf.image.crop_to_bounding_box(rgb, y_crop, x_crop, 
                                         tf.cast(height, tf.int32),
                                         tf.cast(width, tf.int32))

    #translate boxes and apply crop
    x = x - tf.cast(x_crop, tf.float32)
    y = y - tf.cast(y_crop, tf.float32)
    x, wc = _box_crop(x, w, tf.cast(width, tf.float32))
    y, hc = _box_crop(y, h, tf.cast(height, tf.float32))

    #calculate cropped box area, proportion of box in cropped region
    proportion = tf.divide(tf.multiply(wc, hc), tf.multiply(w, h))

    #assign outputs
    mask = tf.greater_equal(proportion, min_fraction)
    x = tf.boolean_mask(x, mask)
    y = tf.boolean_mask(y, mask)
    w = tf.boolean_mask(wc, mask)
    h = tf.boolean_mask(hc, mask)

    boxes = tf.RaggedTensor.from_tensor(tf.stack((x,y,w,h), axis=1))

    return crop, boxes
  

@tf.function
def jitter(boxes, percent=0.05):
    """Randomly displaces bounding boxes using uniform noise proportional
    to a percentage of box dimensions.
    
    This function is used for data augmentation and adds noise proportional
    to bounding box dimensions to the box coordinates.
        
    Parameters
    ----------
    boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    percent: float
        Range of noise applied to bounding boxes. Default value of 0.05
        corresponds to up to a +/- 5 percent displacement.
        
    Returns
    -------
    boxes: tensor (float32)
        M x 4 tensor where the x,y locations of each box have been displaced by 
        up to 'percent' of the box width and height respectively.
    """
    
    #condense ragged tensor
    [x, y, w, h] = tf.unstack(boxes.to_tensor(), num=4, axis=1)
    
    #generate noise vectors
    x_noise = tf.random.uniform(tf.shape(x), -percent, percent, tf.float32)
    y_noise = tf.random.uniform(tf.shape(y), -percent, percent, tf.float32)
    
    #modify locations
    x = x + tf.multiply(x_noise, w)
    y = y + tf.multiply(y_noise, h)
    
    #stack result and return
    boxes = tf.RaggedTensor.from_tensor(tf.stack((x,y,w,h), axis=1))
    
    return boxes


@tf.function
def shrink(boxes, percent=0.05):
    """Randomly resizes bounding boxes proportional to box dimensions.
    
    This function is used for data augmentation.
        
    Parameters
    ----------
    boxes: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    percent: float
        Range of percentages used to resize bounding boxes. Default value of 0.05
        corresponds to up to a +/- 5 percent change in size.
        
    Returns
    -------
    boxes: tensor (float32)
        M x 4 tensor where the width and height of each box have been changed by 
        up to 'percent' of the box width and height respectively.
    """
    
    #condense ragged tensor
    [x, y, w, h] = tf.unstack(boxes.to_tensor(), num=4, axis=1)
    
    #generate noise vectors
    w_noise = tf.random.uniform(tf.shape(x), 1.0-percent, 1.0+percent, tf.float32)
    h_noise = tf.random.uniform(tf.shape(y), 1.0-percent, 1.0+percent, tf.float32)
    
    #modify dimensions
    w = tf.multiply(w_noise, w)
    h = tf.multiply(h_noise, h)
    
    #stack result and return
    boxes = tf.RaggedTensor.from_tensor(tf.stack((x,y,w,h), axis=1))
    
    return boxes
