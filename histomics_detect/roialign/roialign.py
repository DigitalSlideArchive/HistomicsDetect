import tensorflow as tf


@tf.function
def roialign(features, boxes, field, pool=2, tiles=3):
    """Performs roialign on a collection of regressed bounding boxes.
    
    Given an array of N boxes, this first calculates the pixel coordinates
    where features should be interpolated within each regressed bounding box,
    then performs this interpolation and pools the results to generate a
    single feature vector for each regressed box. Each regressed box is
    subdivided into a 'tiles' x 'tiles' array, with each tile containing
    'pool' x 'pool' interpolation points. Pooling of interpolated features
    is performed at the tile level, and the resulting pooled features are
    concatenated to form a single feature vector for objectness / class
    classification.
        
    Parameters
    ----------
    features: tensor
        The three dimensional feature map tensor produced by the backbone network.
    boxes: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a regressed box and its width and height in that order.
    field: float32
        Field size of the backbone network in pixels.
    pool: int32
        pool^2 is the number of locations to interpolate features at within 
        each tile.
    tiles: int32
        tile^2 is the number of tiles that each regressed bounding box is divided
        into.
        
    Returns
    -------
    interpolated: tensor
        An N x pool * tiles x pool * tiles x features array containing the 
        interpolated features in a 2D layout grouped by bounding box. These can
        be max-pooled (pool, pool) along the spatial dimensions (dimensions 2 
        and 3) to produce an N x tile x tile x features array.
    """
    
    #generate pixel coordinates of box sampling locations
    offsets = _roialign_coords(boxes, pool*tiles)

    #convert from pixel coordinates to feature map / receptive field coordinates
    #the top-left feature is centered at (field/2, field/2) pixels, so an offset needs
    #to be applied when comparing features which are located in the center of each
    #receptive fields with boxes that are defined by their upper left corner in the
    #image.
    offsets = (offsets - field / 2) / tf.cast(field, tf.float32)
    
    #bilinear interpolation
    interpolated = _bilinear(features, offsets[:,0], offsets[:,1])

    #reshape so that first dimension is the box id, second/third are the spatial dimensions
    #of the subdivided box, and the last dimension is the feature
    interpolated = tf.reshape(interpolated, [tf.shape(boxes)[0],
                                             tf.cast(pool*tiles, tf.int32),
                                             tf.cast(pool*tiles, tf.int32),
                                             tf.shape(features)[-1]])

    return interpolated


@tf.function
def _roialign_coords(boxes, n_points):
    """Generates arguments for bilinear interpolation of feature maps
    for a single bounding box.
    
    Used for performing roialign on regressed bounding boxes from the
    region proposal network output. This generates the coordinates
    needed to perform bilinear interpolation across the region proposal
    network feature maps at the internal grid points in the regressed
    box.
        
    Parameters
    ----------
    boxes: tensor
        A 2d tensor where each row represents a box, containing 4 elements 
        describing the x,y location of the upper left corner of a regressed 
        box and its width and height in that order.
    n_points: float
        The number of points to interpolate at across the width / height of 
        the regressed box.
        
    Returns
    -------
    sample: tensor
        The interpolation coordinate arguments in pixel units for use with 
        tfa.image.interpolate_bilinear. These are an n_points^2 x 2 array
        of x, y coordinates for interpolation that are specific to the position
        and shape of the input 'box'.
    """

    #generate a sequence of the indices of sample locations
    indices = tf.range(0.0, n_points)

    #calculate horizontal, vertical spacings between samples (in pixels)
    delta_width = boxes[:,2] / tf.cast(tf.size(indices), tf.float32)
    delta_height = boxes[:,3] / tf.cast(tf.size(indices), tf.float32)

    #calculate vertical and horizontal positions of sample locations
    x_offset = tf.expand_dims(delta_width / 2.0, axis=1) + \
        tf.tensordot(delta_width, indices, axes=0)
    y_offset = tf.expand_dims(delta_height / 2.0, axis=1) + \
        tf.tensordot(delta_height, indices, axes=0)

    #generate pairs of all possible x, y coordinates for each box
    x_offset = tf.tile(x_offset, [1, n_points])
    y_offset = tf.repeat(y_offset, n_points*tf.ones(n_points, tf.int32), axis=1)

    #add box corner x,y coordinates to offsets
    x_offset = x_offset + tf.expand_dims(boxes[:,0], axis=1)
    y_offset = y_offset + tf.expand_dims(boxes[:,1], axis=1)

    #reshape offsets to a 2-D array with x,y pairs in rows
    sample = tf.reshape(tf.stack([x_offset, y_offset], axis=2),
                        [n_points * n_points * tf.shape(boxes)[0], 2])

    return sample


@tf.function
def _bilinear(features, x, y):
    """
    Performs bilinear interpolation of a 3d tensor along first two
    dimensions. Used for calculations of roialign. Replacement for 
    tensorflow addons bilinear interpolation.
    
    Parameters
    ----------
    features: tensor (float32)
        A three-dimensional tensor (M, N, K) where the third dimension are
        features / channels.    
    x: tensor (float32)
        Horizontal coordinates where linear interpolation is desired.
    y: tensor (float32)
        Vertical coordinates where linear interpolation is desired.    
        
    Outputs
    -------
    interpolated: tensor (float32)
        An (d, 1, K) tensor where d is the number of points to interpolate.
    """    
    
    #calculate top/bottom, left/right reference positions
    lower_y, upper_y = _linear_indices(y, tf.shape(features)[-3])
    lower_x, upper_x = _linear_indices(x, tf.shape(features)[-2])

    #feature values at horizontal reference locations
    fly_lx, fly_ux = _linear_f(features, lower_y, lower_x, upper_x, axis=1)
    fuy_lx, fuy_ux = _linear_f(features, upper_y, lower_x, upper_x, axis=1)

    #horizontal linear interpolation
    fly = _linear_interp(fly_lx, fly_ux, x, lower_x)
    fuy = _linear_interp(fuy_lx, fuy_ux, x, lower_x)

    #vertical linear interpolation
    interpolated = _linear_interp(fly, fuy, y, lower_y)

    #reshape output to return 3d tensor
    interpolated = tf.reshape(interpolated,
                               [tf.size(x), 1, tf.shape(features)[-1]])
    
    return interpolated


@tf.function
def _linear_indices(x, size):
    """
    Calculates locations of upper and lower reference points used for
    interpolation.
    
    Parameters
    ----------
    x: tensor (float32)
        Coordinates where linear interpolation is desired (either
        x or y values). 
    size: int
        Size of the tensor to interpolate along that dimension.
        
    Outputs
    -------
    lower: tensor (int32)
        Index for left/top location to use for interpolation.
    upper: tensor (int32)
        Index for right/bottom location to use for interpolation.
    """
    
    #calculate lower and upper indices for interpolation
    lower = tf.cast(tf.minimum(tf.maximum(0.0, tf.math.floor(x)), 
                               tf.cast(size-1, tf.float32)), tf.int32)
    upper = tf.cast(tf.minimum(lower+1, size-1), tf.int32)
    
    return lower, upper


@tf.function
def _linear_f(features, fixed, lower, upper, axis):
    """
    Extracts feature values at left/right or top/bottom reference
    points.
    
    Parameters
    ----------
    features: tensor (float32)
        An M x N x K tensor containing features along dimension 2.
    fixed: tensor (float32)
        A d-length tensor containing the fixed coordinates (x or y) to be used 
        in the interpolation. Bilinear interpolation is performed along each axis, 
        holding the other fixed.
    lower: tensor (float32)
        A d-length tensor containing the coordinates of the top/left reference 
        points.
    upper: tensor (float32)
        A d-length tensor containing the coordinates of the bottom/right 
        reference points.
    axis: int
        The axis of features that the interpolation will be performed against.
        Either 0 or 1.
    
    Outputs
    -------
    f_lower: tensor (float32)
        A d x K tensor containing the feature values of top/left reference points
        in rows.
    f_upper: tensor (float32)
        A d x K tensor containing the feature values of bottom/right reference 
        points in rows.        
    """
    
    #build tensors for calling gather_nd - assume axis=1 ('fixed' is y)
    lower_indices = tf.stack([tf.cast(fixed, tf.int32), lower], axis=1)
    upper_indices =  tf.stack([tf.cast(fixed, tf.int32), upper], axis=1)

    #if vertical interpolation (axis=0), switch indices for gather_nd
    lower_indices = tf.gather(lower_indices, [axis, 1-axis], axis=1)
    upper_indices = tf.gather(upper_indices, [axis, 1-axis], axis=1)

    #transform to linear indices to use gather instead of gather_nd and avoid OOM
    lower_linear = lower_indices[:,1] * tf.shape(features)[-2] + lower_indices[:,0]
    upper_linear = upper_indices[:,1] * tf.shape(features)[-2] + upper_indices[:,0]

    #create a reshaped view of 'features' for linear indexing
    reshaped = tf.reshape(features, [-1, tf.shape(features)[-1]])

    #calculate slopes
    f_lower = tf.gather(reshaped, lower_linear)
    f_upper = tf.gather(reshaped, upper_linear)

    return f_lower, f_upper


@tf.function
def _linear_interp(f_lower, f_upper, x, lower):
    """
    One-dimensional interpolation given the features values at reference points.
    
    Parameters
    ----------
    f_lower: tensor (float32)
        A d x K array containing feature values at d left/top reference points.
    f_upper: tensor (float32)
        A d x K array containing feature values at d right/bottom reference points.
    x: tensor (float32)
        A d-length array containing the horizontal/vertical coordinates to interpolate at.
    lower: tensor (float32)
        A d-length array containing the coordinates of the left/top reference points.
    
    Outputs
    -------
    interpolated: tensor (float32)
        A d x K array containing interpolated feature values at x.
    """
    
    #calculate offsets between x and lower
    dx = tf.maximum(0.0, x-tf.cast(lower, tf.float32))
    
    #calculate output
    interpolated = f_lower + tf.multiply(tf.expand_dims(dx, axis=-1), 
                                         f_upper-f_lower)  
    
    return interpolated
