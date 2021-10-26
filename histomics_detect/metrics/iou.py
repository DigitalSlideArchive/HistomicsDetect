import tensorflow as tf

def iou(source, target):
    """Calculates intersection over union (IoU) and intersection areas for two sets
    of objects with box representations.
    
    This uses simple arithmetic and outer products to calculate the IoU and intersections
    between all pairs without looping.
        
    Parameters
    ----------
    source: tensor (float32)
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the
        predictions.
    target: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a box and its width and height in that order. Typically the
        ground truth.
        
    Returns
    -------
    iou: tensor (float32)
        M x N tensor containing IoU values between source and target boxes.
    intersection: tensor (float32)
        M x N tensor containing area overlaps between source and target boxes
        in pixels.
    """
    
    #split into corners and sizes
    xs, ys, ws, hs = tf.split(source, 4, axis=1)
    xt, yt, wt, ht = tf.split(target, 4, axis=1)

    #overlap in dimensions
    left = tf.maximum(xs, tf.transpose(xt))
    top = tf.maximum(ys, tf.transpose(yt))
    right = tf.minimum(xs+ws, tf.transpose(xt+wt))
    bottom = tf.minimum(ys+hs, tf.transpose(yt+ht))

    horizontal = tf.minimum(xs+ws, tf.transpose(xt+wt)) - \
        tf.maximum(xs, tf.transpose(xt))
    vertical = tf.minimum(ys+hs, tf.transpose(yt+ht)) - \
        tf.maximum(ys, tf.transpose(yt))

    #calculate intersection
    intersection = tf.maximum(0.0, horizontal) * tf.maximum(0.0, vertical)

    #calculate iou
    iou = intersection / (ws*hs + tf.transpose(wt*ht) - intersection)

    return iou


def _greedy_iou_mapping_iter(i, ious, source_mask, target_mask, matches):
    """Performs one iteration of greedy IoU mapping.
    
    This is the loop body of the greedy IoU mapping algorithm. This identifies the 
    best match having the highest IoU and removes the corresponding prediction and 
    ground truth element from future consideration in matching.
    
    Parameters
    ----------
    i: int32
        Iteration number in mapping. Used for writing to output TensorArray.
    ious: tensor (float32)
        M x N tensor of IoU values used to generate mapping. Regression predictions
        are in rows and ground truth elements are in columns. This array is masked
        to remove previous matches when identifying the highest IoU match.
    source_mask: tensor (bool)
        1D M-length tensor where unmatched predictions are represented by 'True'.
    target_mask: tensor (bool)
        1D M-length tensor where unmatched ground truth elements are represented 
        by 'True'.    
    matches: tensor (float32)
        2D tensor where each row represents a match, containing the indices of
        the matched prediction and ground truth element in that order.
        
    Returns
    -------
    i: int32
        Loop iteration counter.
    ious: tensor (float32)
        Same as input but updated with current iteration match.
    source_mask: tensor (bool)
        Same as input but updated with current iteration match.
    target_mask: tensor (bool)
        Same as input but updated with current iteration match.
    matches: tensor (float32)
        Same as input but updated with current iteration match.
    """

    #mask targets and get best match for each source
    maxima = tf.reduce_max(tf.boolean_mask(ious, target_mask, axis=1), axis=1)
    target_indices = tf.argmax(tf.boolean_mask(ious, target_mask, axis=1), axis=1)

    #mask sources that were already matched
    maxima = tf.boolean_mask(maxima, source_mask)
    target_indices = tf.boolean_mask(target_indices, source_mask)

    #get source and target indices
    max = tf.reduce_max(maxima)
    source_index = tf.argmax(maxima)
    target_index = tf.gather(target_indices, source_index)

    #correct for masked sources and targets
    source_index = tf.gather(tf.where(source_mask), source_index)
    target_index = tf.gather(tf.where(target_mask), target_index)

    #update masks
    source_mask = tf.tensor_scatter_nd_update(source_mask, [source_index],
                                              [tf.constant(False)])
    target_mask = tf.tensor_scatter_nd_update(target_mask, [target_index],
                                              [tf.constant(False)])
  
    #write (source, target) to TensorArray
    matches = matches.write(i, tf.concat([tf.cast(source_index, tf.float32),
                                          tf.cast(target_index, tf.float32),
                                          [max]], axis=0))
  
    #update index
    i = i + 1

    return i, ious, source_mask, target_mask, matches


def greedy_iou_mapping(ious, min_iou):
    """Calculates greedy IoU mapping between predictions and ground truth.
    
    Uses intersection-over-union scores to compute a greedy mapping between
    ground truth and predicted objects. Greedy mapping can produce suboptimal
    results compared to the Kuhn–Munkres algorithm since matching is greedy.
    
    Parameters
    ----------
    ious: tensor (float32)
        M x N tensor of IoU values used to generate mapping. Regression predictions
        are in rows and ground truth elements are in columns. This array is masked
        to remove previous matches when identifying the highest IoU match.
    min_iou: float32
        Minimum IoU threshold for defining a match between a regression
        prediction and a ground truth box.
        
    Returns
    -------
    precision: float32
        Precision of IoU mapping.
    recall: float32
        Recall of IoU mapping.
    tp: int32
        True positive count of IoU mapping.
    fp: int32
        False positive count of IoU mapping.
    fn: int32
        False negative count of IoU mapping.
    tp_list: int32
        Two-dimensional tensor containing indices of true positive predictions
        in first column, and corresponding matching ground truth indices in second
        column.
    fp_list: int32
        One-dimensional tensor containing indices of false positive predictions.
    fn_list: int32
        One-dimensional tensor containing indices of false negative ground truth.
    """

    #initialize masks
    source_mask = tf.ones(tf.shape(ious)[0], tf.bool)
    target_mask = tf.ones(tf.shape(ious)[1], tf.bool)

    #define loop counter, condition, store for output
    i = tf.constant(0)
    matches = tf.TensorArray(tf.float32, size=tf.shape(ious)[0], dynamic_size=False)
    condition = lambda i, a, b, c, d: tf.less(i, tf.minimum(tf.shape(ious)[0],
                                                            tf.shape(ious)[1]))

    #loop to perform greedy mapping
    _, _, _, _, matches = tf.while_loop(condition, _greedy_iou_mapping_iter,
                                        [i, ious, source_mask, target_mask, matches],
                                        parallel_iterations=10)

    #stack outputs
    matches = matches.stack()

    #discard matches that do not meet min_iou
    matches = tf.boolean_mask(matches,
                              tf.greater_equal(matches[:,2], min_iou), axis=0)

    #calculate TP, FP, FN, precision, recall
    tp = tf.shape(matches)[0]
    fp = tf.shape(ious)[0] - tf.shape(matches)[0]
    fn = tf.shape(ious)[1] - tf.shape(matches)[0]

    #generate lists of indexes for TP, FP, FN
    tp_list = tf.cast(matches[:,0:2], tf.int32)
    fp_list = tf.sets.difference([tf.range(tf.shape(ious)[0], dtype=tf.int32)],
                                 [tf.cast(matches[:,0], dtype=tf.int32)]).values
    fn_list = tf.sets.difference([tf.range(tf.shape(ious)[1], dtype=tf.int32)],
                                 [tf.cast(matches[:,1], dtype=tf.int32)]).values

    return tp, fp, fn, tp_list, fp_list, fn_list
