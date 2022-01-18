import tensorflow as tf


@tf.function
def sample_anchors(positive, negative, max_anchors=256, np_ratio=2.0):
    """Sample to balance the proportion of positive and negative anchors used in 
    training.
    
    Samples up to max_n anchors with a maximum negative : positive ratio from the
    dense anchors in the image space.
        
    Parameters
    ----------
    positive: tensor (float32)
        M x 4 tensor of positive anchor positions satisfying IoU criteria
        with some ground truth object. Each row contains the x,y center location 
        of the anchor in pixel units relative in the image coordinate frame, and
        the anchor width and height.
    negative: tensor (float32)
        N x 4 tensor of negative anchor positions not satisfying IoU criteria with
        any ground truth object. Each row contains the x,y center location of the
        anchor in pixel units relative in the image coordinate frame, and the
        anchor width and height.
    max_anchors: int32
        Maximum number of total anchors to sample. Default value 256.
    np_ratio: float32
        Will sample at most negative : positive ratio anchors. Default value 2.0.
        
    Returns
    -------
    positive: tensor (float32)
        m x 4 tensor of positions of positive sampled anchors, where m <= M.
    negative: tensor (float32)
        n x 4 tensor of positions of negative sampled anchors, where n <= N and
        n + m <= max_anchors and n : m < np_ratio.
    """

    #calculate negative anchor limit if positive anchors abundant
    neg_lim = max_anchors - tf.cast(tf.cast(max_anchors / (1 + np_ratio), tf.float32), tf.int32)
    nneg = tf.minimum(tf.cast(tf.cast(tf.shape(positive)[0], tf.float32) * np_ratio, tf.int32),
                      neg_lim)
    nneg = tf.maximum(nneg, 1)
    
    #sample positive anchors
    npos = tf.minimum(tf.shape(positive)[0], max_anchors-nneg)
    indices = _sample_no_replacement(tf.shape(positive)[0]-1, npos)
    positive = tf.gather(positive, indices)    
    
    #sample negative anchors
    indices = _sample_no_replacement(tf.shape(negative)[0]-1, nneg)
    negative = tf.gather(negative, indices)
    
    return positive, negative
  

@tf.function
def _sample_no_replacement(maxval, size):
    """Generates indices to sample from a tensor without replacement.
    
    This function generates a maxval-length tensor of random values
    and uses the sorting indices of this array to generate the random
    sample.
        
    Parameters
    ----------
    maxval: int32
        Value of the largest possible index to sample. 
    size: int32
        The number of elements to sample.
        
    Returns
    -------
    positive: tensor (float32)
        m x 4 tensor of positions of positive sampled anchors, where m <= M.
    negative: tensor (float32)
        n x 4 tensor of positions of negative sampled anchors, where n <= N and
        n + m <= max_n and m/n >= min_ratio.
    """
    
    #sample 'size' int32 values up to 'maxval' without replacement
    z = tf.random.uniform((maxval+1,), 0, 1)
    _, indices = tf.nn.top_k(z, size)
    
    return indices
