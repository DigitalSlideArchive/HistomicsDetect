import tensorflow as tf


def sample_anchors(positive, negative, max_n=256, min_ratio=0.5):
    """Sample to balance the proportion of positive and negative anchors used in 
    training.
    
    Samples up to max_n anchors with a minimum positive : negative ratio from the
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
    max_n: int32
        Maximum number of total anchors to sample. 
    min_ratio: float32
        Will sample at most positive anchors / min_ratio total anchors.
        
    Returns
    -------
    positive: tensor (float32)
        m x 4 tensor of positions of positive sampled anchors, where m <= M.
    negative: tensor (float32)
        n x 4 tensor of positions of negative sampled anchors, where n <= N and
        n + m <= max_n and m/n >= min_ratio.
    """

    #sample positive anchors
    npos = tf.minimum(tf.shape(positive)[0], tf.cast(max_n/2, tf.int32))
    indices = sample_no_replacement(tf.shape(positive)[0]-1, npos)
    positive = tf.gather(positive, indices)

    #sample negative anchors
    limit = tf.cast(tf.round(tf.cast(npos, tf.float32)/min_ratio), tf.int32)
    nneg = tf.minimum(tf.shape(negative)[0],
                      tf.minimum(limit, tf.cast(max_n/2, tf.int32)))
    nneg = tf.maximum(nneg, 1)
    indices = _sample_no_replacement(tf.shape(negative)[0]-1, nneg)
    negative = tf.gather(negative, indices)

    return positive, negative
  
  
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
