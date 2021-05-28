import tensorflow as tf
from histomics_detect.metrics import greedy_iou, iou


def greedy_pr_auc(obj, reg, boxes, delta, min_iou, parallel=16):
    """Calculates precision-recall AUC using greedy IoU mapping.
    
    Uses intersection-over-union scores to compute a greedy mapping between
    ground truth and predicted objects. Greedy mapping can produce suboptimal
    results compared to the Kuhn–Munkres algorithm since matching is greedy. 
    Trapezoidal integration is used to calculate the area under precision
    recall curve calculated at multiple objectness thresholds in parallel.
    
    Parameters
    ----------
    obj: tensor (float32)
        N x 2 tensor containing softmax objectness scores for each proposed object. 
        Second column contains positive scores for objects.
    reg: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a regressed box and its width and height in that order.
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    delta: float32
        Increment used to define the sequence of thresholds applied to objectness
        scores.
    min_iou: tensor (float32)
        Minimum IoU threshold for defining a match between a regression
        prediction and a ground truth box.
        
    Returns
    -------
    area: float32
        Area under PR curve calculated using trapezoidal integration.
    """
    
    #generate objectness threshold sequence
    taus = tf.range(0, 1, delta)
    
    #calculate precision-recall values
    prs = tf.map_fn(lambda x: greedy_pr(obj, reg, boxes, x, min_iou), taus,
                    parallel_iterations=parallel)
    
    #add in origin
    prs = tf.concat([prs, tf.constant([[1.0, 0.0]], tf.float32)], axis=0)
    
    #calculate area under curve
    area = tf.reduce_sum(tf.multiply(prs[:-1,1] - prs[1:,1],
                                     (prs[:-1,0] + prs[1:,0])) / 2)
    
    return area


def greedy_pr(obj, reg, boxes, tau, min_iou):
    """Calculates precision-recall value using greedy IoU mapping.
    
    Uses intersection-over-union scores to compute a greedy mapping between
    ground truth and predicted objects. Greedy mapping can produce suboptimal
    results compared to the Kuhn–Munkres algorithm since matching is greedy.
    
    Parameters
    ----------
    obj: tensor (float32)
        Objectness scores for each proposed object ranging from [0, 1].
    reg: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a regressed box and its width and height in that order.
    boxes: tensor (float32)
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a ground truth box and its width and height in that order.
    tau: float32
        Threshold applied to objectness scores.
    min_iou: float32
        Minimum IoU threshold for defining a match between a regression
        prediction and a ground truth box.
        
    Returns
    -------
    area: tensor (float32)
        A 1D tensor containing a precision value and a recall value calculated at 
        objectness threshold 'tau'.
    """
    
    #call objects from rpn
    positive = tf.greater_equal(obj[:,1], tau)
    obj = tf.boolean_mask(obj, positive, axis=0)
    reg = tf.boolean_mask(reg, positive, axis=0)

    #greedy mapping regressions to ground-truth boxes
    ious, _ = iou(reg, boxes)
    precision, recall, tp, fp, fn, tp_list, fp_list, fn_list = \
        greedy_iou(ious, min_iou)

    return tf.stack([tf.cast(precision, tf.float32), tf.cast(recall, tf.float32)], 
                     axis=0)
