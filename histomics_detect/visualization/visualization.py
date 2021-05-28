import matplotlib.pyplot as plt
import tensorflow as tf

    
def plot_inference(rgb, boxes, regressions, tp_list, fp_list, fn_list):
    """Generates color coded plot of inference results where true positives,
    false positives, and false negatives are color coded.
        
    Parameters
    ----------
    rgb: array
        Image for display with imshow.
    boxes: tensor
        M x 4 tensor where each row contains the x,y location of the upper left
        corner of each ground truth box and its width and height in that order.
    regressions: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of each regressed box and its width and height in that order.
    tp_list: tensor
        A 1-d tensor of indices of true positives in regressions as produced by
        the function greedy_iou.
    fp_list: tensor
        A 1-d tensor of indices of false positives in regressions as produced by
        the function greedy_iou.
    fn_list: tensor
        A 1-d tensor of indices of false negatives in regressions as produced by
        the function greedy_iou.
        
    See also
    --------
    greedy_iou, _plot_boxes
    """    
    
    plt.imshow(rgb)
    _plot_boxes(tf.gather(regressions, tp_list, axis=0), 'g')
    _plot_boxes(tf.gather(regressions, fp_list, axis=0), 'r')
    _plot_boxes(tf.gather(boxes, fn_list, axis=0), 'b')
    
    
def _plot_boxes(boxes, color):
    """Adds bounding boxes with chosen color to current plot.
        
    Parameters
    ----------
    boxes: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of a regressed box and its width and height in that order.
    color:
        Color argument for matplotlib pyplot.
    """    
    
    x, y, w, h = tf.split(boxes, 4, axis=1)
    for i, (xi, yi, wi, hi) in enumerate(zip(x, y, w, h)):
        plt.plot([xi, xi+wi, xi+wi, xi, xi],
                 [yi, yi, yi+hi, yi+hi, yi],
                 color=color)