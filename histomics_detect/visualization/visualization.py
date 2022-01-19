import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['figure.dpi'] = 300 #set figure dots-per-inch


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

        
def plot_inference(rgb, regressions, color='g'):
    """Generates plot of inference results.
        
    Parameters
    ----------
    rgb: array
        Image for display with imshow.
    regressions: tensor
        N x 4 tensor where each row contains the x,y location of the upper left
        corner of each regressed box and its width and height in that order.
    color:
        Color argument for matplotlib pyplot.    
        
    See also
    --------
    greedy_iou, _plot_boxes
    """    
    
    plt.imshow(rgb)
    _plot_boxes(regressions, color)
        
    
def plot_evaluation(rgb, boxes, regressions, tp_list, fp_list, fn_list):
    """Generates color coded plot of evaluation results where true positives,
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
        A 2-d tensor containing indices of true positive predictions in regressions
        in the first column and the corresponding indices of matched ground truth 
        in boxes.
    fp_list: tensor
        A 1-d tensor of indices of false positive predictions in regressions.
    fn_list: tensor
        A 1-d tensor of indices of false negatives in regressions as produced by
        the function greedy_iou.
        
    See also
    --------
    greedy_iou, _plot_boxes
    """    
    
    plt.imshow(rgb)
    _plot_boxes(tf.gather(regressions, tp_list[:,0], axis=0), 'g')
    _plot_boxes(tf.gather(regressions, fp_list, axis=0), 'r')
    _plot_boxes(tf.gather(boxes, fn_list, axis=0), 'b')
