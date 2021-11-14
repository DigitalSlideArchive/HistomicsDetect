from .iou import greedy_iou_mapping, iou
import tensorflow as tf


class AveragePrecision(tf.keras.metrics.Metric):
    """
    This metric measures the average precision at a specific intersection-over-union
    threshold. The state consists of tensors describing true positive, false positive,
    and false negative detections at various objectness thresholds.
    
    Attributes
    ----------
    iou_thresh: float32
        Threshold for intersection-over-union to determine when a predicted box 
        has detected a ground truth box. Default value 0.5.
    delta: float32
        Increment for thresholds applied to objectness scores for predicted boxes.
        Range is (0, 1). Default value 0.1.
    """
    
    def __init__(self, iou_thresh=0.5, delta=0.1, name='average_precision', **kwargs):
        """
        The constructor accepts the 'iou_thresh' and 'delta' attributes.
        """
        
        super(AveragePrecision, self).__init__(name=name, **kwargs)

        #set iou threshold and objectness increment
        self.iou_thresh = iou_thresh
        self.delta = delta

        #calculate number of thresholds
        n = tf.size(tf.range(0, 1, delta))

        #set states to zero
        self.tp = self.add_weight(name='tp', shape=n, initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=n, initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=n, initializer='zeros')
    

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the true positive, false positive, and false negative states
        based on predictions and ground truth for a single image.
        
        Parameters
        ----------
        y_true: tensor (float32)
            An M x 4 tensor of ground truth boxes where each row contains
            the x, y coordinates of the upper-left box corner and the box 
            width and height.
        y_pred: tensor (float32)
            An N x 6 tensor containing the objectness scores and regressions
            for N predictions. The first two columns contain the objectness 
            score outputs. The last four columns contain the x, y coordinates
            of the upper-left box corners and the box widths and heights.            
        """
    
        #split predictions into objectness, regression
        obj = tf.gather(y_pred, [0, 1], axis=1)
        reg = tf.gather(y_pred, tf.constant([2, 3, 4, 5], tf.int32), axis=1)

        #generate objectness threshold sequence
        taus = tf.range(0, 1, self.delta)
    
        def threshold(obj, reg, boxes, tau, min_iou):

            #call objects from rpn
            positive = tf.greater_equal(obj[:,1], tau)
            obj = tf.boolean_mask(obj, positive, axis=0)
            reg = tf.boolean_mask(reg, positive, axis=0)

            #greedy mapping regressions to ground-truth boxes
            ious = iou(reg, boxes)
            tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou_mapping(ious, min_iou)        

            return tf.stack([tp, fp, fn], axis=0)
    
        #calculate precision-recall values
        packed = tf.map_fn(lambda x: threshold(obj, reg, y_true, x, self.iou_thresh),
                           taus, parallel_iterations=10, fn_output_signature=tf.int32)

        #unpack
        tp, fp, fn = tf.unstack(packed, axis=1)  

        #update
        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fp.assign_add(tf.cast(fp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))
    

    def result(self):
        """
        Returns the average precision calculated over self.delta width steps for objectness
        score.
        """
        
        #calculate precision
        precision = self.tp / (self.tp + self.fp)

        #add 1 to end
        precision = tf.concat([precision, [1.0]], axis=0)

        #average precision
        ap = tf.reduce_mean(precision)

        return ap


    def reset_state(self):
        """
        Resets the true positive, false positive, and false negative states to zero.
        """
        
        self.tp.assign(tf.zeros(self.tp.shape))
        self.fp.assign(tf.zeros(self.fp.shape))
        self.fn.assign(tf.zeros(self.fn.shape))
