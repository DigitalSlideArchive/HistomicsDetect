import tensorflow as tf


class FalseNegativeRate(tf.keras.metrics.Metric):
    """This metric measures false negative rate at specified thresholds to assess
    performance of the binary classifications.
    
    Attributes
    ----------
    thresholds: float32
        Thresholds in range [0, 1]. Default value 0.5.
    false_negatives: float32
        Running sum of false negatives observed at each threshold.
    positives: float32
        Running sum of true positives.
    """

    def __init__(self, thresholds=None, name='fnr', **kwargs):
        super(FalseNegativeRate, self).__init__(name=name, **kwargs)
        if thresholds is None:
            self.thresholds = tf.convert_to_tensor([0.5])
        else:
            self.thresholds = tf.convert_to_tensor(thresholds)
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros',
                                               shape=(tf.size(self.thresholds),))
        self.positives = self.add_weight(name='positives', initializer='zeros')

    def reset_state(self):
        self.false_negatives.assign(tf.zeros(tf.shape(self.false_negatives), tf.float32))
        self.positives.assign(0.)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #generate binary matrix of false positives
        binary = tf.less_equal(tf.reshape(y_pred, (tf.size(y_pred), 1)),
                               self.thresholds)
        positives = tf.equal(tf.cast(y_true, tf.bool), True)
        false_negatives = tf.logical_and(tf.reshape(positives, (tf.size(y_true), 1)),
                                         binary)
        
        #cast to float for possible weighting
        positives = tf.cast(positives, tf.float32)
        false_negatives = tf.cast(false_negatives, tf.float32)
        
        #weight samples
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.broadcast_to(sample_weight, (tf.size(y_true), 1))
            negatives = tf.multiply(positives, sample_weight)
            false_negatives = tf.multiply(false_negatives, sample_weight)
                
        #reduce along sample axes
        self.positives.assign_add(tf.reduce_sum(positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives, axis=0))

    def result(self):
        return tf.cond(tf.equal(tf.reduce_sum(self.positives), 0.0),
                       lambda: 0.0,
                       lambda: self.false_negatives / self.positives)
    

class FalsePositiveRate(tf.keras.metrics.Metric):
    """This metric measures false positive rate at specified thresholds to assess
    performance of the binary classifications.
    
    Attributes
    ----------
    thresholds: float32
        Thresholds in range [0, 1]. Default value 0.5.
    false_positives: float32
        Running sum of false positives observed at each threshold.
    negatives: float32
        Running sum of true negatives.
    """
    
    def __init__(self, thresholds=None, name='fpr', **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        if thresholds is None:
            self.thresholds = tf.convert_to_tensor([0.5])
        else:
            self.thresholds = tf.convert_to_tensor(thresholds)
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros',
                                               shape=(tf.size(self.thresholds),))
        self.negatives = self.add_weight(name='negatives', initializer='zeros')
        
    def reset_state(self):
        self.false_positives.assign(tf.zeros(tf.shape(self.false_positives), tf.float32))
        self.negatives.assign(0.)

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #generate binary matrix of false positives
        binary = tf.greater(tf.reshape(y_pred, (tf.size(y_pred), 1)),
                            self.thresholds)
        negatives = tf.equal(tf.cast(y_true, tf.bool), False)
        false_positives = tf.logical_and(tf.reshape(negatives, (tf.size(y_true), 1)),
                                         binary)
        
        #cast to float for possible weighting
        negatives = tf.cast(negatives, tf.float32)
        false_positives = tf.cast(false_positives, tf.float32)
        
        #weight samples
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.broadcast_to(sample_weight, (tf.size(y_true), 1))
            negatives = tf.multiply(negatives, sample_weight)
            false_positives = tf.multiply(false_positives, sample_weight)
                
        #reduce along sample axes
        self.negatives.assign_add(tf.reduce_sum(negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives, axis=0))

    def result(self):
        return tf.cond(tf.equal(tf.reduce_sum(self.negatives), 0.0),
                       lambda: 0.0,
                       lambda: self.false_positives / self.negatives)
