from histomics_detect.anchors.create import create_anchors
from histomics_detect.anchors.filter import filter_anchors
from histomics_detect.anchors.sampling import sample_anchors
from histomics_detect.boxes import parameterize, unparameterize, clip_boxes, tf_box_transform, filter_edge_boxes
from histomics_detect.metrics import iou, greedy_iou
from histomics_detect.networks.fast_rcnn import fast_rcnn
from histomics_detect.networks.field_size import field_size
from histomics_detect.roialign.roialign import roialign
import tensorflow as tf


def map_outputs(output, anchors, anchor_px, field):
    """Transforms region-proposal network outputs from 3D tensors to 2D anchor arrays.
    
    The region-proposal network outputs 3D tensors containing either objectness scores 
    or parameterized regressions. Given a set of K anchor sizes, the objectness and
    regression tensors produced by the region-proposal network will have sizes 2*K 4*K
    along their third dimension respectively. This function transforms these 3D tensors
    to 2D tensors where the objectness or regressions for anchors are in rows.
        
    Parameters
    ----------
    output: tensor (float32)
        M x N x D tensor containing objectness or regression outputs from the 
        region-proposal network.
    anchors: tensor (float32)
        M*N*K x 4 tensor of anchor positions. Each row contains the x,y upper left
        corner of the anchor in pixels relative in the image coordinate frame, 
        and the anchor width and height.
    anchor_px: tensor (int32)
        K-length 1-d tensor containing the anchor width hyperparameter values in pixel 
        units.
    field: float32
        Edge length of the receptive field in pixels. This defines the area of the 
        image that corresponds to 1 feature map and the anchor gridding.
        
    Returns
    -------
    mapped: tensor (float32)
        M*N*K x 2 array of objectness scores or M*N*K x 4 tensor of regressions
        where each row represents one anchor.
    """
  
    #get anchor size index for each matching anchor
    index = tf.map_fn(lambda x: tf.argmax(tf.equal(x, anchor_px),
                                          output_type=tf.int32),
                      tf.cast(anchors[:,3], tf.int32))

    #use anchor centers to get positions of anchors in rpn output
    px = tf.cast((anchors[:,0]+ anchors[:,2]/2) / field, tf.int32)
    py = tf.cast((anchors[:,1]+ anchors[:,3]/2) / field, tf.int32)

    #add new dimension to split outputs by anchor (batch, y, x, anchor, output)
    reshaped = tf.reshape(output, tf.concat([tf.shape(output)[0:-1],
                                             [tf.size(anchor_px)],
                                             [tf.shape(output)[-1] /
                                              tf.size(anchor_px)]],
                                            axis=0))

    #gather from (batch, y, x, anchor, output) space to 2D array where each row is 
    #an anchor and columns are objectness (2-array) or regression (4-array) scores
    mapped = tf.gather_nd(reshaped,
                          tf.stack([tf.zeros(tf.shape(px), tf.int32),
                                    py, px, index],
                                   axis=1))

    return mapped


class FasterRCNN(tf.keras.Model):
    def __init__(self, rpnetwork, backbone, shape, anchor_px, lmbda, 
                 pool=2, tiles=3, nms_iou=0.3, map_iou=0.5,
                 **kwargs):        
    
        super(FasterRCNN, self).__init__()

        #add models to self
        self.rpnetwork = rpnetwork
        self.backbone = backbone
        self.fastrcnn = fast_rcnn(backbone, tiles=tiles, pool=pool)

        #capture field, anchor sizes, loss mixing
        self.field = tf.cast(field_size(backbone), tf.float32)
        self.anchor_px = anchor_px
        self.lmbda = lmbda
        
        #capture roialign parameters
        self.pool = pool
        self.tiles = tiles
        
        #parameters for nms
        self.nms_iou = nms_iou
        self.map_iou = map_iou        
        
        #generate anchors for training efficiency - works for fixed-size training
        self.anchors = create_anchors(anchor_px, self.field, shape[0], shape[1])

        #define metrics
        mae_xy = tf.keras.metrics.Mean(name='mean_iou_rpn')
        mae_wh = tf.keras.metrics.Mean(name='mean_iou_align')
        auc_roc = tf.keras.metrics.AUC(curve="ROC", name='auc_roc')
        auc_pr = tf.keras.metrics.AUC(curve="PR", name='pr_roc')
        tp = tf.keras.metrics.TruePositives()
        fn = tf.keras.metrics.FalseNegatives()
        fp = tf.keras.metrics.FalsePositives()
        self.standard = [mae_xy, mae_wh, auc_roc, auc_pr, tp, fn, fp]


    @tf.function
    def threshold(self, boxes, objectness, tau=0.5):
        """Thresholds rpn predictions using objectness score. Helpful for processing
        raw inference results.
        
        Parameters
        ----------
        boxes: tensor
            N x 4 tensor where each row contains the x,y location of the upper left
            corner of a regressed box and its width and height in that order.
        objectness:
            N x 2 tensor containing corresponding softmax objectness scores in rows.
            Second column contains score for being an object.
        tau: float
            Scalar threshold applied to objectness scores to define which rpn 
            predictions are objects. Range is [0,1]. Default value is 0.5.
        
        Returns
        -------
        filtered_boxes: tensor
            M x 4 tensor containing objectness score filtered boxes (M <= N).
        filtered_objectness tensor
            M x 2 tensor containing objectness score filtered objectness scores (M <= N).       
        """
        
        #get binary mask of positive objects
        mask = tf.greater(objectness[:,1], 0.5)
        
        #filter using mask
        filtered_boxes = tf.boolean_mask(boxes, mask, axis=0)
        filtered_objectness = tf.boolean_mask(objectness, mask, axis=0)       
        
        return filtered_boxes, filtered_objectness, mask
        
        
    @tf.function
    def nms(self, boxes, objectness, nms_iou=0.3):
        """Performs nms on regressed boxes returning filtered boxes and corresponding
        objectness scores. Helpful for processing raw inference results.
        
        Parameters
        ----------
        boxes: tensor
            N x 4 tensor where each row contains the x,y location of the upper left
            corner of a regressed box and its width and height in that order.
        objectness: tensor
            N x 2 tensor containing corresponding softmax objectness scores in rows.
            Second column contains score for being an object.
        nms_iou: float
            Scalar threshold used by nms to define overlapping objects. Range is
            (0,1]. Default value is 0.3.
            
        Returns
        -------
        nms_boxes: tensor
            M x 4 tensor containing nms filtered boxes (M <= N).
        nms_objectness tensor
            M x 2 tensor containing nms filtered objectness scores (M <= N).
        """    
        
        #get indices of boxes selected by NMS
        selected = tf.image.non_max_suppression(tf_box_transform(boxes),
                                                objectness[:,1], tf.shape(objectness)[0],
                                                iou_threshold=nms_iou)
        
        #filter intputs to discard unselected boxes
        nms_boxes = tf.gather(boxes, selected, axis=0)
        nms_objectness = tf.gather(objectness, selected, axis=0)
    
        return nms_boxes, nms_objectness, selected
    
    
    @tf.function
    def align(self, boxes, features, field, pool, tiles):
        """Performs roialign on filtered inference results. If results are not filtered
        prior to this step an OOM error may occur. Helpful for processing raw inference results.
        
        Parameters
        ----------
        features: tensor
            The three dimensional feature map tensor produced by the backbone network.
        boxes: tensor
            N x 4 tensor where each row contains the x,y location of the upper left
            corner of an rpn regressed box and its width and height in that order.
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
        align_boxes: tensor
            N x 4 tensor containing aligned boxes.
        nms_objectness tensor
            N x 2 tensor containing corresponding objectness scores.
        """
 
        #interpolate features in pool*tiles x pool*tiles grid for each box
        interpolated = roialign(features, boxes, field, pool, tiles)
        
        #generate roialign predictions and transform to box representation
        align_reg = self.fastrcnn(interpolated)
        align_boxes = unparameterize(align_reg, boxes)
        
        return align_boxes
    
    
    def raw(self, rgb):
        """raw() produces unfiltered objectness scores and regressions from the rpn
        network, and backbone features. Additional steps are required for thresholding
        based on scores, nms, and performing roialign. This is useful for users who
        would like to provide their own post-processing of rpn results."""
        
        #normalize image
        rgb = tf.keras.applications.resnet.preprocess_input(tf.cast(rgb, tf.float32))

        #expand dimensions
        rgb = tf.expand_dims(rgb, axis=0)
        
        #predict and capture intermediate features
        features = self.backbone(rgb, training=False)
        output = self.rpnetwork(features, training=False)

        #generate anchors for input size image
        anchors = create_anchors(self.anchor_px, self.field, 
                                 tf.shape(rgb)[2], tf.shape(rgb)[1])

        #transform outputs to 2D arrays with anchors in rows
        rpn_obj = tf.nn.softmax(map_outputs(output[0], anchors,
                                self.anchor_px, self.field))
        rpn_reg = map_outputs(output[1], anchors, self.anchor_px, self.field)        
        rpn_boxes = unparameterize(rpn_reg, anchors)
        
        #clip regressed boxes to border
        rpn_boxes = clip_boxes(rpn_boxes, tf.shape(rgb)[2], tf.shape(rgb)[1])
        
        return rpn_obj, rpn_boxes, features
    
    
    @tf.function
    def call(self, rgb, threshold=0.5, nms_iou=None):
        """call() produces thresholded and roialign refined predictions from a trained
        network. This is the most useful for users who don't want to apply their own
        post-processing to rpn results."""
        
        #generate raw rpn outputs
        rpn_obj, rpn_boxes, features = self.raw(rgb)
        
        #select rpn proposals
        rpn_boxes_positive, rpn_obj_positive, positive = self.threshold(rpn_boxes, rpn_obj, 
                                                                        threshold)
        
        #perform non-max suppression on rpn positive predictions
        if nms_iou is None:
            nms_iou = self.nms_iou
        rpn_boxes_nms, rpn_obj_nms, selected = self.nms(rpn_boxes_positive,
                                                        rpn_obj_positive, nms_iou)
        
        #generate roialign predictions for rpn positive predictions
        align_boxes = self.align(rpn_boxes_nms, features, self.field, 
                                 self.pool, self.tiles)
        
        return align_boxes

        
    @tf.function
    def test_step(self, data):
        
        #unpack image, boxes, and optional image name
        if len(data) ==3:
            rgb, boxes, name = data
        else:
            rgb, boxes = data
            name = ''

        #convert boxes from RaggedTensor
        boxes = boxes.to_tensor()
        
        #generate rpn predictions
        rpn_obj, rpn_boxes, features = self.raw(rgb)
        
        #select rpn proposals
        rpn_boxes_positive, rpn_obj_positive, positive = self.threshold(rpn_boxes, rpn_obj, 0.5)
        
        #perform non-max suppression on boxes
        rpn_boxes_nms, rpn_obj_nms, selected = self.nms(rpn_boxes_positive,
                                                        rpn_obj_positive, self.nms_iou)
        
        #generate roialign predictions for rpn positive predictions
        align_boxes_nms = self.align(rpn_boxes_nms, features, self.field, self.pool, self.tiles)
        
        #clear margin of ground truth boxes
        boxes = filter_edge_boxes(boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], 32)
        
        #roialign accuracy measures via greedy iou mapping
        filtered = filter_edge_boxes(align_boxes_nms, tf.shape(rgb)[1], tf.shape(rgb)[0], 32)
        align_ious, _ = iou(filtered, boxes)
        precision, recall, tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou(align_ious, 
                                                                              self.map_iou)
        tf.print(name)
        tf.print('align precision: ', precision)
        tf.print('align recall: ', recall)
        tf.print('align tp: ', tp)
        tf.print('align fp: ', fp)
        tf.print('align fn: ', fn)
        
        #generate measures of negative boxes for performance measurement
        negative = tf.logical_not(positive)
        rpn_obj_negative = tf.boolean_mask(rpn_obj, negative, axis=0)
        rpn_obj_labels = tf.concat([tf.ones(tf.shape(rpn_obj_positive)[0], tf.uint8),
                                    tf.zeros(tf.shape(rpn_obj_negative)[0], tf.uint8)],
                                   axis=0)        
        
        #measurements - objectness pr-auc, rpn pos box iou, align box iou, align box greedy iou
        #update metrics
        self.standard[0].update_state(rpn_ious)
        self.standard[1].update_state(align_ious)
        self.standard[2].update_state(rpn_obj_labels, 
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[3].update_state(rpn_obj_labels, 
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[4].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[5].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[6].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        
        #return metrics only
        return {m.name: m.result() for m in self.standard} 
    
    
    @tf.function
    def train_step(self, data):
    
        #unpack image, boxes, and optional image name
        if len(data) ==3:
            rgb, boxes, name = data
        else:
            rgb, boxes = data

        #convert boxes from RaggedTensor
        boxes = boxes.to_tensor()

        #normalize image
        norm = tf.keras.applications.resnet.preprocess_input(tf.cast(rgb, tf.float32))

        #expand dimensions
        norm = tf.expand_dims(norm, axis=0)

        #filter and sample anchors
        positive_anchors, negative_anchors = filter_anchors(boxes, self.anchors)
        positive_anchors, negative_anchors = sample_anchors(positive_anchors, negative_anchors)

        #training step
        with tf.GradientTape(persistent=True) as tape:

            #predict and capture intermediate features
            features = self.backbone(norm, training=True)
            output = self.rpnetwork(features, training=True)

            #transform outputs to 2D arrays with anchors in rows
            rpn_obj_positive = tf.nn.softmax(map_outputs(output[0], positive_anchors, 
                                                     self.anchor_px, self.field))
            rpn_obj_negative = tf.nn.softmax(map_outputs(output[0], negative_anchors, 
                                                     self.anchor_px, self.field))
            rpn_reg = map_outputs(output[1], positive_anchors, self.anchor_px,
                                  self.field)

            #generate objectness and regression labels
            rpn_obj_labels = tf.concat([tf.ones(tf.shape(rpn_obj_positive)[0], tf.uint8),
                                    tf.zeros(tf.shape(rpn_obj_negative)[0], tf.uint8)],
                                   axis=0)
            rpn_reg_label = parameterize(positive_anchors, boxes)

            #calculate objectness and regression labels
            rpn_obj_loss = self.loss[0](tf.concat([rpn_obj_positive,
                                                   rpn_obj_negative], axis=0),
                                        tf.one_hot(rpn_obj_labels, 2))
            rpn_reg_loss = self.loss[1](rpn_reg_label, rpn_reg)

            #weighted sum of objectness and regression losses
            rpn_total_loss = rpn_obj_loss / 256 + \
                rpn_reg_loss * self.lmbda / tf.cast(tf.shape(self.anchors)[0], tf.float32)
            
            #fast r-cnn regression of rpn regressions from positive anchors
            rpn_boxes = unparameterize(rpn_reg, positive_anchors)
            rpn_boxes_positive, _ = filter_anchors(boxes, rpn_boxes)
            interpolated = roialign(features, rpn_boxes_positive, self.field, 
                                    pool=self.pool, tiles=self.tiles)
            align_reg = self.fastrcnn(interpolated)
            
            #calculate fast r-cnn regression loss
            align_boxes = unparameterize(align_reg, rpn_boxes_positive)            
            align_reg_label = parameterize(rpn_boxes_positive, boxes)
            align_reg_loss = self.loss[1](align_reg_label, align_reg)

        #calculate backbone gradients and optimize
        gradients = tape.gradient(rpn_total_loss, self.backbone.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_weights))

        #calculate rpn gradients and optimize
        gradients = tape.gradient(rpn_total_loss, self.rpnetwork.trainable_weights)  
        self.optimizer.apply_gradients(zip(gradients, self.rpnetwork.trainable_weights))
            
        #calculate roialign gradients and optimize
        gradients = tape.gradient(align_reg_loss, self.fastrcnn.trainable_weights)  
        self.optimizer.apply_gradients(zip(gradients, self.fastrcnn.trainable_weights))
    
        #ious for rpn, roialign
        rpn_ious, _ = iou(rpn_boxes, boxes)
        rpn_ious = tf.reduce_max(rpn_ious)
        align_ious, _ = iou(align_boxes, boxes)
        align_ious = tf.reduce_max(align_ious)

        #update metrics

        self.standard[0].update_state(rpn_ious)
        self.standard[1].update_state(align_ious)
        self.standard[2].update_state(rpn_obj_labels, 
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[3].update_state(rpn_obj_labels, 
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[4].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[5].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        self.standard[6].update_state(rpn_obj_labels,
                                      tf.concat([rpn_obj_positive, rpn_obj_negative],
                                                axis=0)[:,1])
        
        #build output dicts
        losses = {'rpn_objectness': rpn_obj_loss, 'rpn_regression': rpn_reg_loss,
                  'align_regression': align_reg_loss}
        metrics = {m.name: m.result() for m in self.standard}    

        return {**losses, **metrics}
    

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.standard[0].reset_states()
        self.model.standard[1].reset_states()
        self.model.standard[2].reset_states()
        self.model.standard[3].reset_states()
        self.model.standard[4].reset_states()
        self.model.standard[5].reset_states()
        self.model.standard[6].reset_states()
