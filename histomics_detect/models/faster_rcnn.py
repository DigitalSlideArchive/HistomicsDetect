from histomics_detect.anchors.create import create_anchors
from histomics_detect.anchors.filter import filter_anchors
from histomics_detect.anchors.sampling import sample_anchors
from histomics_detect.boxes import parameterize, unparameterize, clip_boxes, tf_box_transform, filter_edge_boxes
from histomics_detect.metrics import iou, greedy_iou_mapping, AveragePrecision, FalsePositiveRate, FalseNegativeRate
from histomics_detect.networks.backbones import pretrained, residual
from histomics_detect.networks.rpns import rpn
from histomics_detect.networks.fast_rcnn import fast_rcnn
from histomics_detect.networks.field_size import field_size
from histomics_detect.roialign.roialign import roialign
import tensorflow as tf


@tf.function
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
    index = tf.equal(tf.expand_dims(anchors[:,3], axis=1),
                     tf.expand_dims(tf.cast(anchor_px, tf.float32), axis=0))
    index = tf.cast(tf.where(index)[:,1], tf.int32)

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


def faster_rcnn_config():
    """Generates a default configuration for a faster RCNN model, setting default
    parameters for the backbone, rpn, and fast RCNN sub-networks, as well as 
    training and validation settings.
    
    Returns
    -------
    backbone_args: dict
        Contains name of backbone and backbone parameters. Parameters vary 
        by backbone type. Defaults to a resnet50 backbone with a stride of 1
        and 14 residual blocks.    
    rpn_args: dict
        Describes the region proposal architecture, specifying the number
        of layers, kernel sizes, output dimensions, and activations.
    frcnn_args:
        Describes the fast RCNN architecture, including the number and size
        of dense layers, dense layer activations, and the roialign parameters.
    train_args:
        Describes the shape of the training images (uniform), the max number
        of anchors to sample per image during training, the maximum ratio 
        of negative to positive anchors sampled, and the loss weighting of
        region proposal classifier and regression losses.
    validation_args:
        Describes objectness thresholds, iou thresholds for nms, iou threshold
        to evaluate true positive / false positive / false negative rate, and
        a sequence of iou thresholds to evaluate mean average precision.
    """
    
    #feature network parameters
    backbone_args = {'name': 'resnet50v2',
                     'stride': 1, #stride (pixels) in first backbone convolution
                     'blocks': 14} #number of residual blocks to use in backbone

    #rpn network parameters
    rpn_args = {'kernels': [3], #kernel sizes (receptive fields) for rpn convolutions
                'dimensions': [256], #number of kernels per layer
                'activations': ['relu']} #activation for rpn convolutions

    #fast-rcnn network parameters
    frcnn_args = {'units': [1024, 1024], #number of units in fast-rcnn dense layers
                  'activations': ['relu', 'relu'], #activations for each dense layer
                  'pool': 2, #number of tiles to pool during roialign
                  'tiles': 3} #number of tiles to split regressed boxes into during roialign

    #training parameters
    train_args = {'train_shape': (224, 224, 3), #shape of training instances
                  'max_anchors': 256, #maximum number of negative anchors to sample per epoch
                  'np_ratio': 1.0, #largest ratio of negative : positive anchors per batch
                  'lmbda': 10.0, #weighting factor for region-proposal network regression loss
                  'hard_fraction': 0.0} #fraction of sampled negative anchors that are hard

    #validation parameters
    validation_args = {'tau': 0.5, #objectness threshold used to classify anchors at inference
                       'nms_iou': 0.3, #min nms threshold used to filter overlapping objects
                       'tpr_iou': 0.5, #single iou threshold used to calculate tpr/fpr/fnr
                       'margin': 32, #margin of image edge to exclude from validation (pixels)
                       'ap_ious': [0.25, 0.5, 0.75], #ious thresholds to use in evaluating mAP
                       'ap_delta': 0.1} #precision step size for calculating mAP

    return backbone_args, rpn_args, frcnn_args, train_args, validation_args


class FasterRCNN(tf.keras.Model):
    """
    This class implements a faster RCNN model which combines a backbone feature
    extraction network with a region proposal network. RoiAlign is used to 
    interpolate and pool features for the variable-sized proposed boxes to support
    refinement of boxes or downstream operations like classification (future 
    enhancement). Class methods provide different levels of output processing,
    from raw proposed regions and objectness scores, to proposals thresholded by
    objectness score, non-max suppressed, and refined by roialign. Utility 
    functions are available to apply non-max suppression or roialign. Training is
    performed in single image batches.
    
    Attributes
    ----------
    backbone_args: array-like (dict)
        Arguments used to construct model backbone.
    rpn_args: array-like (dict)
        Arguments used to construct region-proposal network.
    frcnn_args: array-like (dict)
        Arguments used to construct fast-rcnn network.
    train_args: array-like (dict)
        Arguments used to specify training behavior.
    validation_args: array-like (dict)
        Arguments used to specify validation behavior.
    backbone: tf.keras.Model
        A fully-convolutional backbone model that produces an M x N x D feature map.
    rpnetwork: tf.keras.Model
        A convolutional model that produces objectness and parameterized regressions.
    fastrcnn: tf.keras.Model
        A fully-connected model that operates on roialigned outputs to produce
        refined regressions. Future enhancements will extend this network to perform
        classification.
    anchor_px: tensor (int32)
        One dimensional tensor of anchor sizes.
    field: float (integer-valued)
        The field size in pixels of the backbone network.
    pool: int32
        RoiAlign parameter. pool^2 is the number of locations to interpolate 
        features at within each tile. Default value 2.
    tiles: int32
        RoiAlign parameter. tile^2 is the number of tiles that each regressed 
        bounding box is divided into. Default value 3.
    lmbda: float32
        Loss weight for balancing objectness and regression losses of region
        proposal network. Default value 10.0.
    max_anchors: int32
        Maximum number of total anchors to sample. Default value 256.
    np_ratio: float32
        Will sample at most negative : positive ratio anchors. Default value 2.0.
    tau: float32
        Threshold in range [0,1] used to select region proposals based on 
        region proposal network objectness scores. Default value 0.5.
    nms_iou: float32
        Intersection over union threshold used to remove redundant proposals
        during non-max suppression. Range is (0, 1]. Default value 0.3.
    tpr_iou: float32
        Minimum intersection over union threshold between a proposal and ground
        truth object to call that proposal a detection. Used in calculating
        test performance statistics.
    margin: int32
        The margin value is used to clear ground truth objects and proposals
        from the edges of test images. All objects intersecting partially or
        wholly with this margin will be removed. Default value 32 pixels.
    objectness_metrics: list (tf.keras.Metric)
        A list of tf.keras.Metric objects that can assess the binary
        classification performance of the objectness score from the region
        proposal network. Defaults to precision-recall auc, true positive rate,
        false negatives, and false positives.
    regression_metrics: list (tf.keras.Metric)
        A list of tf.keras.Metric objects that can assess the regression
        performance region proposal network or RoiAlign refined regressions.
        Defaults to average precision at 0.25, 0.50, and 0.75 iou thresholds
        with an objectness threshold step size of 0.1.
    anchors: tensor(float32)
        Stored anchor locations for preset training size to reduce calculations 
        during training.
    
    Methods
    -------
    call
        Produces roialigned, non-max suppressed, and objectness thresholded
        predictions given 
    constructor
        Initializes model, constructing all networks and capturing all parameters
        needed for training and validation. Captures all data necessary to 
        produce configs needed for saving as a keras model.
    input_size
        Used to set or reset input image size dimensions. Useful during training
        and tiled inference.
    threshold
        Applies a threshold to objectness scores from the region proposal network
        and returns filtered proposed boxes and objectness scores.
    nms
        Applies non-max suppression to region proposals using objectness scores.
    align
        Applies RoiAlign to region proposals.
    raw
        Generates the full unfiltered outputs from the region proposal network
        inclding proposal regressions and their objectness scores.
    call
        Generates the objectness thresholded, non-max suppressed, and RoiAligned
        proposals.
    """
    
    def __init__(self, backbone_args, rpn_args, frcnn_args, train_args, validation_args, anchor_sizes,
                 **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)

        #capture input configuration parameters
        self.backbone_args = backbone_args
        self.rpn_args = rpn_args
        self.frcnn_args = frcnn_args
        self.train_args = train_args
        self.validation_args = validation_args
        self.anchor_sizes = anchor_sizes
        
        #build backbone, rpn, and terminal network
        backbone, preprocessor = pretrained(backbone_args['name'])
        self.backbone = residual(backbone, preprocessor, backbone_args['blocks'], backbone_args['stride'])
        self.rpnetwork = rpn(self.backbone.output.shape[-1], len(anchor_sizes), **rpn_args)
        self.fastrcnn = fast_rcnn(self.backbone.output.shape[-1], **frcnn_args)        

        #capture field, anchor sizes, loss mixing
        self.field = tf.cast(field_size(self.backbone), tf.float32)
        
        #capture roialign parameters
        self.pool = frcnn_args['pool']
        self.tiles = frcnn_args['tiles']
        
        #convert anchor_size to tensor
        self.anchor_px = tf.constant(anchor_sizes, dtype=tf.int32)
        
        #capture training arguments
        self.lmbda = train_args['lmbda'] #loss mixing weights
        self.max_anchors = train_args['max_anchors'] #max negative anchors to sample
        self.np_ratio = train_args['np_ratio'] #max acceptable ratio of negative : positive anchors
        self.hard_fraction = train_args['hard_fraction'] #fraction of sampled anchors that are hard
        
        #capture validation arguments
        self.tau = validation_args['tau']
        self.nms_iou = validation_args['nms_iou']
        self.tpr_iou = validation_args['tpr_iou']
        self.margin = validation_args['margin']
        
        #generate anchors for training efficiency - works for fixed-size training
        self.anchors = create_anchors(self.anchor_px, self.field, 
                                      train_args['train_shape'][0], 
                                      train_args['train_shape'][1])

        #define metrics
        self.objectness_metrics = [tf.keras.metrics.AUC(curve="PR", name='prauc'),
                                   tf.keras.metrics.Recall(name='tpr'),
                                   FalsePositiveRate(name='fpr'),
                                   FalseNegativeRate(name='fnr')]
        self.regression_metrics = [AveragePrecision(iou_thresh = t, 
                                                    delta=validation_args['ap_delta'],
                                                    name='ap' + str(int(100*t)))
                                   for t in validation_args['ap_ious']]


    def get_config(self):
        return {'backbone_args': self.backbone_args,
                'rpn_args': self.rpn_args,
                'frcnn_args': self.frcnn_args,
                'train_args': self.train_args,
                'validation_args': self.validation_args,
                'anchor_sizes': self.anchor_sizes}
    
    
    @classmethod
    def from_config(cls, config):
        #defining this function is required to support keras restore
        return cls(**config)
    
    
    def _update_objectness_metrics(self, objectness, positive):
        """
        Updates objectness metrics that are based on binary classification performance.
        
        Parameters
        ----------
        objectness: tensor
            N x 2 tensor containing corresponding softmax objectness scores in rows.
            Second column contains score for being an object.
        positive: tensor (bool)
            N length bool tensor indicating which rows contain objects that were
            judged positive based on

        Returns
        -------
        metrics: dict
            Returns a dict of updated metric values keyed by metric names.       
        """
        
        #update metrics values
        for metric in self.objectness_metrics:
            metric.update_state(tf.cast(positive, tf.uint32), objectness[:,1])
        
        return {m.name: m.result() for m in self.metrics}
    
    
    def _update_regression_metrics(self, boxes, predictions):
        """
        Updates objectness metrics that are based on binary classification performance.
        
        Parameters
        ----------
        boxes: tensor (float32)
            M x 4 tensor where each row contains the x,y location of the upper left
            corner of a ground-truth box and its width and height in that order.
        predictions: tensor (float32)
            N x 6 tensor where each row contains the objectness scores and regressed
            boxes for one proposal.

        Returns
        -------
        metrics: dict
            Returns a dict of updated metric values keyed by metric names.       
        """
        
        #update metrics values
        for metric in self.regression_metrics:
            metric.update_state(boxes, predictions)
        
        return {m.name: m.result() for m in self.metrics}
    
        
    def input_size(self, size=[None, None]):
        """
        Sets input size dimensions for the backbone and region proposal networks. 
        By default, the backbone and region proposal network have variable input image '
        sizes. In some cases this may reduce the speed of training or inference. When
        training or performing inference on uniform sized images, this procedure allows
        the input to be set to a specific size. Default parameters reset these networks 
        to variable size inputs.

        Parameters
        ----------
        size: list of integers
            The height and width of the input images for the backbone. Size of input to
            region proposal network is calculated by devault. Default value is 
            [None, None] for a variable input size.
        """
        
        #get current backbone and rpnetwork channels
        backbone_channels = self.backbone.layers[0].input_shape[0][-1]
        rpnetwork_channels = self.rpnetwork.layers[0].input_shape[0][-1]
        
        #calculate rpnetwork input size from field size and input size
        if all([dim is None for dim in size]):
            rpnetwork_size = [None, None, rpnetwork_channels]
        else:
            rpnetwork_size = [tf.cast(tf.math.ceil(size[0]/self.field), tf.int32),
                              tf.cast( tf.math.ceil(size[1]/self.field), tf.int32), 
                              rpnetwork_channels]
        
        #pop input layers
        self.backbone.layers.pop(0)
        self.rpnetwork.layers.pop(0)
                              
        #create new input layers
        backbone_input = tf.keras.Input(shape = size + [backbone_channels])
        rpnetwork_input = tf.keras.Input(shape = rpnetwork_size)
        
        #update backbone
        backbone_output = self.backbone(backbone_input)
        self.backbone = tf.keras.Model(backbone_input, backbone_output)
        
        #update rpnetwork
        rpnetwork_output = self.rpnetwork(rpnetwork_input)
        self.rpnetwork = tf.keras.Model(rpnetwork_input, rpnetwork_output)
        
        
    @tf.function
    def threshold(self, boxes, objectness, tau):
        """
        Thresholds rpn predictions using objectness score. Helpful for processing
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
            predictions are objects. Range is [0,1].
        
        Returns
        -------
        filtered_boxes: tensor
            M x 4 tensor containing objectness score filtered boxes (M <= N).
        filtered_objectness tensor
            M x 2 tensor containing objectness score filtered objectness scores (M <= N).       
        """
        
        #get binary mask of positive objects
        mask = tf.greater(objectness[:,1], tau)
        
        #filter using mask
        filtered_boxes = tf.boolean_mask(boxes, mask, axis=0)
        filtered_objectness = tf.boolean_mask(objectness, mask, axis=0)       
        
        return filtered_boxes, filtered_objectness, mask
        
        
    @tf.function
    def nms(self, boxes, objectness, nms_iou):
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
            (0,1].
            
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
        using objectness thresholding or non-max suppression prior to this step an OOM 
        error may occur. Helpful for refinement of post-processed inference results from 
        raw.
        
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
        align_boxes: tensor (float32)
            N x 4 tensor containing aligned boxes.
        nms_objectness: tensor (float32)
            N x 2 tensor containing corresponding objectness scores.
        """
 
        #interpolate features in pool*tiles x pool*tiles grid for each box
        interpolated = roialign(features, boxes, field, pool, tiles)
        
        #generate roialign predictions and transform to box representation
        align_reg = self.fastrcnn(interpolated)
        align_boxes = unparameterize(align_reg, boxes)
        
        return align_boxes
    
    
    @tf.function
    def raw(self, rgb):
        """
        raw() produces unfiltered objectness scores and regressions from the rpn
        network, and backbone features. Additional steps are required for thresholding
        based on scores, nms, and performing roialign. This is useful for users who
        would like to provide their own post-processing of rpn results.
        
        Parameters
        ----------
        rgb: tensor (uint8)
            An M x N x 3 image to perform inference on.
       
        Returns
        -------
        rpn_obj: tensor (float32)
            D x 2 tensor containing objectness scores from the region proposal network.
        rpn_boxes: tensor (float32)
            D x 4 tensor containing regressions from the region proposal network.
        features: tensor (float32)
            An M x N x K tensor of feature maps generated by the backbone. Can be used 
            to subsequently apply RoiAlign or other operations.
        """

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
    def call(self, rgb, tau=None, nms_iou=None, margin=0):
        """
        call() produces thresholded and roialign refined predictions from a trained
        network. This is the most useful for users who don't want to apply their own
        post-processing to rpn results.
        
        Parameters
        ----------
        rgb: tensor (uint8)
            An M x N x 3 image to perform inference on.
        tau: float32
            Threshold in range [0,1] used to select region proposals based on 
            region proposal network objectness scores. Defaults to model attribute 
            (default value 0.5).
        nms_iou: float32
            Intersection over union threshold used to remove redundant proposals
            during non-max suppression. Range is (0, 1]. Defaults to model
            attribute (default value 0.3).
        margin: int32
            The margin value is used to clear predictions from the edge of the
            input image. All objects intersecting partially or wholly with this 
            margin will be removed. Default value 0 pixels performs no clearing.
            
        Returns
        -------
        align_boxes: tensor (float32)
            A D x 4 tensor of objectness thresholded, non-max suppressed, and 
            RoiAligned regoin proposals.
        """
        
        #generate raw rpn outputs
        rpn_obj, rpn_boxes, features = self.raw(rgb)
        
        #select rpn proposals
        if tau is None:
            tau = self.tau
        rpn_boxes_positive, rpn_obj_positive, positive = self.threshold(rpn_boxes, rpn_obj, 
                                                                        tau)
        
        #perform non-max suppression on rpn positive predictions
        if nms_iou is None:
            nms_iou = self.nms_iou
        rpn_boxes_nms, rpn_obj_nms, selected = self.nms(rpn_boxes_positive,
                                                        rpn_obj_positive, nms_iou)
        
        #generate roialign predictions for rpn positive predictions
        align_boxes = self.align(rpn_boxes_nms, features, self.field, 
                                 self.pool, self.tiles)
        
        #filter edge boxes
        filtered, mask = filter_edge_boxes(align_boxes, tf.shape(rgb)[1],
                                           tf.shape(rgb)[0], margin)
        
        return filtered

        
    @tf.function(experimental_relax_shapes=True)
    def test_step(self, data):
        """
        Test step performs inference on a single image, calculating both single-image
        and aggregate metrics. Single-image metrics are printed to the stdout.
        """
        
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
     
        #evaluate rpn - generate anchors for input size image
        anchors = create_anchors(self.anchor_px, self.field, 
                                 tf.shape(rgb)[1], tf.shape(rgb)[0])
        
        #evaluate rpn - filter into positive and negative anchors
        positive_anchors, negative_anchors = filter_anchors(boxes, anchors)

        #evaluate rpn - generate objectness scores
        output = self.rpnetwork(features, training=False)[0]
 
        #evaluate rpn - calculate objectness predictions and labels
        positive_obj = tf.nn.softmax(map_outputs(output, positive_anchors,
                                                 self.anchor_px, self.field))
        negative_obj = tf.nn.softmax(map_outputs(output, negative_anchors,
                                                 self.anchor_px, self.field))
        obj_labels = tf.concat([tf.ones(tf.shape(positive_obj)[0], tf.uint8),
                                tf.zeros(tf.shape(negative_obj)[0], tf.uint8)],
                                axis=0)
        
        #select positive rpn proposals for nms and roialign
        rpn_boxes_pred, rpn_obj_pred, _ = self.threshold(rpn_boxes, rpn_obj, self.tau)
        
        #perform non-max suppression on boxes
        rpn_boxes_nms, rpn_obj_nms, selected = self.nms(rpn_boxes_pred,
                                                        rpn_obj_pred, self.nms_iou)
        
        #generate roialign predictions for rpn positive predictions
        align_boxes_nms = self.align(rpn_boxes_nms, features, self.field, self.pool, self.tiles)
        
        #clear margin of ground truth boxes
        boxes, _ = filter_edge_boxes(boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], self.margin)     
        
        #filter edge boxes
        filtered, mask = filter_edge_boxes(align_boxes_nms, tf.shape(rgb)[1],
                                           tf.shape(rgb)[0], self.margin)
        filtered_objectness = tf.boolean_mask(rpn_obj_nms, mask, axis=0)
        
        #calculate ious
        align_ious = iou(filtered, boxes)
        
        #greedy iou mapping for precision-recall auc
        tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou_mapping(align_ious, self.tpr_iou)
        
        #update console
        tf.print(name)
        tf.print('greedy tp: ', tp)
        tf.print('greedy fp: ', fp)
        tf.print('greedy fn: ', fn)
        
        #best iou with ground truth for each prediction - reduce along rows
        align_ious = tf.reduce_max(align_ious, axis=1)
        
        #update objectness binary classification metrics
        obj_metrics = self._update_objectness_metrics(tf.concat([positive_obj,
                                                                 negative_obj],
                                                                axis=0),
                                                      obj_labels)
        
        #update regression metrics
        reg_metrics = self._update_regression_metrics(boxes,
                                                      tf.concat([filtered_objectness,
                                                                 filtered],
                                                                axis=1))
        
        #combine metrics
        metrics = {**obj_metrics, **reg_metrics}
        
        #dummy losses
        losses = {'loss_rpn_obj': 0., 'loss_rpn_reg': 0., 'loss_align_reg': 0.}        
        
        return {**losses, **metrics}
    
    
    @tf.function
    def train_step(self, data):
        """
        Train step performs gradient updates on single image batches. Aggregate metrics are
        calculated.
        """
    
        #unpack image, boxes, and optional image name
        if len(data) ==3:
            rgb, boxes, name = data
        else:
            rgb, boxes = data

        #convert boxes from RaggedTensor
        boxes = boxes.to_tensor()

        #expand dimensions
        norm = tf.expand_dims(tf.cast(rgb, tf.float32), axis=0)

        #label anchors as negative, positive
        positive_anchors, negative_anchors = filter_anchors(boxes, self.anchors)

        #training step
        with tf.GradientTape(persistent=True) as tape:
            
            #predict and capture intermediate features
            features = self.backbone(norm, training=True)
            output = self.rpnetwork(features, training=True)
            
            #calculate rpn objectness loss for all positive and negative anchors
            positive_obj = tf.nn.softmax(map_outputs(output[0], positive_anchors,
                                                         self.anchor_px, self.field))
            negative_obj = tf.nn.softmax(map_outputs(output[0], negative_anchors,
                                                         self.anchor_px, self.field))
            
            #sample anchors and perform hard-negative mining
            positive_anchors, negative_anchors = sample_anchors(positive_anchors, 
                                                                negative_anchors,
                                                                negative_obj[:,1],
                                                                self.max_anchors, 
                                                                self.np_ratio,
                                                                self.hard_fraction)

            #calculate objectness loss for sampled positive and negative anchors
            positive_obj = tf.nn.softmax(map_outputs(output[0], positive_anchors,
                                                         self.anchor_px, self.field))
            negative_obj = tf.nn.softmax(map_outputs(output[0], negative_anchors,
                                                         self.anchor_px, self.field))
            obj_labels = tf.concat([tf.ones(tf.shape(positive_obj)[0], tf.uint8),
                                        tf.zeros(tf.shape(negative_obj)[0], tf.uint8)],
                                       axis=0)
            obj_loss = self.loss[0](tf.concat([positive_obj, negative_obj], axis=0),
                                    tf.one_hot(obj_labels, 2))

            #ccalculate rpn regression loss
            regression = map_outputs(output[1], positive_anchors, self.anchor_px,
                                     self.field)
            reg_label = parameterize(positive_anchors, boxes)
            reg_loss = self.loss[1](reg_label, regression)          

            #weighted sum of objectness and regression losses
            rpn_loss = obj_loss / self.max_anchors + \
                reg_loss * self.lmbda / tf.cast(tf.shape(self.anchors)[0], tf.float32)
            
            #roi align and fast rcnn operations to be performed if positive anchors exist
            def nonempty():
            
                #fast r-cnn regression of roialign results for positive anchors
                rpn_boxes = unparameterize(regression, positive_anchors)
                rpn_boxes_positive, _ = filter_anchors(boxes, rpn_boxes)
                interpolated = roialign(features, rpn_boxes_positive, self.field, 
                                        pool=self.pool, tiles=self.tiles)
                align = self.fastrcnn(interpolated)
            
                #transform aligned regressions to box representation and calculate labels
                align_boxes = unparameterize(align, rpn_boxes_positive)
                align_label = parameterize(rpn_boxes_positive, boxes)
            
                return self.loss[1](align_label, align)
        
            def empty():
                return 0.0
        
            #conditional when positive anchors exist
            align_loss = tf.cond(tf.size(positive_anchors) > 0, nonempty, empty)
                
        #calculate backbone gradients and optimize
        gradients = tape.gradient(rpn_loss, self.backbone.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_weights))

        #calculate rpn gradients and optimize
        gradients = tape.gradient(rpn_loss, self.rpnetwork.trainable_weights)  
        self.optimizer.apply_gradients(zip(gradients, self.rpnetwork.trainable_weights))
            
        #calculate roialign gradients and optimize
        gradients = tape.gradient(align_loss, self.fastrcnn.trainable_weights)  
        self.optimizer.apply_gradients(zip(gradients, self.fastrcnn.trainable_weights))
        
        #update metrics
        metrics = self._update_objectness_metrics(tf.concat([positive_obj,
                                                             negative_obj], 
                                                            axis=0),
                                                  obj_labels)

        #build output loss dict
        losses = {'loss_rpn_obj': obj_loss, 'loss_rpn_reg': reg_loss,
                  'loss_align_reg': align_loss}

        return {**losses, **metrics}
