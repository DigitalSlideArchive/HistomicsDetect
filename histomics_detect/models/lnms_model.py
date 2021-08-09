from abc import ABC
import tensorflow as tf

from histomics_detect.networks.field_size import field_size
from histomics_detect.anchors.create import create_anchors
from histomics_detect.models.block_model import BlockModel
from histomics_detect.roialign.roialign import roialign
from histomics_detect.models.faster_rcnn import map_outputs
from histomics_detect.boxes.transforms import unparameterize
from histomics_detect.models.lnms_loss import normal_loss, clustering_loss, paper_loss, xor_loss
from histomics_detect.metrics.lnms import lnms_metrics
from histomics_detect.models.model_utils import extract_data
from histomics_detect.boxes.match import cluster_assignment


class LearningNMS(tf.keras.Model, ABC):
    def __init__(self, configs: dict, rpnetwork: tf.keras.Model, backbone: tf.keras.Model,
                 compression_network: tf.keras.Model, shape, *args, **kwargs):
        """
        Learning-NMS model for training a Block-Model

        Parameters
        ----------
        configs: dictionary
            configuration dict
        rpnetwork: tf.keras.Model
            rpnetwork for extracting the boxes from the image
        backbone: tf.keras.Model
            network to extract image features
        shape:
        args
        kwargs
        """
        super(LearningNMS, self).__init__(*args, **kwargs)
        self.configs = configs

        for key, value in configs.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # add models to self
        self.rpnetwork = rpnetwork
        self.backbone = backbone
        self.lmbda = configs['rpn_lmbda']

        self.field = field_size(self.backbone)
        self.anchors = create_anchors(self.anchor_px, self.field, shape[0], shape[1])

        self.net = BlockModel([self._initialize_block(i) for i in range(self.num_blocks)],
                              self._initialize_final_output(), threshold=self.threshold,
                              train_tile=self.train_tile, use_image_features=self.use_image_features)
        self.compression_net = compression_network

        # define metrics
        loss_col = tf.keras.metrics.Mean(name='loss')
        loss_pos = tf.keras.metrics.Mean(name='pos_loss')
        loss_neg = tf.keras.metrics.Mean(name='neg_loss')
        tp = tf.keras.metrics.Mean(name='tp')
        fp = tf.keras.metrics.Mean(name='fp')
        tn = tf.keras.metrics.Mean(name='tn')
        fn = tf.keras.metrics.Mean(name='fn')
        self.standard = [loss_col, loss_pos, loss_neg, tp, fp, tn, fn]

    def _initialize_final_output(self) -> tf.keras.Model:
        """
        builds the final layers of the lnms network

        'num_hidden_layers' Dense layers with 'final_hidden_layer_features' per layer
        output dimension is 1

        Returns
        -------
        final_output: tf.keras.Model
            network for compressing neighborhood to single prediction vector
        """
        final_input = tf.keras.Input(shape=self.feature_size + 1, name="final_after_block_layers_input")
        x = final_input
        for i in range(self.num_hidden_layers):
            x = tf.keras.layers.Dense(self.final_hidden_layer_features, activation=self.activation, use_bias=True,
                                      name=f'final_after_block_layer_{i}')(x)
        x = tf.keras.layers.Dense(1, activation=self.final_activation, name="output_score_layer", use_bias=True)(x)

        return tf.keras.Model(inputs=final_input, outputs=x)

    def _initialize_block(self, block_id: int):
        """
        builds two keras models for each block

        block:
            - block: 'num_layers_block' hidden dense layers with 'block_hidden_layer_features' features for
                each layer
                processes a neighborhood at a time

                input size: [N x 2*D]
                hidden size: [N x 'block_hidden_layer_features']
                output size: hidden size

            - output: single dense layer that maps the pooled neighborhood to the size of a single feature

                input size: [N x 'block_hidden_layer_features']
                output size: [1 x D]

        Parameters
        ----------
        block_id: int
            number of the current block

        Returns
        -------
        block, output: tf.keras.Model, tf.keras.Model
            block keras model, output keras model
        """

        shape = 6 + (2 * self.feature_size + 2 if self.use_image_features else 2)
        block_input = tf.keras.Input(shape=shape, name=f'block_{block_id}_input')
        x = tf.keras.layers.BatchNormalization(axis=1, name=f'block_{block_id}_batch_norm_0_layer')(block_input)
        for i in range(self.num_layers_block):
            x = tf.keras.layers.Dense(self.block_hidden_layer_features, activation=self.activation,
                                      name=f'block_{block_id}layer{i}', use_bias=True)(x)
        normed_x = tf.keras.layers.BatchNormalization(axis=1, name=f'block_{block_id}_batch_norm_1_layer')(x)
        block = tf.keras.Model(inputs=block_input, outputs=normed_x, name=f'block_{block_id}_layers')

        final_input = tf.keras.Input(shape=self.block_hidden_layer_features, name=f'block_{block_id}_after_pool_input')
        output = tf.keras.Model(inputs=final_input,
                                outputs=tf.keras.layers.Dense(self.feature_size + 1, activation='linear',
                                                              name=f'block_{block_id}_after_pool_layer', use_bias=True)(
                                    final_input),
                                name=f'block_{block_id}_output')

        return block, output

    def _interpolate_features(self, features: tf.Tensor, rpn_boxes: tf.Tensor) -> tf.Tensor:
        """
        interpolate features for the given boxes

        Parameters
        ----------
        features: tensor (float32)
            features extracted from the image
        rpn_boxes: tensor (float32)
            shape: N x 4
            region proposal boxes

        Returns
        -------
        interpolated: tensor (float32)
            shape: N x s
            interpolated feature for each box
        """
        # calculate interpolated features
        interpolated = roialign(features, rpn_boxes, self.field,
                                pool=self.roialign_pool, tiles=self.roialign_tiles)
        interpolated = tf.reduce_mean(interpolated, axis=1)
        interpolated = tf.reduce_mean(interpolated, axis=1)

        interpolated = tf.reshape(interpolated, [tf.shape(interpolated)[0], -1])

        return interpolated

    def extract_boxes_n_scores(self, norm: tf.Tensor):
        """
        extracts rpn_boxes and corresponding objectiveness scores with the faster r-cnn network

        Parameters
        ----------
        norm: tensor (float32)
            normalized image

        - N: number of predictions
        - d: feature dimensions

        Returns
        -------
        features: tensor (float32)
            shape: N x d
            features extracted from the image for each box
        rpn_boxes: tensor (float32)
            shape: N x 4
            each prediction in box form
        scores: tensor (float32)
            shape: N x 1
            objectiveness score for each prediction
        """
        # extract features and rpn boxes from image
        features = self.backbone(norm, training=False)
        outputs = self.rpnetwork(features, training=False)

        rpn_reg = map_outputs(outputs[1], self.anchors, self.anchor_px, self.field)
        rpn_boxes = unparameterize(rpn_reg, self.anchors)

        # get objectiveness scores
        rpn_obj = tf.nn.softmax(map_outputs(outputs[0], self.anchors, self.anchor_px, self.field))
        scores = rpn_obj[:, 1] / (tf.reduce_sum(rpn_obj, axis=1))

        # filter out negative predictions
        condition = tf.where(tf.greater(scores, self.initial_prediction_threshold))
        rpn_boxes = tf.gather_nd(rpn_boxes, condition)
        scores = tf.expand_dims(tf.gather_nd(scores, condition), axis=1)

        return features, rpn_boxes, scores

    def call(self, inputs, training=None, mask=None):
        self.train_step(inputs)

    def test_step(self, data):
        norm, boxes, sample_weight = extract_data(data)

        features, rpn_boxes, scores = self.extract_boxes_n_scores(norm)

        compressed_features = self.compression_net(features, training=False)

        interpolated = self._interpolate_features(compressed_features, rpn_boxes)
        interpolated = tf.concat([scores, interpolated], axis=1)
        # run network
        nms_output = self.net((interpolated, rpn_boxes), training=True)

        self._cal_update_performance_stats(boxes, rpn_boxes, nms_output)

        metrics = {m.name: m.result() for m in self.standard}
        return metrics

    def _cal_update_performance_stats(self, boxes, rpn_boxes, nms_output):
        tp, tn, fp, fn = lnms_metrics(boxes, rpn_boxes, nms_output)
        self.standard[3].update_state(tp/tf.shape(rpn_boxes)[0])
        self.standard[4].update_state(fp/tf.shape(rpn_boxes)[0])
        self.standard[5].update_state(tn/tf.shape(rpn_boxes)[0])
        self.standard[6].update_state(fn/tf.shape(rpn_boxes)[0])

    def train_step(self, data):
        norm, boxes, sample_weight = extract_data(data)

        # extract features and rpn boxes from image
        features, rpn_boxes, scores = self.extract_boxes_n_scores(norm)

        if self.distributed_training:
            self._train_function(features, boxes, rpn_boxes, scores)
        else:
            loss = tf.cond(tf.shape(rpn_boxes)[0] > 0, lambda: self._train_function(features, boxes, rpn_boxes, scores),
                           lambda: 0.0)

        # save loss and metrics
        losses = {'lnms_loss': loss}
        metrics = {m.name: m.result() for m in self.standard}

        return {**losses, **metrics}

    def _train_function(self, features, boxes, rpn_boxes, scores):

        if not self.compressed_gradient:
            compressed_features = self.compression_net(features, training=False)

            # calculate interpolated features
            interpolated = self._interpolate_features(compressed_features, rpn_boxes)
            interpolated = tf.concat([scores, interpolated], axis=1)

        # training step
        with tf.GradientTape(persistent=True) as tape:

            if self.compressed_gradient:
                compressed_features = self.compression_net(features, training=True)

                # calculate interpolated features
                interpolated = self._interpolate_features(compressed_features, rpn_boxes)
                interpolated = tf.concat([scores, interpolated], axis=1)

            # run network
            nms_output = self.net((interpolated, rpn_boxes), training=True)

            # calculate loss
            if self.loss_type == 'dummy':
                loss = tf.reduce_sum(self.loss_object(nms_output, tf.ones(tf.shape(nms_output))))
            elif self.loss_type == 'xor':
                clusters = cluster_assignment(boxes, rpn_boxes)
                loss, labels = xor_loss(nms_output, clusters)
                # loss, labels = self._cal_xor_loss(nms_output, cluster_assignment)
            elif self.loss_type == 'clustering':
                clusters = cluster_assignment(boxes, rpn_boxes)
                loss, labels = clustering_loss(nms_output, clusters, self.loss_object, self.positive_weight,
                                               self.standard, self.weighted_loss, self.neg_pos_loss)
            elif self.loss_type == 'paper':
                loss, labels = paper_loss(boxes, rpn_boxes, nms_output, self.loss_object, self.positive_weight,
                                          self.standard, self.weighted_loss, self.neg_pos_loss)
            else:
                loss, labels = normal_loss(self.loss_object, boxes, rpn_boxes, nms_output, self.positive_weight,
                                           self.standard, neg_pos_loss=True)

        gradients = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

        if self.use_image_features and self.compressed_gradient:
            gradients = tape.gradient(loss, self.compression_net)
            self.optimizer.apply_gradients(zip(gradients, self.compression_net.trainable_weights))

        self.standard[0].update_state(loss + 1e-8)

        if self.calculate_train_metrics:
            self._cal_update_performance_stats(boxes, rpn_boxes, nms_output)

        return loss
