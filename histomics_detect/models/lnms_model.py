from abc import ABC
from typing import Tuple
import tensorflow as tf

from histomics_detect.networks.field_size import field_size
from histomics_detect.anchors.create import create_anchors
from histomics_detect.models.block_model import BlockModel
from histomics_detect.roialign.roialign import roialign
from histomics_detect.boxes.transforms import unparameterize, parameterize, clip_boxes
from histomics_detect.models.lnms_loss import normal_loss, clustering_loss, paper_loss, xor_loss, normal_clustering_loss
from histomics_detect.boxes.match import cluster_assignment
from histomics_detect.boxes.cross_boxes import cross_from_boxes


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

        self.compression_net = compression_network

        if not self.data_only:
            self.net = BlockModel([self._initialize_block(i) for i in range(self.num_blocks)],
                                  self._initialize_final_output(), threshold=self.threshold,
                                  train_tile=self.train_tile, use_image_features=self.use_image_features,
                                  use_distance=self.use_distance, original_lnms=self.original_lnms)

            if self.use_reg:
                self.init_regression = self._initialize_init_regression()

            # define metrics
            loss_col = tf.keras.metrics.Mean(name='loss')
            loss_pos = tf.keras.metrics.Mean(name='pos_loss')
            loss_neg = tf.keras.metrics.Mean(name='neg_loss')
            self.standard = [loss_col, loss_pos, loss_neg]

    def _initialize_init_regression(self) -> tf.keras.Model:
        feature_size_multiplier = 2 if self.combine_box_and_cross else 1
        shape = (feature_size_multiplier * self.feature_size + 5)

        init_input = tf.keras.Input(shape=shape, name="init_regression_input")

        layer1 = tf.keras.layers.Dense(int(shape / 8), activation=self.activation, use_bias=True,
                                       name=f'init_regression_layer_1')(init_input)
        layer2 = tf.keras.layers.Dense(int(shape / 8), activation=self.activation, use_bias=True,
                                       name=f'init_regression_layer_2')(layer1)
        layer3 = tf.keras.layers.Dense(4, activation='linear', use_bias=True,
                                       name=f'init_regression_layer_3')(layer2)

        return tf.keras.Model(inputs=init_input, outputs=layer3, name="init_regression_network")

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
        feature_size_multiplier = 2 if self.combine_box_and_cross and self.use_image_features else 1

        final_input = tf.keras.Input(shape=self.feature_size * feature_size_multiplier + 1,
                                     name="final_after_block_layers_input")
        x = final_input
        for i in range(self.num_hidden_layers):
            x = tf.keras.layers.Dense(self.final_hidden_layer_features, activation=self.activation, use_bias=True,
                                      name=f'final_after_block_layer_{i}')(x)
        output_size = self.add_regression_param * 2 + 1 + int(self.objectness_format)
        x = tf.keras.layers.Dense(output_size, activation=self.final_activation, name="output_score_layer",
                                  use_bias=True)(x)

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
        feature_size_multiplier = 2 if self.combine_box_and_cross and self.use_image_features else 1

        shape = 6 + (feature_size_multiplier * 2 * self.feature_size + 2)
        block_input = tf.keras.Input(shape=shape, name=f'block_{block_id}_input')
        x = tf.keras.layers.BatchNormalization(axis=1, name=f'block_{block_id}_batch_norm_0_layer')(block_input)
        for i in range(self.num_layers_block):
            x = tf.keras.layers.Dense(self.block_hidden_layer_features, activation=self.activation,
                                      name=f'block_{block_id}layer{i}', use_bias=True)(x)
        normed_x = tf.keras.layers.BatchNormalization(axis=1, name=f'block_{block_id}_batch_norm_1_layer')(x)
        block = tf.keras.Model(inputs=block_input, outputs=normed_x, name=f'block_{block_id}_layers')

        final_input = tf.keras.Input(shape=self.block_hidden_layer_features, name=f'block_{block_id}_after_pool_input')
        output = tf.keras.Model(inputs=final_input,
                                outputs=tf.keras.layers.Dense(self.feature_size * feature_size_multiplier + 1,
                                                              activation='linear',
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
        reduction_func = tf.reduce_mean if self.reduce_mean else tf.reduce_max

        # calculate interpolated features
        if self.cross_boxes:
            cross_boxes = cross_from_boxes(rpn_boxes, self.cross_scale, image_width=self.width,
                                           image_height=self.height)
            interpolated = roialign(features, tf.reshape(cross_boxes, (-1, 4)), self.field,
                                    pool=self.roialign_pool, tiles=self.roialign_tiles)
            interpolated = reduction_func(interpolated, axis=1)
            interpolated = reduction_func(interpolated, axis=1)
            interpolated = tf.reshape(interpolated, (tf.shape(rpn_boxes)[0], 2, -1))
            interpolated = reduction_func(interpolated, axis=1)

            if self.combine_box_and_cross:
                interpolated_box = roialign(features, rpn_boxes, self.field,
                                            pool=self.roialign_pool, tiles=self.roialign_tiles)
                interpolated_box = reduction_func(interpolated_box, axis=1)
                interpolated_box = reduction_func(interpolated_box, axis=1)
                interpolated = tf.concat([interpolated, interpolated_box], axis=1)
            # TODO maybe concatenate instead of mean for cross
        else:
            if self.expand_boxes:
                rpn_boxes = tf.concat(
                    [rpn_boxes[:, :2] - self.box_expand_value, rpn_boxes[:, 2:] + self.box_expand_value * 2])
                rpn_boxes = clip_boxes(rpn_boxes, self.width, self.height)
            interpolated = roialign(features, rpn_boxes, self.field,
                                    pool=self.roialign_pool, tiles=self.roialign_tiles)
            interpolated = reduction_func(interpolated, axis=1)
            interpolated = reduction_func(interpolated, axis=1)

        interpolated = tf.reshape(interpolated, [tf.shape(interpolated)[0], -1])

        return interpolated

    def call(self, data, training=None, mask=None):
        features, boxes, rpn_boxes, scores = data

        compressed_features = self.compression_net(features, training=False)

        interpolated = self._interpolate_features(compressed_features, rpn_boxes)
        interpolated = tf.concat([scores, interpolated], axis=1)

        # run network
        nms_output = self.net((interpolated, rpn_boxes), training=True)
        return features, rpn_boxes, scores, nms_output

    def train_step(self, data):
        """
        training function

        Parameters
        ----------
        data: Tuple[features, boxes, rpn_boxes, scores]
            - features: tensor (float32)
                shape: W x H x M
                features extracted from the image
            - boxes: tensor (float32)
                shape: N x 4
                ground truth boxes
            - rpn_boxes: tensor (float32)
                shape: G x 4
                region proposal boxes
            - scores:
                shape: G x 1
                objectness scores

        Returns
        -------
        losses, metrics
        """
        features, boxes, rpn_boxes, scores = data

        if not self.compressed_gradient:
            compressed_features = self.compression_net(features, training=False)

        if not self.interpolated_gradient:
            # calculate interpolated features
            interpolated = self._interpolate_features(compressed_features, rpn_boxes)
            interpolated_score = tf.concat([scores, interpolated], axis=1)

        if self.use_reg:
            # TODO does not work fix or remove
            rpn_reg_label = parameterize(rpn_boxes, boxes)

        # training step
        with tf.GradientTape(persistent=True) as tape:

            if self.compressed_gradient:
                compressed_features = self.compression_net(features, training=True)

            if self.interpolated_gradient:
                # calculate interpolated features
                interpolated = self._interpolate_features(compressed_features, rpn_boxes)
                interpolated_score = tf.concat([scores, interpolated], axis=1)

            if self.use_reg:
                init_regression_input = tf.concat([interpolated_score, rpn_boxes], axis=1)
                init_regression_output = self.init_regression(init_regression_input)

                rpn_boxes = unparameterize(init_regression_output, rpn_boxes)
                rpn_reg_loss = self.loss[1](rpn_reg_label, init_regression_output)

            if self.manipulate_rpn:
                rpn_boxes, scores = self.maipulate_rpn_func(self, rpn_boxes, boxes, interpolated, scores)
                interpolated_score = tf.concat([scores, interpolated], axis=1)

            # run network
            nms_output = self.net((interpolated_score, rpn_boxes), training=True)

            loss, labels = self._calculate_loss(nms_output, boxes, rpn_boxes)

        gradients = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

        if self.use_reg:
            gradients = tape.gradient(rpn_reg_loss, self.init_regression.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.init_regression.trainable_weights))

        if self.use_image_features and self.compressed_gradient:
            gradients = tape.gradient(loss, self.compression_net.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.compression_net.trainable_weights))

        self.standard[0].update_state(loss + 1e-8)

        if self.calculate_train_metrics:
            self._cal_update_performance_stats(boxes, rpn_boxes, nms_output)

        # save loss and metrics
        losses = {'lnms_loss': loss}
        metrics = {m.name: m.result() for m in self.standard}

        return {**losses, **metrics}

    def _calculate_loss(self, nms_output, boxes, rpn_boxes) -> Tuple[float, tf.Tensor]:
        """
        call the loss function of the loss specified in 'loss_type'

        S: size of neighborhood
        N: number of predictions
        D: size of a single prediction
        G: number of ground truth boxes

        Parameters
        ----------
        nms_output: tensor (float32)
            objectiveness scores corresponding to the predicted boxes after lnms processing
            shape: N x 1
        boxes tensor (float32)
            ground truth boxes
            shape: G x 4
        rpn_boxes: tensor (float32)
            shape: N x 4
            each prediction in box form

        Returns
        -------
        loss
        labels
        """
        # calculate loss
        if self.loss_type == 'dummy':
            scores = tf.expand_dims(nms_output[:, 0], axis=1)
            loss = tf.reduce_sum(self.loss_object(scores, tf.ones(tf.shape(nms_output))))
            labels = []
        elif self.loss_type == 'xor':
            scores = tf.expand_dims(nms_output[:, 0], axis=1)
            clusters = cluster_assignment(boxes, rpn_boxes)
            loss, labels = xor_loss(scores, clusters)
            # loss, labels = self._cal_xor_loss(nms_output, cluster_assignment)
        elif self.loss_type == 'clustering':
            clusters = cluster_assignment(boxes, rpn_boxes)
            loss, labels = clustering_loss(nms_output, clusters, self.loss_object, self.positive_weight,
                                           self.standard, boxes, rpn_boxes, self.weighted_loss, self.neg_pos_loss,
                                           self.add_regression_param)
        elif self.loss_type == 'paper':
            scores = tf.expand_dims(nms_output[:, 0], axis=1)
            loss, labels = paper_loss(boxes, rpn_boxes, scores, self.loss_object, self.positive_weight,
                                      self.standard, self.weighted_loss, self.neg_pos_loss, self.iou_threshold)
        elif self.loss_type == 'clustering_normal':
            clusters = cluster_assignment(boxes, rpn_boxes)
            loss, labels = normal_clustering_loss(nms_output, boxes, rpn_boxes, clusters, self.loss_object,
                                                  self.positive_weight, self.standard, self.weighted_loss,
                                                  self.neg_pos_loss, self.use_pos_neg_loss, self.norm_loss_weight,
                                                  self.add_regression_param, self.iou_threshold)
        elif self.loss_type == 'custom':
            loss, labels = self.custom_loss(self, nms_output, boxes, rpn_boxes)
        else:
            scores = tf.expand_dims(nms_output[:, 0], axis=1)
            loss, labels = normal_loss(self.loss_object, boxes, rpn_boxes, scores, self.positive_weight,
                                       self.standard, neg_pos_loss=True, min_iou=self.iou_threshold)

        return loss, labels
