from abc import ABC
from typing import Tuple, List
import tensorflow as tf

from histomics_detect.boxes.neighborhood import all_neighborhoods_additional_info


class BlockModel(tf.keras.Model, ABC):
    def __init__(self, blocks: List[Tuple[tf.keras.Model, tf.keras.Model]], final_layers: tf.keras.Model,
                 threshold: float = 0.5, train_tile: float = 224, use_image_features: bool = True,
                 use_distance: bool = False, original_lnms: bool = True):
        """
        Learning-NMS block model

        Parameters
        ----------
        blocks: List[Tuple[tf.keras.Model, tf.keras.Model]]
            list of blocks each consists of two keras model
            - block: layers, which process the neighborhoods representation
            - output: layers, which process a neighborhood back into a single representation
        final_layers: tf.keras.Model
            layers that compress the final prediction representations into the final output score
        threshold: float
            threshold for two predictions to belong in the same neighborhood
        train_tile: float
            size of the image tile used for training
        use_image_features: bool
            true use image features when creating prediction representation
        use_distance: bool
            assemble neighborhood with distance instead of iou
        original_lnms: bool
            reuse the learned representation for next block
        """
        super(BlockModel, self).__init__(name='block_model')
        self.blocks = blocks
        self.final_layers = final_layers

        self.threshold = threshold
        self.train_tile = train_tile
        self.use_image_features = use_image_features
        self.use_distance = use_distance
        self.original_lnms = original_lnms

    def call(self, x: Tuple[tf.Tensor, tf.Tensor], training: bool = False, mask=None):
        """
        executes a forward pass of the block model for the given data 'x'

        S: size of neighborhood
        N: number of predictions
        D: size of a single prediction

        Parameters
        ----------
        x: (tensor, tensor) (float32)
            (interpolated, rpn_boxes)
            interpolated: interpolated representation of each rpn_box + the corresponding objectiveness score
                        shape (N, ...+1)
            rpn_boxes: boxes
                       shape (N, 4)
        training: bool
            true while training
            false during inference
        mask:
            not used

        Returns
        -------
        nms_output: tensor (float32)
            an updated score (objectiveness) for each rpn_box

        """

        interpolated, rpn_boxes_positive = x

        # initialize loop paramters
        num_predictions = tf.shape(rpn_boxes_positive)[0]
        prediction_ids = tf.range(0, num_predictions)

        neighborhood_sizes, neighborhoods_add_info, neighborhoods_indexes, self_indexes = \
            all_neighborhoods_additional_info(rpn_boxes_positive, prediction_ids, self.train_tile, self.threshold,
                                              self.use_distance)

        if not self.use_image_features and not self.original_lnms:
            empty_features = tf.zeros((tf.shape(interpolated)[0], tf.shape(interpolated)[1] - 1))
            interpolated = tf.concat([tf.expand_dims(interpolated[:, 0], axis=1), empty_features], axis=1)

        for block, output in self.blocks:
            # run network on block

            # assemble neighborhood
            if self.use_image_features or self.original_lnms:
                # assemble the neighboring prediction representation for each neighborhood
                neighborhood = tf.reshape(tf.gather(interpolated, neighborhoods_indexes),
                                          [-1, tf.shape(interpolated)[1]])
                # assemble the tiled prediction representation of the main prediction of each neighborhood
                main_predictions = tf.reshape(tf.gather(interpolated, self_indexes), [-1, tf.shape(interpolated)[1]])
                neighborhoods = tf.concat([neighborhood, main_predictions, neighborhoods_add_info], axis=1)
            else:
                neighborhood = tf.reshape(tf.gather(interpolated[:, 0], neighborhoods_indexes),
                                          [-1, 1])
                main_predictions = tf.reshape(tf.gather(interpolated[:, 0], self_indexes),
                                              [-1, 1])
                empty_features = tf.zeros((tf.shape(neighborhoods_add_info)[0], tf.shape(interpolated)[1] - 1))
                neighborhoods = tf.concat(
                    [neighborhood, empty_features, main_predictions, empty_features, neighborhoods_add_info], axis=1)
            num_predictions = tf.size(neighborhood_sizes)

            # run block
            processed_neighborhoods = block(neighborhoods, training=training)

            # pool neighborhoods
            pooled_predictions = tf.TensorArray(dtype=tf.float32, size=num_predictions)

            counter = tf.constant(0, dtype=tf.int32)
            start_index = tf.constant(0)

            # pool each neighborhood to one prediction representation
            for size in neighborhood_sizes:
                neighborhood_slice = processed_neighborhoods[start_index:start_index + size]
                pooled_predictions = pooled_predictions.write(counter, tf.math.reduce_max(neighborhood_slice, axis=0))

                counter += 1
                start_index += size

            pooled_predictions = pooled_predictions.stack()

            # run final block layers
            processed_predictions = output(pooled_predictions, training=training)
            block_output = tf.cast(processed_predictions, tf.float32)

            # skip connection
            interpolated = (interpolated + block_output)
        return self.final_layers(interpolated, training=training)
