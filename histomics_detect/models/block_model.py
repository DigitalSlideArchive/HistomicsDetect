from abc import ABC
from typing import Tuple, List
import tensorflow as tf

from histomics_detect.boxes.neighborhood import assemble_single_neighborhood, all_neighborhoods_additional_info


class BlockModel(tf.keras.Model, ABC):
    def __init__(self, blocks: List[Tuple[tf.keras.Model, tf.keras.Model]], final_layers: tf.keras.Model,
                 threshold: float = 0.5, train_tile: float = 224, use_image_features: bool = True):
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
            threshold for two predictions to belong in the same neighbohood
        train_tile: float
            size of the image tile used for training
        use_image_features: bool
            true use image features when creating prediction representation
        """
        super(BlockModel, self).__init__(name='block_model')
        self.blocks = blocks
        self.final_layers = final_layers

        self.threshold = threshold
        self.train_tile = train_tile
        self.use_image_features = use_image_features

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
        prediction_ids = tf.range(0, num_predictions) #TODO figure out why error here with lymphoma

        neighborhood_sizes, neighborhoods_add_info, neighborhoods_indeces = all_neighborhoods_additional_info(
            rpn_boxes_positive, prediction_ids, self.train_tile, self.threshold)

        for block, output in self.blocks:
            # run network on block
            neighborhoods = assemble_single_neighborhood(prediction_ids[0], interpolated,
                                                         neighborhoods_indeces[0:neighborhood_sizes[0]],
                                                         neighborhoods_add_info[0:neighborhood_sizes[0]],
                                                         self.use_image_features)
            start_index = neighborhood_sizes[0]
            counter = 1

            # assemble neighborhoods
            for x in prediction_ids[1:]:
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(neighborhoods, tf.TensorShape([None, None]))])
                end_index = start_index + neighborhood_sizes[counter]
                new_neighborhood = assemble_single_neighborhood(x, interpolated,
                                                                neighborhoods_indeces[start_index:end_index],
                                                                neighborhoods_add_info[start_index:end_index],
                                                                self.use_image_features)

                start_index = end_index
                counter += 1
                neighborhoods = tf.concat([neighborhoods, new_neighborhood], axis=0)
            num_predictions = tf.size(neighborhood_sizes)

            # run block
            processed_neighborhoods = block(neighborhoods, training=training)

            # pool neighborhoods
            pooled_predictions = tf.TensorArray(dtype=tf.float32, size=num_predictions)

            counter = tf.constant(0, dtype=tf.int32)
            start_index = tf.constant(0)

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
