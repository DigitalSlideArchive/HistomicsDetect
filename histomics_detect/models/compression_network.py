from abc import ABC

import tensorflow as tf


class CompressionNetwork(tf.keras.Model, ABC):

    def __init__(self, feature_size: int, anchor_size: int, *args, **kwargs):
        super(CompressionNetwork, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.anchor_size = anchor_size
        self.activation = 'sigmoid'
        self.compression_layers = self._initialize_compression_layers()
        self.decompression_layers = self._initialize_decompression_layers()

    def _initialize_decompression_layers(self) -> tf.keras.Model:
        """
                builds the initial feature decompression keras model

                Returns
                -------
                compression_model: tf.keras.Model
                    compression network that can compress image features
                """
        input_layer = tf.keras.Input(shape=[None, None, self.feature_size], name="compressed_input")
        layer_1 = tf.keras.layers.Conv2D(self.feature_size, 3, activation=self.activation, padding='same',
                                         name="decompression_layer_1")(input_layer)
        layer_2 = tf.keras.layers.Conv2D(self.anchor_size, 3, activation=self.activation, padding='same',
                                         name="decompression_layer_2")(layer_1)
        return tf.keras.Model(inputs=input_layer, outputs=layer_2, name="decompression_network")

    def _initialize_compression_layers(self) -> tf.keras.Model:
        """
        builds the initial feature compression keras model

        Returns
        -------
        compression_model: tf.keras.Model
            compression network that can compress image features
        """
        input_layer = tf.keras.Input(shape=[None, None, self.anchor_size], name="uncompressed_input")
        layer_1 = tf.keras.layers.Conv2D(self.feature_size, 3, activation=self.activation, padding='same',
                                         name="compression_layer_1")(input_layer)
        layer_2 = tf.keras.layers.Conv2D(self.feature_size, 3, activation=self.activation, padding='same',
                                         name="compression_layer_2")(layer_1)
        return tf.keras.Model(inputs=input_layer, outputs=layer_2, name="compression_network")

    def call(self, inputs, training=None, mask=None):
        return self.decompression_layers(self.compression_layers(inputs))

    def train_step(self, data):

        with tf.GradientTape(persistent=True) as tape:
            network_output = self(data)
            loss = self.loss(network_output, data)

        gradients = tape.gradient(loss, self.compression_layers)
        self.optimizer.apply_gradients(zip(gradients, self.compression_layers.trainable_weights))

        gradients = tape.gradient(loss, self.decompression_layers)
        self.optimizer.apply_gradients(zip(gradients, self.decompression_layers.trainable_weights))
