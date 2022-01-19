from abc import ABC

import tensorflow as tf


class CompressionNetwork(tf.keras.Model, ABC):

    def __init__(self, feature_size: int, anchor_size: int, backbone: tf.keras.Model, *args, **kwargs):
        """
        Network for training the compression layers of for the lnms network

        Parameters
        ----------
        feature_size: int
            compression size of the feature
        anchor_size: int
        backbone: tf.keras.Model
            feature extraction model
        args
        kwargs
        """
        super(CompressionNetwork, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.anchor_size = anchor_size
        self.activation = 'sigmoid'
        self.compression_layers = self._initialize_compression_layers()
        self.decompression_layers = self._initialize_decompression_layers()
        self.backbone = backbone

    def _initialize_decompression_layers(self) -> tf.keras.Model:
        """
        builds the initial feature decompression keras model

        Returns
        -------
        compression_model: tf.keras.Model
            compression network that can compress image features
        """
        input_layer = tf.keras.Input(shape=[None, None, self.feature_size], name="compressed_input")
        layer_1 = tf.keras.layers.Conv2DTranspose(self.feature_size, 1, activation=self.activation, padding='same',
                                                  name="decompression_layer_1")(input_layer)
        layer_2 = tf.keras.layers.Conv2DTranspose(self.anchor_size, 1, activation=self.activation, padding='same',
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

    @tf.function
    def train_step(self, data):
        (img, boxes, name) = data
        # normalize image
        norm = tf.keras.applications.resnet.preprocess_input(tf.cast(img, tf.float32))
        # expand dimensions
        norm = tf.expand_dims(norm, axis=0)

        features = self.backbone(norm)

        with tf.GradientTape(persistent=True) as tape:
            network_output = self(features)
            loss = self.loss(network_output, features)

        gradients = tape.gradient(loss, self.compression_layers.trainable_weights +
                                  self.decompression_layers.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.compression_layers.trainable_weights +
                                           self.decompression_layers.trainable_weights))

        # save loss and metrics
        losses = {'loss': loss}
        metrics = {}

        return {**losses, **metrics}
