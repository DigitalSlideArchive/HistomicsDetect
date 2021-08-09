import tensorflow as tf


def _generate_block(_id: int, num_layers: int, input_features: int, hidden_features: int,
                    kernel_size: int = 5, activation: str = 'relu') -> tf.keras.Model:
    input_layer = tf.keras.Input(shape=[None, None, input_features], name=f"block_{_id}_input")
    current_layer = input_layer

    for i in range(num_layers - 1):
        new_layer = tf.keras.layers.Conv2D(hidden_features, kernel_size, activation=activation, padding='same',
                                           name=f"block_{_id}_hidden_layer_{i}")(current_layer)

        current_layer = new_layer
    final_layer = tf.keras.layers.Conv2D(input_features, 3, activation=activation, padding='same',
                                         name=f"block_{_id}_final_layer")(current_layer)

    return tf.keras.Model(inputs=input_layer, outputs=final_layer, name=f"block_model_{_id}")


def LNMSCNN(input_features: int, hidden_features: int, num_layers: int = 4, num_blocks: int = 2,
            skip_connection: bool = True, activation: str = 'relu', final_activation: str = 'sigmoid',
            kernel_size: int = 5) -> tf.keras.Model:
    """
    Similar to the LNMS Model but instead of doing the expensive neighborhood assembly this model uses the anchors
    in their grid shape to approximate the neighborhoods with ordinary cnn filters

    first the anchors are loaded and some additional information is added to each anchor:
    - the box representation
    - the objectiveness score

    then the novel representation is passed through multiple blocks of bottleneck layers

    the final output is an updated objectiveness score for each anchor

    Network Architecture:

        Input
      |      |
    Block1   |
       |    |
        add
    |      |
    Block2   |
       |    |
        add
    |      |
    Block3   |
       |    |
        add
    ...
    |      |
    BlockN   |
       |    |
        add
         |
      output

    Parameters
    ----------
    input_features: int
        the feature size of an anchor before adding the additional features
    hidden_features: int
        the size of the compression in each block
    num_layers: int
        the number of layers per block
    num_blocks: int
        the number of blocks
    skip_connection: bool
        if true adds skip connections for each block
    activation: str
        between layer activation
    final_activation: str
        final output activation
    kernel_size: int
        the size of the kernels

    Returns
    -------
    LNMS_CNN_model: tf.keras.Model
        the model with the aforementioned architecture
    """
    input_features = input_features + 4 + 1

    blocks = [_generate_block(_id, num_layers, input_features, hidden_features, kernel_size, activation)
              for _id in range(num_blocks)]

    input_layer = tf.keras.Input(shape=[None, None, input_features], name=f"LNMS_CNN_model_input")
    current_layer = input_layer

    for block in blocks:
        if skip_connection:
            current_layer = block(current_layer) + current_layer
        else:
            current_layer = block(current_layer)

    final_layer = tf.keras.layers.Conv2D(1, kernel_size, activation=final_activation, padding='same',
                                         name=f"LNMS_CNN_final_layer")(current_layer)

    return tf.keras.Model(inputs=input_layer, outputs=final_layer, name=f"LNMS_CNN_model")
