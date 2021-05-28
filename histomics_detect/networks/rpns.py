from histomics_detect.networks.backbones import residual_backbone
from tensorflow import tf


def rpn(resnet, n_anchors=3, stride=1, blocks=4, kernels=[3], dimensions=[256],
        activations=['relu']):
    #builds rpn network that consumes features and performs 1x1 objectness, 
    #regression convolutions.

    #build backbone network
    backbone = residual_backbone(resnet, stride, blocks)
    
    #create input layer
    rpn_input = tf.keras.Input(shape=(None, None, backbone.output.shape[-1]))

    #build region-proposal convolutional layers
    for i, (kernel, dimension, activation) in \
        enumerate(zip(kernels, dimensions, activations)):
        if i == 0:
            layer_input = rpn_input
        else:
            layer_input = x  
        x = tf.keras.layers.Conv2D(dimension, kernel, padding='same',
                                   activation=activation)(layer_input)

    #build regression and objectness classification layers
    regression = tf.keras.layers.Conv2D(4*n_anchors, 1, padding='same',
                                        activation='linear')(x)
    objectness = tf.keras.layers.Conv2D(2*n_anchors, 1, padding='same',
                                        activation='linear')(x)

    #create rpn model
    rpn = tf.keras.Model(inputs=rpn_input, 
                         outputs=[objectness, regression])

    return rpn, backbone
