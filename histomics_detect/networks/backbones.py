import tensorflow as tf


def residual_backbone(model, stride, blocks, shape=(None, None, 3)):
    #builds backbone feature extraction network for region-proposal network

    #sweep layers to identify activation layer of the block 'blocks'
    add = False
    block = 0
    for (i,layer) in enumerate(model.layers):

        #change state if add layer encountered and continue with next layer
        if type(layer) == tf.keras.layers.Add:
            add = True
            continue

        #look for activation layer following add layer
        if add:
            if type(layer) == tf.keras.layers.Activation:
                block = block+1
                if block == blocks:
                    terminal = i
                    break
            else:
                add = False

    #replace input layer
    input = tf.keras.Input(shape=shape)

    #duplicate zero padding layer (layer 1)
    padding = tf.keras.layers.serialize(model.layers[1])
    padding = tf.keras.layers.deserialize(padding)

    #modify stride of first convolutional layer (layer 2)
    conv = tf.keras.layers.serialize(model.layers[2])
    weights = model.layers[2].get_weights()
    conv['config']['strides'] = (stride, stride)
    conv = tf.keras.layers.deserialize(conv)

    #re-build additional feature extraction layers
    features = transfer_layers(model.layers[3:terminal+1],
                               'b', conv(padding(input)))

    #set weights for first convolution
    backbone = tf.keras.Model(inputs=input, outputs=features)
    backbone.layers[2].set_weights(weights)

    return backbone
