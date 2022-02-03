from histomics_detect.networks.transfer_layers import transfer_layers
import tensorflow as tf


def pretrained(name):
    """
    Loads backbone keras model from tf.keras.applications.
    
    Parameters
    ----------
    name : string
        Name of pre-trained network to load. Case insensitive.

    Returns
    -------
    model : tf.keras.Model
        A pre-trained tf.keras.Model network.
    preprocessor : function
        Function used for transforming model inputs.
    """
    
    if str.lower(name) == 'resnet50':
        model = tf.keras.applications.resnet.ResNet50(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet.preprocess_input
    elif str.lower(name) == 'resnet101':
        model = tf.keras.applications.resnet.ResNet101(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet.preprocess_input
    elif str.lower(name) == 'resnet152':
        model = tf.keras.applications.resnet.ResNet152(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet.preprocess_input
    elif str.lower(name) == 'resnet50v2':
        model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet_v2.preprocess_input
    elif str.lower(name) == 'resnet101v2':
        model = tf.keras.applications.resnet_v2.ResNet101V2(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet_v2.preprocess_input
    elif str.lower(name) == 'resnet152v2':
        model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(None, None, 3), pooling=None)
        preprocessor = tf.keras.applications.resnet_v2.preprocess_input
    else:
        raise ValueError("Network name not recognized")
    
    return model, preprocessor


def residual(model, preprocessor, blocks, stride=None):
    """
    Creates a feature extraction backbone from a tf.keras.applications resnet model.
    Allows user to select the number of residual blocks to keep and to set the convolution
    stride. Optionally merges the model with a preprocessor function.
        
    Parameters
    ----------
    model : tf.keras.Model
        A resnet keras Model obtained from tf.keras.applications. Can be loaded using 'pretrained()'.
    preprocesor : function
        Function used to transform input images. Can be loaded using 'pretrained()'.
    blocks : int
        The desired number of residual blocks to keep from the input model. Terminal blocks
        > blocks are truncated from the output model.
    stride : int
        The desired stride for the first convolution. Default value None does not alter stride.
    
    Returns
    -------
    backbone : tf.keras.Model
        A keras model with desired number of blocks, stride, and preprocessing capability.
    """

    #sweep layers to identify activation layer of the block 'blocks'
    add = False
    block = 0
    for (i,layer) in enumerate(model.layers):

        #change state if add layer encountered and continue with next layer
        if type(layer) == tf.keras.layers.Add:
            add = True
            continue

        #look for activation or batch norm layer following add layer
        if add:
            if (type(layer) == tf.keras.layers.Activation): #resnet v1 - keep activation
                block = block + 1
                add = False
                if block == blocks:
                    terminal = i
                    break
            elif (type(layer) == tf.keras.layers.BatchNormalization): #resnet v2 - keep add
                block = block + 1
                add = False
                if block == blocks:
                    terminal = i-1
                    break
            else:
                add = False

    #replace input layer
    input = tf.keras.Input(shape=(None, None, 3))

    #duplicate zero padding layer (layer 1)
    padding = tf.keras.layers.serialize(model.layers[1])
    padding = tf.keras.layers.deserialize(padding)

    #modify stride of first convolutional layer (layer 2)
    weights = model.layers[2].get_weights()
    name = model.layers[2].name
    if stride is None:
        stride = model.layers[2].strides
    conv = tf.keras.layers.serialize(model.layers[2])
    conv['config']['strides'] = (stride, stride)
    conv = tf.keras.layers.deserialize(conv)
    
    #apply preprocessor
    normalized = preprocessor(input)

    #compose model
    features = transfer_layers(model.layers[3:terminal+1],
                                   'b', conv(padding(normalized)))   
    backbone = tf.keras.Model(inputs=input, outputs=features)
    
    #set weights for first convolution
    for (i, layer) in enumerate(backbone.layers):
        if layer.name == name:
            backbone.layers[i].set_weights(weights)

    return backbone
