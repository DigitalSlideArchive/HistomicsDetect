from histomics_detect.networks.transfer_layers import transfer_layers
import tensorflow as tf


def pretrained(name, train_shape=(224, 224, 3)):
    """
    Loads backbone keras model from tf.keras.applications.
    
    Parameters
    ----------
    name : string
        Name of pre-trained network to load. One of 'resnet50', 'resnet101',
        or 'resnet152'. Case insensitive.
    train_shape : int
        Size of input images used for training. Network can be modified later to 
        alter input shape.
    Returns
    -------
    model : tf.keras.Model
        A pre-trained tf.keras.Model network.
    preprocessor : function
        Function used for transforming model inputs.
    """
    
    if str.lower(name) == 'resnet50':
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=train_shape, pooling=None)
        preprocessor = tf.keras.applications.resnet50.preprocess_input
    elif str.lower(name) == 'resnet101':
        model = tf.keras.applications.resnet.ResNet101(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=train_shape, pooling=None)
        preprocessor = tf.keras.applications.resnet.preprocess_input
    elif str.lower(name) == 'resnet152':
        model = tf.keras.applications.resnet.ResNet152(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=train_shape, pooling=None)
        preprocessor = tf.keras.applications.resnet.preprocess_input
    else:
        raise ValueError("Network name not recognized")
    
    return model, preprocessor


def residual(model, blocks, stride=None, preprocessor=tf.keras.applications.resnet.preprocess_input):
    """Creates a feature extraction backbone from a tf.keras.applications resnet model.
    Allows user to select the number of residual blocks to keep and to set the convolution
    stride. Optionally merges the model with a preprocessor function.
        
    Parameters
    ----------
    model : tf.keras.Model
        A resnet keras Model obtained from tf.keras.applications.
    blocks : int
        The desired number of residual blocks to keep from the input model. Terminal blocks
        > blocks are truncated from the output model.
    stride : int
        The desired stride for the first convolution. Default value None does not alter stride.
    preprocesor : function
        Function used to transform input images. Default value
        tf.keras.applications.resnet.preprocess_input.
    
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
    input = tf.keras.Input(shape=(None, None, 3))

    #duplicate zero padding layer (layer 1)
    padding = tf.keras.layers.serialize(model.layers[1])
    padding = tf.keras.layers.deserialize(padding)

    #modify stride of first convolutional layer (layer 2)
    if stride is not None:
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
