import tensorflow as tf


def transfer_layers(layers, prefix, input):
    #This function copies layers from the input models into a single
    #model with appropriate name scopes to help separate variables by
    #shared, predictor, and discriminator sub-networks. Supports feed 
    #forward networks with skip layers and branching architectures. 
    #layers - a list of tf.keras.layers
    #prefix - string used for scoping variable and layer names
    #input - optional model input from other subnetwork, if 'None'
    #        model.layers[0] must be tf.keras.layers.Input.

    #check input layer type and discard input layer if present
    if layers[0].__class__.__name__ == 'InputLayer':
        layers = layers[1:]

    #history of layer output names and their indices in model.layers
    history = {}

    #list of transferred model layer output, used to feed later layers
    outputs = []

    #iterate through layers, transferring configurations and weights
    for i, layer in enumerate(layers):
        
        #layer output name is key, index in 'output' is value
        history[layer.output.name] = i

        #generate new layer w/ configuration w/o weights
        serialized = tf.keras.layers.serialize(layer)
        serialized['config']['name'] = prefix + '/' + serialized['config']['name']
        transfered = tf.keras.layers.deserialize(serialized)

        #set new layer inputs and outputs
        if i == 0:
            outputs.append(transfered(input))
        else:
            if type(layer.input) is list:
                inputs = [outputs[history[input.name]] for input in layer.input]
                outputs.append(transfered(inputs))
            else:
                inputs = outputs[history[layer.input.name]]
                outputs.append(transfered(inputs))

        #copy weights to new layer
        transfered.set_weights(layer.get_weights())

    return outputs[-1]
