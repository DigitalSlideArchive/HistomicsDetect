def fast_rcnn(backbone, units=[4096, 4096], activations=['relu', 'relu'], pool=2, tiles=3):
    #generates fast RCNN model used in second regression that follows region proposal network.
    
    #create input layer
    fastrcnn_input = tf.keras.Input(shape=(pool*tiles, pool*tiles, backbone.output.shape[-1]))

    #pooling over each roialign tile
    pooled = tf.keras.layers.MaxPool2D(pool_size=(pool, pool), padding='valid')(fastrcnn_input)

    #stack the pooled feature vectors from each tile into a single feature vector
    pooled = tf.reshape(pooled, (tf.shape(pooled)[0], tiles*tiles*backbone.output.shape[-1]))

    #fully connected layers
    for i, (unit, activation) in enumerate(zip(units, activations)):
        if i == 0:
            layer_input = pooled
        else:
            layer_input = x
        x = tf.keras.layers.Dense(unit, activation=activation)(layer_input)

    #final layer to generate parameterized regression prediction
    regression_align = tf.keras.layers.Dense(4, activation='linear')(x)
    
    #create model
    fastrcnn = tf.keras.Model(inputs=fastrcnn_input,
                              outputs=[regression_align])
    
    return fastrcnn
