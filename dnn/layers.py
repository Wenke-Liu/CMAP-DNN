import tensorflow as tf


def fc_bn_dropout(x,
                  scope="fc_bn_dropout",
                  size=None,
                  dropout=1.,
                  activation=tf.nn.elu,
                  training=True):
    
    assert size, "Must specify layer size (num nodes)"
    # use linear activation for pre-activation batch_normalization
    
    with tf.variable_scope(scope):
        
        fc = tf.contrib.layers.fully_connected(inputs=x,
                                               num_outputs=size,
                                               activation_fn=None)

        fc_bn = tf.contrib.layers.batch_norm(inputs=fc,
                                             is_training=training,
                                             activation_fn=activation)
        
        fc_bn_drop = tf.contrib.layers.dropout(inputs=fc_bn,
                                               keep_prob=dropout,
                                               is_training=training)

    return fc_bn_drop


def fc_dropout(x,
               scope="fc_dropout",
               size=None,
               dropout=1.,
               activation=tf.nn.elu,
               training=True):
    
    assert size, "Must specify layer size (num nodes)"
    # use linear activation for pre-activation batch_normalization
    
    with tf.variable_scope(scope):
        
        fc = tf.contrib.layers.fully_connected(inputs=x,
                                               num_outputs=size,
                                               activation_fn=activation)

        fc_drop = tf.contrib.layers.dropout(inputs=fc,
                                            keep_prob=dropout,
                                            is_training=training)

    return fc_drop


