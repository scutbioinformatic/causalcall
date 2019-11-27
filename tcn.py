import tensorflow as tf
import math


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if layer_name not in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def batchnorm(inp, training, decay=0.99, epsilon=1e-5):
    """Applied batch normalization on the last axis of the tensor.

    Args:
        inp: A input Tensor 3D
        training (Boolean)): A scalar boolean tensor. training: true
        decay (float, optional): Defaults to 0.99. The mean renew as follow: mean = pop_mean * (1- decay) + decay * old_mean
        epsilon (float, optional): Defaults to 1e-5. A small float number to avoid dividing by 0.

    Returns:
        The normalized, scaled, offset tensor.
    """

    inpshape = inp.get_shape().as_list()
    tf.reshape(inp,[inpshape[0],1,inpshape[1],inpshape[2]])
    size = inpshape[-1]
    scale = tf.get_variable(
        'scale', shape=[size], initializer=tf.constant_initializer(0.1))
    offset = tf.get_variable('offset', shape=[size])

    pop_mean = tf.get_variable(
        'pop_mean',
        shape=[size],
        initializer=tf.zeros_initializer(),
        trainable=False)
    pop_var = tf.get_variable(
        'pop_var',
        shape=[size],
        initializer=tf.ones_initializer(),
        trainable=False)
    batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2])

    train_mean_op = tf.assign(pop_mean,
                              pop_mean * decay + batch_mean * (1 - decay))
    train_var_op = tf.assign(pop_var,
                             pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            normout = tf.nn.batch_normalization(inp, batch_mean, batch_var, offset, scale, epsilon)
            return tf.reshape(normout,[inpshape[0],inpshape[1],inpshape[2]])

    def population_statistics():
        normout = tf.nn.batch_normalization(inp, pop_mean, pop_var, offset, scale, epsilon)
        return tf.reshape(normout,[inpshape[0],inpshape[1],inpshape[2]])

    return tf.cond(training, batch_statistics, population_statistics)


def weightNormConvolution1d(x,
                            num_filters,
                            dilation_rate,
                            filter_size=3,
                            stride=[1],
                            pad='SAME',
                            init_scale=1.,
                            counters={},
                            reuse=False):
    """A dilated convolution with weight normalization (Salimans & Kingma 2016)
       Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
    Args:
        x: A tensor of shape [N, L, Cin]
        num_filters: number of convolution filters
        dilation_rate: dilation rate / holes
        filter_size: window / kernel width of each filter
        stride: stride in convolution
    Returns:
        A tensor of shape [N, L, num_filters]
    """
    # print(x.get_shape())
    name = get_name('weightnorm_conv', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
        num_filters = num_filters * 2
        stdval = math.sqrt(1 / int(x.get_shape()[-1]))

        V = tf.get_variable(
            'V',
            [filter_size, int(x.get_shape()[-1]), num_filters],
            tf.float32,
            tf.random_normal_initializer(0, stdval),
            trainable=True)
        g = tf.get_variable(
            'g',
            shape=[num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.),
            trainable=True)
        b = tf.get_variable(
            'b',
            shape=[num_filters],
            dtype=tf.float32,
            initializer=None,
            trainable=True)

        # use weight normalization (Salimans & Kingma, 2016),* means multiply
        W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])
        # print('W shape: ', W.get_shape())

        up_pad = dilation_rate * (filter_size - 1)
        # print('pad nums:', up_pad)
        tf.pad(x, [[0, 0], [up_pad, 0], [0, 0]])
        # print('after pad:', x.get_shape())
        x = tf.nn.bias_add(
            tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)
        split0, split1 = tf.split(x,num_or_size_splits=2,axis=2)
        # split0 = tf.tanh(split0)
        split1 = tf.sigmoid(split1)
        x = tf.multiply(split0,split1)
        # x = tf.nn.relu(x)
        # print("weightnorm_conv outs size : ", x.get_shape())
        return x


def TemporalBlock(input_layer, out_channels, filter_size, training, stride,
                  dilation_rate, counters, dropout):
    """temporal block in TCN
    Args:
        input_layer: A tensor of shape [N, L, Cin]
        out_channels: output dimension
        filter_size: receptive field of a conv. filter
        stride: same as what's need in conv. function
        dilation_rate: holes inbetween
        counters: to keep track of layer names
        dropout: prob. to drop weights

    Returns:
        A tensor of shape [N, L, out_channels]
    """
    keep_prob = 1.0 - dropout
    in_channels = int(input_layer.get_shape()[-1])
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        out1 = weightNormConvolution1d(
            input_layer,
            out_channels,
            dilation_rate,
            filter_size, [stride],
            counters=counters)

        out2 = weightNormConvolution1d(
            out1,
            out_channels,
            dilation_rate,
            filter_size, [stride],
            counters=counters)
        
        if in_channels != out_channels:
            stdval = math.sqrt(1 / in_channels)
            W_h = tf.get_variable(
                'W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                tf.float32,
                tf.random_normal_initializer(0, stdval),
                trainable=True)
            b_h = tf.get_variable(
                'b_h',
                shape=[out_channels],
                dtype=tf.float32,
                initializer=None,
                trainable=True)
            residual = tf.nn.bias_add(
                tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            residual = input_layer
        out2 = out2 + residual
        out2 = batchnorm(out2, training)
        return tf.nn.leaky_relu(out2,alpha=0.01)


def TemporalConvNet(inputs,
                    num_channels,
                    sequence_length,
                    dropout,
                    training,
                    kernel_size=3):
    """A stacked dilated architecture of TemporalBlock
    Args:
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN

    Returns
        A tensor of shape [N, L, num_channels[-1]]
    """
    in_shape = inputs.get_shape().as_list()
    # print(in_shape)
    batch_n = tf.shape(inputs)[0]
    input_layer = tf.reshape(inputs, [batch_n, in_shape[1], 1])
    num_levels = len(num_channels) - 1
    counters = {}
    for i in range(num_levels):
        dilation_size = 2**i
        out_channels = num_channels[i]
        input_layer = TemporalBlock(
            input_layer,
            out_channels,
            kernel_size,
            training,
            stride=1,
            dilation_rate=dilation_size,
            counters=counters,
            dropout=dropout)
    with tf.variable_scope("FC"):
        input_shape = input_layer.get_shape().as_list()
        # print('input layer shape : ',input_shape)
        tempconvout1 = tf.layers.dense(input_layer, input_shape[-1] / 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='FC1')   
        tempconvout1 = tf.nn.leaky_relu(tempconvout1, 0.01)
        input_shape = tempconvout1.get_shape().as_list()
        # print('after f1 shape : ',input_shape) 
        full_input_chan = input_shape[-1]    
        tempconvout1 = tf.reshape(tempconvout1, [-1, full_input_chan])
        Wh = tf.get_variable(
            "logit_weights",
            shape=[full_input_chan, num_channels[-1]],
            initializer=tf.contrib.layers.xavier_initializer())
        ba = tf.get_variable(
            "logit_bias",
            shape=[num_channels[-1]],
            initializer=None)
        tempconvout2 = tf.reshape(tf.nn.bias_add(tf.matmul(tempconvout1, Wh), ba, name='cnn_logits'),[input_shape[0], input_shape[1], num_channels[-1]],name='cnnlogits_rs')
    return tempconvout2
