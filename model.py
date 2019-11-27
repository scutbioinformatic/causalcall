import tensorflow as tf
import os
from tcn import *

# for pretrain
LR_BOUNDARY = [0.03,0.07,0.25,0.5,0.7]
LR_DECAY = [0.4,0.2,0.1,0.06,0.03,0.01]
# for finetunning
# LR_BOUNDARY = [0.75]
# LR_DECAY = [0.1,0.01]


def loss(logits, seq_len, label):
    """Calculate a CTC loss from the input logits and label.

    Args:
        logits: Tensor of shape [batch_size,max_time,class_num], logits from last layer
        seq_len: Tensor of shape [batch_size], sequence length for each sample in the batch.
        label: A Sparse Tensor of labels, sparse tensor of the true label.

    Returns:
        Tensor of shape [batch_size], losses of the batch.
    """
    loss = tf.reduce_mean(
        tf.nn.ctc_loss(
            label,
            logits,
            seq_len,
            ctc_merge_repeated=True,
            time_major=False,
            ignore_longer_outputs_than_inputs=True))
    tf.add_to_collection('losses', loss) 
    tf.summary.scalar('loss', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train_opt(init_rate, max_steps, global_step=None, opt_name="Adam"):
    """Generate training optimizer
    Args:
        init_rate: initial learning rate.
        max_steps: maximum training steps.
        global_step: A optional Scalar tensor, the global step recorded, if None no global stop will be recorded.
        opt_name: name of optimizer
    Returns:
        opt: Optimizer
    """
    optimizers = {
        "Adam": tf.train.AdamOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer
    }
    boundaries = [int(max_steps * bound) for bound in LR_BOUNDARY]
    values = [init_rate * decay for decay in LR_DECAY]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values) 
    opt = optimizers[opt_name](learning_rate)
    return opt


def prediction(logits, seq_length, label, beam_width=10, top_paths=1):
    """
    Args:
        logits:Input logits from TCN.Shape = [batch_size,max_time,class_num]
        seq_length:sequence length of logits. Shape = [batch_size]
        label:Sparse tensor of label.
        beam_width(Int):Beam width used in beam search decoder.
        top_paths:The number of top score path to choice from the decoder.
    """
    logits = tf.transpose(logits, perm=[1, 0, 2])
    if beam_width == 0:
        predict = tf.nn.ctc_greedy_decoder(
            logits, seq_length, merge_repeated=True)
    else:
        predict = tf.nn.ctc_beam_search_decoder(
            logits,
            seq_length,
            merge_repeated=False,
            top_paths=top_paths,
            beam_width=beam_width)
    predict = predict[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(
            tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d, axis=0)
    d_min = tf.reduce_min(edit_d, axis=0)
    error = tf.reduce_mean(d_min, axis=0)
    tf.summary.scalar('Error_rate', error)
    predictdense=[]
    for x in predict:
        predictdense.append(tf.sparse_to_dense(x.indices,x.dense_shape,x.values))
    return error, predictdense


def inference(x, num_channels, sequence_length, kernel_size, training, dropout):
    """Infer a logits of the input current measurements batch.

    Args:
        x: Tensor of shape [batch_size, max_time,channel], a batch of the input signal with a maximum length `max_time`.
        sequence_len: Tensor of shape [batch_size], given the real lenghs of the segments.
        training: Placeholder of Boolean, Ture if the inference is during training.
        full_sequence_len: Scalar float, the maximum length of the sample in the batch.
        configure:Model configuration.
    Returns:
        logits: Tensor of shape [batch_size, max_time, class_num]
        ratio: Scalar float, the scale factor between the output logits and the input maximum length.
    """
    logits = TemporalConvNet(
        x,
        num_channels,
        sequence_length,
        dropout=dropout,
        training=training,
        kernel_size=kernel_size)

    logitsshape = logits.get_shape().as_list()
    ratio = sequence_length / logitsshape[1]
    return logits, ratio
