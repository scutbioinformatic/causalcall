import os
import sys
import time
import argparse
import tensorflow as tf
import model
from data_input import read_tfrecord
from data_input import read_cache_dataset


def train():
    global FLAGS
    training = tf.placeholder(tf.bool)
    global_step = tf.get_variable(
        'global_step',
        trainable=False,
        shape=(),
        dtype=tf.int32,
        initializer=tf.zeros_initializer())
    x = tf.placeholder(
        tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs, y_values, y_shape)
    num_channels = [FLAGS.hiddens] * (FLAGS.layers - 1) + [5]
    kernel_size = FLAGS.ksize
    dropout = FLAGS.dout
    logits, ratio = model.inference(x, num_channels, FLAGS.sequence_len, kernel_size, training, dropout)
    ctc_loss = model.loss(logits, seq_length, y)
    tf.summary.scalar('ctc_loss', ctc_loss)
    opt = model.train_opt(
        FLAGS.step_rate,
        FLAGS.max_steps,
        global_step=global_step,
        opt_name="Adam")

    gradients = opt.compute_gradients(ctc_loss)
    """ for g in gradients:
        tf.summary.histogram("%s-grad" % g[1].name, g[0]) """
    step = opt.apply_gradients(gradients, global_step=global_step)
    learningrate=opt._lr
    error,predictdense = model.prediction(logits, seq_length, y)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)
    summary = tf.summary.merge_all()
    print("Training settings:")
    for pro in dir(FLAGS):
        if not pro.startswith('_'):
            print("%s:%s" % (pro, getattr(FLAGS, pro)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    if FLAGS.retrain == False:
        sess.run(init)
        print("Model inited, begin loading data. \n")
    else:
        saver.restore(
            sess, tf.train.latest_checkpoint(FLAGS.model_dir + FLAGS.model_name))
        print("Model loaded, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(
        FLAGS.model_dir + FLAGS.model_name + os.path.sep + 'summary' +
        os.path.sep, sess.graph)
    train_data, vali_data = generate_train_valid_datasets()
    start = time.time()

    for i in range(FLAGS.max_steps):
        # print('train_data shape : ', len(train_data.event))
        batch_x, seq_len, batch_y = train_data.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        feed_dict = {
            x: batch_x,
            seq_length: seq_len / ratio,
            y_indexs: indxs,
            y_values: values,
            y_shape: shape,
            training: True
        }
        loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
        print('learning rate: ',sess.run(learningrate))
        print('iteration:%d,loss_val:%5.3f' % (i,loss_val))
        
        # validation every 10 iterations
        if i % 10 == 0:
            global_step_val = tf.train.global_step(sess, global_step)
            valid_x, valid_len, valid_y = vali_data.next_batch(FLAGS.batch_size)
            indxs, values, shape = valid_y
            feed_dict = {
                x: valid_x,
                seq_length: valid_len / ratio,
                y_indexs: indxs,
                y_values: values,
                y_shape: shape,
                training: True
            }
            error_val = sess.run(error, feed_dict=feed_dict)
            # print(sess.run(predictdense,feed_dict=feed_dict))
            end = time.time()
            print("Step %d/%d Epoch %d, batch number %d, train_loss: %5.3f validate_edit_distance: %5.3f Elapsed Time/step: %5.3f" % (i, FLAGS.max_steps, train_data.epochs_completed,train_data.index_in_epoch, loss_val, error_val,(end - start) / (i + 1)))
            saver.save(
                sess,
                FLAGS.model_dir + FLAGS.model_name + os.path.sep + 'model.ckpt',
                global_step=global_step_val)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(
                summary_str, global_step=global_step_val)
            summary_writer.flush()
    global_step_val = tf.train.global_step(sess, global_step)
    print("Model %s saved." % (FLAGS.model_dir + FLAGS.model_name))
    print("Reads number %d" % (train_data.reads_n))
    saver.save(
        sess,
        FLAGS.model_dir + FLAGS.model_name + os.path.sep + 'final.ckpt',
        global_step=global_step_val)


def generate_train_valid_datasets(initial_offset=10):
    if FLAGS.read_cache:
        train_data = read_cache_dataset(FLAGS.train_cache)
        if FLAGS.vali_tf is not None:
            vali_data = read_cache_dataset(FLAGS.valid_cache)
        else:
            vali_data = train_data
        if train_data.event.shape[1] != FLAGS.sequence_len:
            raise ValueError(
                "The event length of training cached dataset %d is inconsistent with given sequene_len %d"
                % (train_data.event.shape()[1], FLAGS.sequence_len))
        if vali_data.event.shape[1] != FLAGS.sequence_len:
            raise ValueError(
                "The event length of training cached dataset %d is inconsistent with given sequene_len %d"
                % (vali_data.event.shape()[1], FLAGS.sequence_len))
        return train_data, vali_data
    sys.stdout.write("Begin reading training dataset.\n")
    train_data = read_tfrecord(
        FLAGS.tfrecord_dir,
        FLAGS.train_tf,
        FLAGS.train_cache,
        FLAGS.sequence_len,
        k_mer=FLAGS.k_mer,
        max_segments_num=FLAGS.segments_num,
        skip_start=initial_offset)
    sys.stdout.write("Begin reading validation dataset.\n")
    if FLAGS.vali_tf is not None:
        vali_data = read_tfrecord(
            FLAGS.tfrecord_dir,
            FLAGS.vali_tf,
            FLAGS.valid_cache,
            FLAGS.sequence_len,
            k_mer=FLAGS.k_mer,
            max_segments_num=FLAGS.segments_num)
    else:
        vali_data = train_data
    return train_data, vali_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training causalcall model with tfrecords file')
    parser.add_argument(
        '-i',
        '--tfrecord_dir',
        required=True,
        help="Directory that store the tfrecord files.")
    parser.add_argument(
        '-o',
        '--model_dir',
        required=True,
        help="directory to store the training model.")
    parser.add_argument('-m', '--model_name', required=True, help='model_name')
    parser.add_argument(
        '-v',
        '--vali_tf',
        default='validate.tfrecords',
        help=
        "file name of validation data, default is validation.tfrecords"
    )
    parser.add_argument(
        '-t', '--train_tf', default="train.tfrecords", help='file name of training data, default is train.tfrecords')
    parser.add_argument(
        '--train_cache', default=None, help="Cache file for training dataset.")
    parser.add_argument(
        '--valid_cache',
        default=None,
        help="Cache file for validation dataset.")
    parser.add_argument(
        '-s',
        '--sequence_len',
        type=int,
        default=512,
        help='the length of input segment')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument(
        '-st', '--step_rate', type=float, default=1e-2, help='Step rate')
    parser.add_argument(
        '-x', '--max_steps', type=int, default=200000, help='Maximum step')
    parser.add_argument(
        '-hd',
        '--hiddens',
        type=int,
        default=256,
        help='# hiddens units of the cascul conv')
    parser.add_argument(
        '-l', '--layers', type=int, default=6, help='# of Temp block layer')
    parser.add_argument(
        '-kr', '--ksize', type=int, default=3, help='kernel size')
    parser.add_argument('-dp', '--dout', type=float, default=0, help='dropout')
    parser.add_argument(
        '-n',
        '--segments_num',
        type=int,
        default=None,
        help=
        'Maximum number of segments read into the training queue, default(None) read all segments.'
    )
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument(
        '--retrain',
        dest='retrain',
        action='store_true',
        help='Set retrain to true')
    parser.add_argument(
        '--read_cache',
        dest='read_cache',
        action='store_true',
        help="Read from cached hdf5 file.")
    parser.set_defaults(retrain=False)
    
    args = parser.parse_args(sys.argv[1:])
    if args.train_cache is None:
        args.train_cache = args.tfrecord_dir + os.path.sep + 'train_cache.hdf5'
    if (args.valid_cache is None) and (args.vali_tf is not None):
        args.valid_cache = args.tfrecord_dir + os.path.sep + 'valid_cache.hdf5'

    FLAGS = args
    FLAGS.tfrecord_dir = FLAGS.tfrecord_dir + os.path.sep
    FLAGS.model_dir = FLAGS.model_dir + os.path.sep
    train()
