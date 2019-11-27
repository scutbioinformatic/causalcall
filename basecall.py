import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import model
from data_input import read_data_for_eval
import easy_assembler
from time_os import time_os
from progress import multi_pbars
from extract_sig_ref import extract
import threading
from collections import defaultdict

# this file is revised based on chiron_eval.py

def sparse2dense(predict_val):
    """Transfer a sparse input in to dense representation"""
    
    predict_val_top5 = predict_val[0]
    predict_read = list()
    uniq_list = list()
    for i in range(len(predict_val_top5)):
        predict_val = predict_val_top5[i]
        unique, pre_counts = np.unique(
            predict_val.indices[:, 0], return_counts=True)
        uniq_list.append(unique)
        pos_predict = 0
        predict_read_temp = list()
        for indx, _ in enumerate(pre_counts):
            predict_read_temp.append(
                predict_val.values[pos_predict:pos_predict + pre_counts[indx]])
            pos_predict += pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read, uniq_list


def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    """
    base = ['A', 'C', 'G', 'T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def path_prob(logits):
    """Calculate the mean of the difference between highest and second highest logits in path.
    Given the p_i = exp(logit_i)/sum_k(logit_k), we can get the quality score for the concensus sequence as:
        qs = 10 * log_10(p1/p2) = 10 * log_10(exp(logit_1 - logit_2)) = 10 * ln(10) * (logit_1 - logit_2), 
        where p_1,logit_1 are the highest probability, logit, and p_2, logit_2 are the second highest probability logit.
    Args:
        logits (Float): Tensor of shape [batch_size, max_time,class_num], output logits.

    Returns:
        prob_logits(Float): Tensor of shape[batch_size].
    """

    fea_shape = tf.shape(logits)
    bsize = fea_shape[0]
    seg_len = fea_shape[1]
    top2_logits = tf.nn.top_k(logits, k=2)[0]
    logits_diff = tf.slice(top2_logits, [0, 0, 0], [bsize, seg_len, 1]) - tf.slice(
        top2_logits, [0, 0, 1], [bsize, seg_len, 1])
    prob_logits = tf.reduce_mean(logits_diff, axis=-2)
    return prob_logits


def qs(consensus, consensus_qs, output_standard='phred+33'):
    """Calculate the quality score for the consensus read.
    Args:
        consensus (Int): 2D Matrix ，(4 x length) given the count of base on each position.
        consensus_qs (Float): (4 x length) given the accumulated logits_prob on each position,logits_prob means the different between logit_1 and logit_2
        output_standard (str, optional): Defaults to 'phred+33'. Quality score output format.

    Returns:
        quality score: Return the queality score as int or string depending on the format.
    """

    sort_ind = np.argsort(consensus, axis=0)   # Sort by column ascending, return the index
    L = consensus.shape[1]
    sorted_consensus = consensus[sort_ind, np.arange(L)[np.newaxis, :]]  # return the sorted matrix , elements are ascending by column
    sorted_consensus_qs = consensus_qs[sort_ind, np.arange(L)[np.newaxis, :]]
    quality_score = 10 * (np.log10((sorted_consensus[3, :] + 1) / (
        sorted_consensus[2, :] + 1))) + sorted_consensus_qs[3, :] / sorted_consensus[3, :] / np.log(10)
    mean_qs = int(np.mean(quality_score.astype(int)))
    if output_standard == 'number':
        return quality_score.astype(int), mean_qs
    elif output_standard == 'phred+33':
        q_string = [chr(x + 33) for x in quality_score.astype(int)]
        return ''.join(q_string), mean_qs


def write_output(consensus, 
                 time_list, 
                 file_pre,
                 global_setting,
                 concise=False, 
                 suffix='fasta', 
                 seg_q_score=None,
                 q_score=None
                 ):
    """Write the output to the fasta(q) file.

    Args:
        consensus (str): String of the read represented in AGCT.
        time_list (Tuple): Tuple of time records.
        file_pre (str): Output fasta(q) file name(prefix).
        concise (bool, optional): Defaults to False. If False, the time records and segments will not be output.
        suffix (str, optional): Defaults to 'fasta'. Output file suffix from 'fasta', 'fastq'.
        seg_q_score ([str], optional): Defaults to None. Quality scores of read segment.
        q_score (str, optional): Defaults to None. Quality scores of the read.
        global_setting: The global Flags of chiron_eval.
    """
    start_time, reading_time, basecall_time, assembly_time = time_list
    result_folder = os.path.join(global_setting.output, 'result')
    meta_folder = os.path.join(global_setting.output, 'meta')
    path_con = os.path.join(result_folder, file_pre + '.' + suffix)
    if global_setting.mode == 'rna':
        consensus = consensus.replace('T','U').replace('t','u')
    if not concise:
        path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_con, 'w+', encoding = 'utf-8') as out_con:
        if (suffix == 'fastq') and (q_score is not None):
            out_con.write(
                '@{}\n{}\n+\n{}\n'.format(file_pre, consensus, q_score))
        else:
            out_con.write('>{}\n{}'.format(file_pre, consensus))
    if not concise:
        with open(path_meta, 'w+') as out_meta:
            total_time = time.time() - start_time
            output_time = total_time - assembly_time
            assembly_time -= basecall_time
            basecall_time -= reading_time
            total_len = len(consensus)
            total_time = time.time() - start_time
            out_meta.write(
                "# Reading Basecalling assembly output total rate(bp/s)\n")
            out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (
                reading_time, basecall_time, assembly_time, output_time, total_time, total_len / total_time))
            out_meta.write(
                "# read_len batch_size segment_len jump start_pos\n")
            out_meta.write(
                "%d %d %d %d %d\n" % (total_len, 
                                      global_setting.batch_size, 
                                      global_setting.segment_len, 
                                      global_setting.jump, 
                                      global_setting.start))
            out_meta.write("# input_name model_name\n")
            out_meta.write("%s %s\n" % (global_setting.input, global_setting.model))


def call():
    pbars = multi_pbars(["files processed:"])
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    num_channels = [FLAGS.hiddens] * (FLAGS.layers - 1) + [5]
    kernel_size = FLAGS.ksize
    dropout = FLAGS.dout
    logits, ratio = model.inference(x, num_channels, FLAGS.segment_len, kernel_size, training, dropout)

    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=FLAGS.threads,
                            inter_op_parallelism_threads=FLAGS.threads)
    config.gpu_options.allow_growth = True
    logits_index = tf.placeholder(tf.int32, shape=())
    logits_fname = tf.placeholder(tf.string, shape=())
    logits_queue = tf.FIFOQueue(
        capacity=1000,
        dtypes=[tf.float32, tf.string, tf.int32, tf.int32],
        shapes=[logits.shape,logits_fname.shape,logits_index.shape, seq_length.shape]       
    )
    logits_queue_size = logits_queue.size()
    logits_enqueue = logits_queue.enqueue((logits, logits_fname, logits_index, seq_length))
    logits_queue_close = logits_queue.close()

    ## Decoding logits into bases
    decode_predict_op, decode_prob_op, decoded_fname_op, decode_idx_op, decode_queue_size = decoding_queue(logits_queue)
    saver = tf.train.Saver(max_to_keep=2)
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config)) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model))
        if os.path.isdir(FLAGS.input):
            file_list = os.listdir(FLAGS.input)
            file_dir = FLAGS.input
        else:
            file_list = [os.path.basename(FLAGS.input)]
            file_dir = os.path.abspath(
                os.path.join(FLAGS.input, os.path.pardir))
        file_n = len(file_list)
        pbars.update(0,total = file_n)

        if not os.path.exists(FLAGS.output):
            os.makedirs(FLAGS.output)
        if not os.path.exists(os.path.join(FLAGS.output, 'result')):
            os.makedirs(os.path.join(FLAGS.output, 'result'))
        if not os.path.exists(os.path.join(FLAGS.output, 'meta')):
            os.makedirs(os.path.join(FLAGS.output, 'meta'))
        def worker_fn():
            for f_i, name in enumerate(file_list):
                if not name.endswith('.signal'):
                    continue
                input_path = os.path.join(file_dir, name)
                eval_data = read_data_for_eval(input_path, FLAGS.start,
                                               seg_length=FLAGS.segment_len,
                                               step=FLAGS.jump)
                reads_n = eval_data.reads_n

                for i in range(0, reads_n, FLAGS.batch_size):
                    batch_x, seq_len, _ = eval_data.next_batch(
                        FLAGS.batch_size, shuffle=False, sig_norm=False)
                    batch_x = np.pad(
                        batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='constant')
                    seq_len = np.pad(
                        seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')
                    feed_dict = {
                        x: batch_x,
                        seq_length: np.round(seq_len/ratio).astype(np.int32),
                        training: False,
                        logits_index:i,
                        logits_fname: name,
                    }
                    sess.run(logits_enqueue,feed_dict=feed_dict)
            sess.run(logits_queue_close)

        worker = threading.Thread(target=worker_fn,args=())
        worker.setDaemon(True)
        worker.start()

        val = defaultdict(dict)
        mean_qs_list = list()
        for f_i, name in enumerate(file_list):
            start_time = time.time()
            if not name.endswith('.signal'):
                continue
            file_pre = os.path.splitext(name)[0]
            input_path = os.path.join(file_dir, name)
            if FLAGS.mode == 'rna':
                eval_data = read_data_for_eval(input_path, FLAGS.start,
                                           seg_length=FLAGS.segment_len,
                                           step=FLAGS.jump)
            else:
                eval_data = read_data_for_eval(input_path, FLAGS.start,
                                           seg_length=FLAGS.segment_len,
                                           step=FLAGS.jump)
            reads_n = eval_data.reads_n
            reading_time = time.time() - start_time
            reads = list()

            N = len(range(0, reads_n, FLAGS.batch_size))
            while True:
                l_sz, d_sz = sess.run([logits_queue_size, decode_queue_size])
                decode_ops = [decoded_fname_op, decode_idx_op, decode_predict_op, decode_prob_op]
                decoded_fname, i, predict_val, logits_prob = sess.run(decode_ops, feed_dict={training: False})
                decoded_fname = decoded_fname.decode("UTF-8")
                val[decoded_fname][i] = (predict_val, logits_prob)               
                if len(val[name]) == N:
                    break

            pbars.update(0,progress = f_i+1)
            pbars.update_bar()
            qs_list = np.empty((0, 1), dtype=np.float)
            qs_string = None
            mean_qs = 0
            for i in range(0, reads_n, FLAGS.batch_size):
                predict_val, logits_prob = val[name][i]
                predict_read, unique = sparse2dense(predict_val)
                predict_read = predict_read[0]
                unique = unique[0]

                if FLAGS.extension == 'fastq':
                    logits_prob = logits_prob[unique]
                if i + FLAGS.batch_size > reads_n:
                    predict_read = predict_read[:reads_n - i]
                    if FLAGS.extension == 'fastq':
                        logits_prob = logits_prob[:reads_n - i]
                if FLAGS.extension == 'fastq':
                    qs_list = np.concatenate((qs_list, logits_prob))
                reads += predict_read
            val.pop(name)  # Release the memory

            basecall_time = time.time() - start_time
            bpreads = [index2base(read) for read in reads]
            if FLAGS.extension == 'fastq':
                consensus, qs_consensus = easy_assembler.simple_assembly_qs(bpreads, qs_list)
                qs_string, mean_qs = qs(consensus, qs_consensus)
                mean_qs_list.append((file_pre, str(mean_qs)))
            else:
                consensus = easy_assembler.simple_assembly(bpreads)
            c_bpread = index2base(np.argmax(consensus, axis=0))
            assembly_time = time.time() - start_time
            list_of_time = [start_time, reading_time,
                            basecall_time, assembly_time]
            write_output(c_bpread, list_of_time, file_pre, concise=FLAGS.concise, suffix=FLAGS.extension,
                         q_score=qs_string, global_setting=FLAGS)
        if FLAGS.extension == 'fastq':
            out_file = os.path.join(FLAGS.output, 'mean_qscore.txt')
            with open(out_file, 'w') as ofile:
                for files in mean_qs_list:
                    ofile.writelines(files[0]+'\n'+files[1]+'\n')


def decoding_queue(logits_queue, num_threads=12):
    """ decoding one batch data"""
    q_logits, q_name, q_index, seq_length = logits_queue.dequeue()
    if FLAGS.extension == 'fastq':
        prob = path_prob(q_logits)
    else:
        prob = tf.constant(0.0)
    if FLAGS.beam == 0:
        decode_decoded, decode_log_prob = tf.nn.ctc_greedy_decoder(tf.transpose(
            q_logits, perm=[1, 0, 2]), seq_length, merge_repeated=True)
    else:
        decode_decoded, decode_log_prob = tf.nn.ctc_beam_search_decoder(
            tf.transpose(q_logits, perm=[1, 0, 2]),
            seq_length, merge_repeated=False,
            beam_width=FLAGS.beam)
    decodeedQueue = tf.FIFOQueue(
        capacity=2 * num_threads,
        dtypes=[tf.int64 for _ in decode_decoded] * 3 + [tf.float32, tf.float32, tf.string, tf.int32],
    )
    ops = []
    for x in decode_decoded:
        ops.append(x.indices)
        ops.append(x.values)
        ops.append(x.dense_shape)
    decode_enqueue = decodeedQueue.enqueue(tuple(ops + [decode_log_prob, prob, q_name, q_index]))

    decode_dequeue = decodeedQueue.dequeue()
    decode_prob, decode_fname, decode_idx = decode_dequeue[-3:]

    decode_dequeue = decode_dequeue[:-3]
    decode_predict = [[], decode_dequeue[-1]]
    for i in range(0, len(decode_dequeue) - 1, 3):
        decode_predict[0].append(
            tf.SparseTensor(
                indices=decode_dequeue[i],
                values=decode_dequeue[i + 1],
                dense_shape=decode_dequeue[i + 2],
            )
        )

    decode_qr = tf.train.QueueRunner(decodeedQueue, [decode_enqueue]*num_threads)
    tf.train.add_queue_runner(decode_qr)
    return decode_predict, decode_prob, decode_fname, decode_idx, decodeedQueue.size()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Causalcall',
                                     description='nanopore basecaller.')
    parser.add_argument('-i', '--input', required = True,
                        help="Path of the fast5 files.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output Path")
    parser.add_argument('-m', '--model', required = True,
                        help="Path of the trained model")
    parser.add_argument('-s', '--start', type=int, default=0,
                        help="Start index of the signal file.")
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help="Batch size for run")
    parser.add_argument('-l', '--segment_len', type=int, default=512,
                        help="Segment length to be divided into.")
    parser.add_argument('-j', '--jump', type=int, default=120,
                        help="Step size for segment")
    parser.add_argument(
        '-hd',
        '--hiddens',
        type=int,
        default=256,
        help='# hiddens units of the cascul conv')
    parser.add_argument(
        '-ly', '--layers', type=int, default=6, help='# of Temp block layer')
    parser.add_argument(
        '-kr', '--ksize', type=int, default=3, help='kernel size')
    parser.add_argument('-dp', '--dout', type=float, default=0, help='dropout')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help="Threads number")
    parser.add_argument('-e', '--extension', default='fastq',
                        help="Output file extension.")
    parser.add_argument('--beam', type=int, default=10,
                        help="Beam width used in beam search decoder, default is 10.")
    parser.add_argument('--concise', action='store_true',
                        help="If set concise, the meta files will not be output.")
    parser.add_argument('--mode', default='dna',
                        help="Only dna is available at present.")
    pyfile_path = os.path.split(os.path.realpath(__file__))[0]
    args = parser.parse_args(sys.argv[1:])
    FLAGS = args
    print("input path: ",FLAGS.input)
    print("output path: ",FLAGS.output)
    print("model: ",FLAGS.model)
    FLAGS.input_dir = FLAGS.input
    FLAGS.output_dir = FLAGS.output
    FLAGS.recursive = True
    extract(FLAGS)
    FLAGS.input = FLAGS.output + os.sep + 'raw' + os.sep
    print("start basecalling")
    time_dict = time_os(call)
    print("basecall completed!")
    meta_folder = os.path.join(FLAGS.output, 'meta')
    if os.path.isdir(FLAGS.input):
        file_pre = 'all'
    else:
        file_pre = os.path.splitext(os.path.basename(FLAGS.input))[0]
    path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_meta, 'a+') as out_meta:
        out_meta.write("# Wall_time Sys_time User_time Cpu_time\n")
        out_meta.write("%5.3f %5.3f %5.3f %5.3f\n" % (
            time_dict['real'], time_dict['sys'], time_dict['user'], time_dict['sys'] + time_dict['user']))
    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f' %
          (time_dict['real'], time_dict['sys'], time_dict['user']))
