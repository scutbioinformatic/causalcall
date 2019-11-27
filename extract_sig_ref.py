# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Thu May  4 10:57:35 2017

import os
import h5py
import logging
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
logger = logging.getLogger(name='basecall_log')


def set_logger(log_file):
    global logger
    log_hd = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_hd.setFormatter(formatter)
    logger.addHandler(log_hd) 
    logger.propagate = False
    if __name__ == "__main__":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

def extract(FLAGS):
    root_folder = FLAGS.input_dir
    out_folder = FLAGS.output_dir
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if out_folder is None:
        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        log_folder = os.path.abspath(os.path.join(out_folder, 'log'))
    else:
        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        log_folder = os.path.abspath(os.path.join(out_folder, 'log'))
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    FLAGS.raw_folder = raw_folder
    FLAGS.log_folder = log_folder
    set_logger(os.path.join(FLAGS.log_folder,'extract.log'))
    FLAGS.count = 0
    tqdm.monitor_interval = 0
    if FLAGS.threads == 0:
        FLAGS.threads = cpu_count()
    pool = Pool(FLAGS.threads)
    if FLAGS.recursive:
        dir_list = os.walk(root_folder)
    else:
        dir_list = [root_folder]
    for dir_tuple in tqdm(dir_list,desc = "Subdirectory processing:",position = 0):
        if FLAGS.recursive:
            directory = dir_tuple[0]
            file_list = dir_tuple[2]
        else:
            file_list = os.listdir(dir_tuple)
            directory = dir_tuple
        file_list = [(os.path.join(directory,f),FLAGS) for f in file_list]
        for _ in tqdm(pool.imap_unordered(extract_file_wrapper,file_list),total = len(file_list)):
            pass
    pool.close()
    pool.join()        
            
def extract_file_wrapper(args):
    global logger
    full_file_n, FLAGS = args
    file_n = os.path.basename(full_file_n)
    if full_file_n.endswith('fast5'):
        try:
            raw_signal, reference = extract_file(full_file_n,FLAGS.mode)
            if raw_signal is None:
                raise ValueError("Fail in extracting raw signal.")
            if len(raw_signal) == 0:
                raise ValueError("Got empty raw signal")
        except Exception as e:
            logger.error("Cannot extract file %s. %s"%(full_file_n,e))
            return
        with open(os.path.join(FLAGS.raw_folder, os.path.splitext(file_n)[0] + '.signal'), 'w+') as signal_file:
            signal_file.write(" ".join([str(val) for val in raw_signal]))
    return

def extract_file(input_file,mode = 'dna'):
    global logger 
    try:
        input_data = h5py.File(input_file, 'r')
    except IOError as e:
        logger.error(e)
        raise IOError(e)
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    raw_signal = list(input_data['/Raw/Reads'].values())[0]['Signal'][()]
    reference = ''
    return raw_signal, reference
