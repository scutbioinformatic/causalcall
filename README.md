# Causalcall
Code for the paper *Causalcall: nanopore basecalling using a temporal convolutional network*.

# Environment
- Ubuntu 14.04 
- python 3.6 
- tensorflow 1.8 </br>
Dependencies: </br>
numpy, collections, threading, tempfile, h5py, statsmodels, difflib, argparse, tqdm, multiprocessing, psutil

# Usages
### Preparing the training data:
Extract training and validation data (.tfrecords) from resquiggled fast5 files:
```
python raw.py -i training/validation_fast5_files_folder -o tfrecords_files_folder -f train.tfrecords/validate.tfrecords
```
### Training:
Train the model with default parameters:
```
CUDA_VISIBLE_DEVICES=cuda_id python train.py -i tfrecords_files_folder -o model_folder -m model_name
```
Make sure that train.tfrecords and validate.tfrecords are both in the tfrecords_directory.</br>
If you want to change parameters, use `train.py -h` for more details.

### Basecalling:
```
CUDA_VISIBLE_DEVICES=cuda_id python basecall.py -i fast5_files_folder -o results_folder -m path_to_model
```
Path to default model:   *./model/DNAmodel/* </br>

### Acknowledgement
We thank Chiron for providing the [source code](https://github.com/haotianteng/Chiron). Causalcall is developed on the basic framework of Chiron's code.
(The parts of preprocessing input data and converting outputs of the TCN-based model to base sequences are revised based on Chiron's code following MPL 2.0.)
