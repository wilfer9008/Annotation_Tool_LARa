'''
Created on Feb 27, 2019

@author: fmoya

HAR using Pytorch
Code updated from caffe/theano implementations
'''

from __future__ import print_function
import os
import logging
from logging import handlers
import torch
import numpy as np
import random

from modus_selecter import Modus_Selecter

import datetime


def configuration(dataset_idx, network_idx, output_idx, usage_modus_idx=0, dataset_fine_tuning_idx=0,
                  reshape_input=False, learning_rates_idx=0, name_counter=0, freeze=0, proportions_id=0,
                  gpudevice="0", fully_convolutional=False):
    """
    Set a configuration of all the possible variables that were set in the experiments.
    This includes the datasets, hyperparameters for training, networks, outputs, datasets paths,
    results paths

    @param dataset_idx: id of dataset
    @param network_idx: id of network 0 for tcnn, 1, for tcnn-lstm, 2 tcnn-IMU
    @param output_idx: 0 for softmax, 1 for attributes
    @param usage_modus_idx: id of the modus 0 for train, 1 for test, 2 for evolution, 3 for train_final,...
    @param dataset_fine_tuning_idx: id of source dataset in case of finetuning
    @param reshape_input: reshaping the input False for [C,T] or, True for [3,C/3,T]=[[x,y,z], [sensors], Time]
    @param learning_rates_idx: id for the Learning Rate
    @param name_counter: counter for experiments
    @param freeze: 0 for freezing the CNN layers, or 1 for fine-tuning
    @param proportions_id: Percentage for the training dataset
    @param gpudevice: GPU ID device
    @param fully_convolutional: False for FC or True for FCN
    @return: configuration: dict with all the configurations
    """

    #Flags
    plotting = False
    fine_tunning = False

    #Options
    dataset = {0: 'locomotion', 1: 'gesture', 2: 'carrots', 3: 'pamap2', 4: 'orderpicking', 5: 'virtual',
               6: 'mocap_half', 7: 'virtual_quarter', 8: 'mocap_quarter', 9: 'mbientlab_quarter',
               10: 'mbientlab'}
    network = {0: 'cnn', 1: 'lstm', 2: 'cnn_imu'}
    output = {0: 'softmax', 1: 'attribute'}
    usage_modus = {0: 'train', 1: 'test', 2: 'evolution', 3: 'train_final', 4: 'train_random', 5: 'fine_tuning'}

    assert usage_modus_idx == 2 and output_idx == 1, "Output should be Attributes for starting evolution"

    #Dataset Hyperparameters
    NB_sensor_channels = {'locomotion': 113, 'gesture': 113, 'carrots': 30, 'pamap2': 40, 'orderpicking': 27}
    sliding_window_length = {'locomotion': 24, 'gesture': 24, 'carrots': 64, 'pamap2': 100, 'orderpicking': 100}
    sliding_window_step = {'locomotion': 12, 'gesture': 2, 'carrots': 5, 'pamap2': 22, 'orderpicking': 1}
    num_attributes = {'locomotion': 10, 'gesture': 32, 'carrots': 32, 'pamap2': 24, 'orderpicking': 16}
    num_classes = {'locomotion': 5, 'gesture': 18, 'carrots': 16, 'pamap2': 12, 'orderpicking': 8}
    
    # Learning rate
    learning_rates = [0.0001, 0.00001, 0.000001]
    lr = {'locomotion': {'cnn': 0.00001, 'lstm': 0.001, 'cnn_imu': 0.0001},
          'gesture': {'cnn': 0.00001, 'lstm': 0.001, 'cnn_imu': 0.0001},
          'carrots': {'cnn': 0.00001, 'lstm': 0.000001, 'cnn_imu': 0.0001},
          'pamap2': {'cnn': 0.0001, 'lstm': 0.0001, 'cnn_imu': 0.00001},
          'orderpicking': {'cnn': 0.0001, 'lstm': 0.0001, 'cnn_imu': 0.001}}
    lr_mult = 1.0
        
    #Maxout
    use_maxout = {'cnn': False, 'lstm': False, 'cnn_imu': False}

    #Balacing
    balancing = {'locomotion': False, 'gesture': False, 'carrots': False, 'pamap2': False, 'orderpicking': False}
    
    #Epochs
    epochs = {'locomotion': {'cnn': {'softmax': 40, 'attribute': 5},
                              'lstm': {'softmax': 10, 'attribute': 5},
                              'cnn_imu': {'softmax': 40, 'attribute': 5}},
              'gesture': {'cnn': {'softmax': 10, 'attribute': 5},
                           'lstm': {'softmax': 6, 'attribute': 5},
                           'cnn_imu': {'softmax': 10, 'attribute': 32}},
              'carrots': {'cnn': {'softmax': 32, 'attribute': 32},
                           'lstm': {'softmax': 30, 'attribute': 5},
                           'cnn_imu': {'softmax': 32, 'attribute': 32}},
              'pamap2': {'cnn': {'softmax': 50, 'attribute': 32},
                          'lstm': {'softmax': 25, 'attribute': 1},
                          'cnn_imu': {'softmax': 50, 'attribute': 32}},
              'orderpicking': {'cnn': {'softmax': 10, 'attribute': 10},
                                'lstm': {'softmax': 25, 'attribute': 1},
                                'cnn_imu': {'softmax': 24, 'attribute': 32}}}
    division_epochs = {'locomotion': 1, 'gesture': 3, 'carrots': 2, 'pamap2': 3, 'orderpicking': 1}
    
    #Batch size
    batch_size = {'cnn': {'locomotion': 200, 'gesture': 200, 'carrots': 128, 'pamap2': 100, 'orderpicking': 100},
                  'lstm': {'locomotion': 100, 'gesture': 100, 'carrots': 128, 'pamap2': 50, 'orderpicking': 100},
                  'cnn_imu': {'locomotion': 200, 'gesture': 200, 'carrots': 128, 'pamap2': 100, 'orderpicking': 100}}

    accumulation_steps = {'locomotion': 4, 'gesture': 4, 'carrots': 4, 'pamap2': 4, 'orderpicking': 4}
    
    #Filters
    filter_size =  {'locomotion': 5, 'gesture': 5, 'carrots': 5, 'pamap2': 5, 'orderpicking': 5}
    num_filters = {'locomotion': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64},
                   'gesture': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64},
                   'carrots': {'cnn': 256, 'lstm': 64, 'cnn_imu': 64},
                   'pamap2': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64},
                   'orderpicking': {'cnn': 32, 'lstm': 32, 'cnn_imu': 32}}

    freeze_options = [False, True]

    #Evolution
    evolution_iter = 10000

    # Results will be stored in different folders according to the dataset and network
    # This as a sort of organisation for tracking the experiments
    # dataset/network/output/MLP_type/input_shape/
    # dataset/network/output/MLP_type/input_shape/experiment
    # dataset/network/output/MLP_type/input_shape/experiment/plots
    # dataset/network/output/MLP_type/input_shape/final
    # dataset/network/output/MLP_type/input_shape/final/plots
    # dataset/network/output/MLP_type/input_shape/fine_tuning
    # dataset/network/output/MLP_type/input_shape/fine_tuning/plots

    # User gotta take care of creating these folders, or storing the results in a different way

    reshape_input = reshape_input
    if reshape_input:
        reshape_folder = "reshape"
    else:
        reshape_folder = "noreshape"

    if fully_convolutional:
        fully_convolutional = "FCN"
    else:
        fully_convolutional = "FC"
    
    # Folder
    if usage_modus[usage_modus_idx] == 'train':
        folder_exp = '/path_where_results_will_be_stored/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + '/' \
                     + reshape_folder + '/' + 'experiment/'
        folder_exp_base_fine_tuning = '/path_where_results_will_be_stored/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                                      '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'test':
        folder_exp = '/path_where_results_will_be_stored/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                     '/' + reshape_folder + '/' + 'final/'
        folder_exp_base_fine_tuning = '/path_where_results_will_be_stored/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                                      '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'train_final':
        folder_exp = '/path_where_results_will_be_stored/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional +\
                     '/' + reshape_folder + '/' + 'final/'
        folder_exp_base_fine_tuning = '/path_where_results_will_be_stored/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                                      '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'fine_tuning':
        folder_exp = '/path_where_results_will_be_stored/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                     '/' + reshape_folder + '/' + 'fine_tuning/'
        folder_exp_base_fine_tuning = '/path_where_results_will_be_stored/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + fully_convolutional + \
                                      '/' + reshape_folder + '/' + 'final/'
    else:
        raise ("Error: Not selected fine tuning option")

    #dataset
    # Paths are given according to the ones created in *preprocessing.py for the datasets
    dataset_root = {'locomotion': '/path_to_dataset/OpportunityUCIDataset/',
                    'gesture': '/path_to_dataset/OpportunityUCIDataset/',
                    'pamap2': '/path_to_dataset/PAMAP/',
                    'orderpicking': '/path_to_dataset/Order_Picking/'}
    
    #GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpudevice
    GPU = 0

    #Labels position on the segmented window
    label_pos = {0: 'middle', 1: 'mode', 2: 'end'}

    if dataset[dataset_idx] == 'carrots':
        train_show = {'cnn': 50, 'lstm': 50, 'cnn_imu':20}
        valid_show = {'cnn': 100, 'lstm': 20, 'cnn_imu':200}
    if dataset[dataset_idx] == 'pamap2':
        train_show = {'cnn': 50, 'lstm': 100, 'cnn_imu':50}
        valid_show = {'cnn': 400, 'lstm': 500, 'cnn_imu':400}
    else:
        train_show = {'cnn': 50, 'lstm': 100, 'cnn_imu':50}
        valid_show = {'cnn': 400, 'lstm': 500, 'cnn_imu':400}

    proportions = [0.2, 0.5, 1.0]

    reshape_input = False

    now = datetime.datetime.now()


    configuration = {'dataset': dataset[dataset_idx],
                     'dataset_finetuning': dataset[dataset_fine_tuning_idx],
                     'network': network[network_idx],
                     'output': output[output_idx],
                     'num_filters': num_filters[dataset[dataset_idx]][network[network_idx]],
                     'filter_size': filter_size[dataset[dataset_idx]],
                     'lr': lr[dataset[dataset_idx]][network[network_idx]] * lr_mult,
                     'epochs': epochs[dataset[dataset_idx]][network[network_idx]][output[output_idx]],
                     'evolution_iter': evolution_iter,
                     'train_show': int(train_show[network[network_idx]] * proportions[proportions_id]),
                     'valid_show': int(valid_show[network[network_idx]] * proportions[proportions_id]),
                     #'test_person': test_person,
                     'plotting': plotting,
                     'usage_modus': usage_modus[usage_modus_idx],
                     'folder_exp': folder_exp,
                     'folder_exp_base_fine_tuning': folder_exp_base_fine_tuning,
                     'use_maxout': use_maxout[network[network_idx]],
                     'balancing': balancing[dataset[dataset_idx]],
                     'GPU': GPU,
                     'division_epochs': division_epochs[dataset[dataset_idx]],
                     'NB_sensor_channels': NB_sensor_channels[dataset[dataset_idx]],
                     'sliding_window_length': sliding_window_length[dataset[dataset_idx]],
                     'sliding_window_step': sliding_window_step[dataset[dataset_idx]],
                     'num_attributes': num_attributes[dataset[dataset_idx]],
                     'batch_size': batch_size[network[network_idx]][dataset[dataset_idx]],
                     'num_classes': num_classes[dataset[dataset_idx]],
                     'label_pos': label_pos[2],
                     'file_suffix': 'results_yy{}mm{}dd{:02d}hh{:02d}mm{:02d}.xml'.format(now.year,
                                                                                          now.month,
                                                                                          now.day,
                                                                                          now.hour,
                                                                                          now.minute),
                     'dataset_root': dataset_root[dataset[dataset_idx]],
                     'accumulation_steps': accumulation_steps[dataset[dataset_idx]],
                     'reshape_input': reshape_input,
                     'name_counter': name_counter,
                     'freeze_options': freeze_options[freeze],
                     'proportions': proportions[proportions_id],
                     'fully_convolutional': fully_convolutional}
    
    return configuration


def setup_experiment_logger(logging_level=logging.DEBUG, filename=None):
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
    if filename != None:
        logging.basicConfig(filename=filename,level=logging.DEBUG,
                            format=logging_format,
                            filemode='w')
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format,
                            filemode='w')
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)   


    return



def pamap2_main():
    """
    Run experiment for a certain set of parameters for PAMAP@

    User is welcome to revise in detail the configuration function
    for more information about all of possible configurations for the experiments

    """
    datasets_opts = [3]
    networks_arc = [2]
    fine_tunings = [10]
    frezze_opts = [0]
    proportions_opts = [0, 1, 2]
    for dset in datasets_opts:
        for ft in fine_tunings:
            for arch in networks_arc:
                for fopt in frezze_opts:
                    for pp in proportions_opts:
                        config = configuration(dataset_idx=dset, network_idx=arch, output_idx=0, usage_modus_idx=5,
                                               dataset_fine_tuning_idx=ft, learning_rates_idx=0, name_counter=0,
                                               freeze=fopt, proportions_id=pp, gpudevice="6")
                        setup_experiment_logger(logging_level=logging.DEBUG,
                                                filename=config['folder_exp'] + "logger.txt")
                        logging.info('Finished')
                        modus = Modus_Selecter(config)
                        #Starting process
                        modus.net_modus()
    return


def locomotion_main():
    """
    Run experiment for a certain set of parameters for Opportunity Locomotion

    User is welcome to revise in detail the configuration function
    for more information about all of possible configurations for the experiments

    """
    datasets_opts = [0]
    networks_arc = [0]
    fine_tunings = [8]
    frezze_opts = [0]
    proportions_opts = [2]
    for dset in datasets_opts:
        for ft in fine_tunings:
            for arch in networks_arc:
                for fopt in frezze_opts:
                    for pp in proportions_opts:
                        config = configuration(dataset_idx=dset, network_idx=arch, output_idx=0, usage_modus_idx=5,
                                               dataset_fine_tuning_idx=ft, learning_rates_idx=0, name_counter=0,
                                           freeze=fopt, proportions_id = pp, gpudevice = "5")
                        setup_experiment_logger(logging_level=logging.DEBUG, filename= config['folder_exp'] + "logger.txt")
                        logging.info('Finished')
                        modus = Modus_Selecter(config)
                        #Starting process
                        modus.net_modus()
    return

def gestures_main():
    """
    Run experiment for a certain set of parameters for Opportunity Gestures

    User is welcome to revise in detail the configuration function
    for more information about all of possible configurations for the experiments

    """

    datasets_opts = [1]
    networks_arc = [0]
    fine_tunings = [7]
    frezze_opts = [0]
    proportions_opts = [2]
    for dset in datasets_opts:
        for ft in fine_tunings:
            for arch in networks_arc:
                for fopt in frezze_opts:
                    for pp in proportions_opts:
                        config = configuration(dataset_idx=dset, network_idx=arch, output_idx=0, usage_modus_idx=5,
                                               dataset_fine_tuning_idx=ft, learning_rates_idx=0, name_counter=0,
                                               freeze=fopt, proportions_id = pp, gpudevice = "4")
                        setup_experiment_logger(logging_level=logging.DEBUG, filename= config['folder_exp'] + "logger.txt")
                        logging.info('Finished')
                        modus = Modus_Selecter(config)
                        #Starting process
                        modus.net_modus()
    return




if __name__ == '__main__':

    #Setting the same RNG seed
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)
    
    pamap2_main()
    #locomotion_main()
    #gestures_main()
    
    print("Done")
