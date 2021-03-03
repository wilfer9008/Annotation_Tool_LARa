'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import os
import sys
import logging
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hdfs.config import catch

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import PolyCollection

from network import Network

from HARWindows import HARWindows

from metrics import Metrics


class Network_User(object):
    '''
    classdocs
    '''


    def __init__(self, config, exp):
        '''
        Constructor
        '''

        logging.info('        Network_User: Constructor')

        self.config = config
        self.device = torch.device("cuda:{}".format(self.config["GPU"]) if torch.cuda.is_available() else "cpu")

        self.attrs = self.reader_att_rep("atts_per_class_lara.txt")
        self.attr_representation = self.reader_att_rep("atts_per_class_lara.txt")

        self.normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.001]))
        self.exp = exp

        return


    ##################################################
    ###################  reader_att_rep  #############
    ##################################################

    def reader_att_rep(self, path: str) -> np.array:
        '''
        gets attribute representation from txt file.

        returns a numpy array

        @param path: path to file
        @param att_rep: Numpy matrix with the attribute representation
        '''

        att_rep = np.loadtxt(path, delimiter=',', skiprows=1)
        return att_rep


    ##################################################
    ###################  plot  ######################
    ##################################################

    def plot(self, fig, axis_list, plot_list, metrics_list, activaciones, tgt, pred):
        '''
        Plots the input, and feature maps through the network.
        Deprecated for now.

        returns a numpy array

        @param fig: figure object
        @param axis_list: list with all of the axis. Each axis will represent a feature map
        @param plot_list: list of all of the plots of the feature maps
        @param metrics_list: Matrix with results
        @param tgt: Target class
        @param pred: Predicted class
        '''

        logging.info('        Network_User:    Plotting')
        if self.config['plotting']:
            #Plot

            for an, act in enumerate(activaciones):
                X = np.arange(0, act.shape[1])
                Y = np.arange(0, act.shape[0])
                X, Y = np.meshgrid(X, Y)

                axis_list[an * 2].plot_surface(X, Y, act, cmap=cm.coolwarm, linewidth=1, antialiased=False)

                axis_list[an * 2].set_title('Target {} and Pred {}'.format(tgt, pred))
                axis_list[an * 2].set_xlim3d(X.min(), X.max())
                axis_list[an * 2].set_xlabel('Sensor')
                axis_list[an * 2].set_ylim3d(Y.min(), Y.max())
                axis_list[an * 2].set_ylabel('Time')
                axis_list[an * 2].set_zlim3d(act.min(), act.max())
                axis_list[an * 2].set_zlabel('Measurement')


            for pl in range(len(metrics_list)):
                plot_list[pl].set_ydata(metrics_list[pl])
                plot_list[pl].set_xdata(range(len(metrics_list[pl])))


            '''      
                
            plot_list[0].set_ydata(metrics_list[0])
            plot_list[0].set_xdata(range(len(metrics_list[0])))
            
            plot_list[1].set_ydata(metrics_list[1])
            plot_list[1].set_xdata(range(len(metrics_list[1])))
            
            plot_list[2].set_ydata(metrics_list[2])
            plot_list[2].set_xdata(range(len(metrics_list[2])))
            
            plot_list[3].set_ydata(metrics_list[3])
            plot_list[3].set_xdata(range(len(metrics_list[3])))
            
            '''

            axis_list[1].relim()
            axis_list[1].autoscale_view()
            axis_list[1].legend(loc='best')

            axis_list[3].relim()
            axis_list[3].autoscale_view()
            axis_list[3].legend(loc='best')

            axis_list[5].relim()
            axis_list[5].autoscale_view()
            axis_list[5].legend(loc='best')

            axis_list[7].relim()
            axis_list[7].autoscale_view()
            axis_list[7].legend(loc='best')

            fig.canvas.draw()
            plt.savefig(self.config['folder_exp'] + 'training.png')
            #plt.show()
            plt.pause(0.2)
            axis_list[0].cla()
            axis_list[2].cla()
            axis_list[4].cla()
            axis_list[6].cla()
            axis_list[8].cla()

        return



    ##################################################
    ################  load_weights  ##################
    ##################################################

    def load_weights(self, network):
        '''
        Load weights from a trained network

        @param network: target network with orthonormal initialisation
        @return network: network with transferred CNN layers
        '''
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Loading Weights')

        #print(torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict'])

        # Selects the source network according to configuration
        pretrained_dict = torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict']
        logging.info('        Network_User:        Pretrained model loaded')

        #for k, v in pretrained_dict.items():
        #    print(k)

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        #print(pretrained_dict)

        logging.info('        Network_User:        Pretrained layers selected')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info('        Network_User:        Pretrained layers selected')
        # 3. load the new state dict
        network.load_state_dict(model_dict)
        logging.info('        Network_User:        Weights loaded')

        return network


    ##################################################
    ############  set_required_grad  #################
    ##################################################

    def set_required_grad(self, network):
        '''
        Setting the computing of the gradients for some layers as False
        This will act as the freezing of layers

        @param network: target network
        @return network: network with frozen layers
        '''

        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Setting Required_grad to Weights')

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight', 'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']

        for pn, pv in network.named_parameters():
            if pn in list_layers:
                pv.requires_grad = False

        return network



    ##################################################
    ###################  train  ######################
    ##################################################


    def train(self, ea_itera):
        '''
        Training and validating a network

        @param ea_itera: evolution iteration
        @return results_val: dict with validation results
        @return best_itera: best iteration when validating
        '''

        logging.info('        Network_User: Train---->')

        logging.info('        Network_User:     Creating Dataloader---->')
        #if self.config["dataset"] == "mbientlab":
        #    harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train_{}.csv".format(self.config["percentages_names"]),
        #                                  root_dir=self.config['dataset_root'])
        #else:

        # Selecting the training sets, either train or train final (train  + Validation)
        if self.config['usage_modus'] == 'train':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                          root_dir=self.config['dataset_root'])
        elif self.config['usage_modus'] == 'train_final':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                         root_dir=self.config['dataset_root'])
        elif self.config['usage_modus'] == 'fine_tuning':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                         root_dir=self.config['dataset_root'])

        # Creating the dataloader
        dataLoader_train = DataLoader(harwindows_train, batch_size=self.config['batch_size_train'], shuffle=True)

        # Setting the network
        logging.info('        Network_User:    Train:    creating network')
        if self.config['network'] == 'cnn' or self.config['network'] == 'cnn_imu':
            network_obj = Network(self.config)
            network_obj.init_weights()

            # IF finetuning, load the weights from a source dataset
            if self.config["usage_modus"] == "fine_tuning":
                network_obj = self.load_weights(network_obj)

            # Displaying size of tensors
            logging.info('        Network_User:    Train:    network layers')
            for l in list(network_obj.named_parameters()):
                logging.info('        Network_User:    Train:    {} : {}'.format(l[0], l[1].detach().numpy().shape))

            logging.info('        Network_User:    Train:    setting device')
            network_obj.to(self.device)

        # Setting loss
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Train:    setting criterion optimizer Softmax')
            if self.config["fully_convolutional"] == "FCN":
                criterion = nn.CrossEntropyLoss()
            elif self.config["fully_convolutional"] == "FC":
                criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Train:    setting criterion optimizer Attribute')
            if self.config["fully_convolutional"] == "FCN":
                criterion = nn.BCELoss()
            elif self.config["fully_convolutional"] == "FC":
                criterion = nn.BCELoss()

        # Setting the freezing or not freezing from conv layers
        if self.config['freeze_options']:
            network_obj = self.set_required_grad(network_obj)

        # Setting optimizer
        optimizer = optim.RMSprop(network_obj.parameters(), lr=self.config['lr'], alpha=0.95)

        # Setting scheduler
        step_lr = self.config['epochs'] / 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(step_lr), gamma=0.1)

        if self.config['plotting']:
            # Plotting the input or feature maps through the network.
            # For now deprecated, as features for all the layers arent stored.

            logging.info('        Network_User:    Train:    setting plotting objects')

            fig = plt.figure(figsize=(16, 12), dpi=160)
            axis_list = []
            axis_list.append(fig.add_subplot(521, projection='3d'))
            axis_list.append(fig.add_subplot(522))
            axis_list.append(fig.add_subplot(523, projection='3d'))
            axis_list.append(fig.add_subplot(524))
            axis_list.append(fig.add_subplot(525, projection='3d'))
            axis_list.append(fig.add_subplot(526))
            axis_list.append(fig.add_subplot(527, projection='3d'))
            axis_list.append(fig.add_subplot(528))
            axis_list.append(fig.add_subplot(529, projection='3d'))

            plot_list = []
            # loss_plot, = axis_list[1].plot([], [],'-r',label='red')
            # plots acc, f1w, f1m for training
            plot_list.append(axis_list[1].plot([], [],'-r',label='acc')[0])
            plot_list.append(axis_list[1].plot([], [],'-b',label='f1w')[0])
            plot_list.append(axis_list[1].plot([], [],'-g',label='f1m')[0])

            # plot loss training
            plot_list.append(axis_list[3].plot([], [],'-r',label='loss tr')[0])

            # plots acc, f1w, f1m for training and validation
            plot_list.append(axis_list[5].plot([], [],'-r',label='acc tr')[0])
            plot_list.append(axis_list[5].plot([], [],'-b',label='f1w tr')[0])
            plot_list.append(axis_list[5].plot([], [],'-g',label='f1m tr')[0])

            plot_list.append(axis_list[5].plot([], [],'-c',label='acc vl')[0])
            plot_list.append(axis_list[5].plot([], [],'-m',label='f1w vl')[0])
            plot_list.append(axis_list[5].plot([], [],'-y',label='f1m vl')[0])

            # plot loss for training and validation
            plot_list.append(axis_list[7].plot([], [],'-r',label='loss tr')[0])
            plot_list.append(axis_list[7].plot([], [],'-b',label='loss vl')[0])

            # Customize the z axis.
            for al in range(len(axis_list)):
                if al%2 ==0:
                    axis_list[al].set_zlim(0.0, 1.0)
                    axis_list[al].zaxis.set_major_locator(LinearLocator(10))
                    axis_list[al].zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Initializing lists for plots and results
        losses_train = []
        accs_train = []
        f1w_train = []
        f1m_train = []

        losses_val = []
        accs_val = []
        f1w_val = []
        f1m_val = []

        loss_train_val = []
        accs_train_val = []
        f1w_train_val = []
        f1m_train_val = []

        best_acc_val = 0

        # initialising object for computing metrics
        metrics_obj = Metrics(self.config, self.device, self.attrs)

        itera = 0
        start_time_train = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        best_itera = 0

        # loop for training
        # Validation is not carried out per epoch, but after certain number of iterations, specified
        # in configuration in main
        for e in range(self.config['epochs']):
            start_time_train = time.time()
            logging.info('\n        Network_User:    Train:    Training epoch {}'.format(e))
            start_time_batch = time.time()

            #loop per batch:
            for b, harwindow_batched in enumerate(dataLoader_train):
                start_time_batch = time.time()
                sys.stdout.write("\rTraining: Epoch {}/{} Batch {}/{} and itera {}".format(e,
                                                                                           self.config['epochs'],
                                                                                           b,
                                                                                           len(dataLoader_train),
                                                                                           itera))
                sys.stdout.flush()

                #Setting the network to train mode
                network_obj.train(mode=True)

                #Counting iterations
                itera = (e * harwindow_batched["data"].shape[0]) + b

                #Selecting batch
                train_batch_v = harwindow_batched["data"]
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 0]
                        train_batch_l = train_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        train_batch_l = harwindow_batched["label"][:, 0]
                        train_batch_l = train_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 1:]
                    elif self.config["fully_convolutional"] == "FC":
                        train_batch_l = harwindow_batched["label"]

                # Adding gaussian noise
                noise = self.normal.sample((train_batch_v.size()))
                noise = noise.reshape(train_batch_v.size())
                noise = noise.to(self.device, dtype=torch.float)

                # Sending to GPU
                train_batch_v = train_batch_v.to(self.device, dtype=torch.float)
                train_batch_v += noise
                if self.config['output'] == 'softmax':
                    train_batch_l = train_batch_l.to(self.device, dtype=torch.long) #labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    train_batch_l = train_batch_l.to(self.device, dtype=torch.float) #labels for binerycrossentropy needs float type

                # forward + backward + optimize
                if self.config["dataset"] == "virtual_quarter" or self.config["dataset"] == "mocap_quarter" \
                        or self.config["dataset"] == "mbientlab_quarter":
                    idx_frequency = np.arange(0, 100, 4)
                    train_batch_v = train_batch_v[:, :, idx_frequency, :]
                feature_maps = network_obj(train_batch_v)
                if self.config["fully_convolutional"] == "FCN":
                    feature_maps = feature_maps.reshape(-1, feature_maps.size()[2])
                if self.config['output'] == 'softmax':
                    loss = criterion(feature_maps, train_batch_l) * (1 / self.config['accumulation_steps'])
                elif self.config['output'] == 'attribute':
                    loss = criterion(feature_maps, train_batch_l[:, 1:]) * (1 / self.config['accumulation_steps'])
                loss.backward()

                if (itera + 1) % self.config['accumulation_steps'] == 0:
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                loss_train = loss.item()

                elapsed_time_batch = time.time() - start_time_batch

                ################################## Validating ##################################################

                if (itera + 1) % self.config['valid_show'] == 0 or \
                        (itera + 1) == (self.config['epochs'] * harwindow_batched["data"].shape[0]):
                    logging.info('\n')
                    logging.info('        Network_User:        Validating')
                    start_time_val = time.time()

                    #Setting the network to eval mode
                    network_obj.eval()

                    # Metrics for training for keeping the same number of metrics for val and training
                    # Metrics return a dict with the metrics.
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)
                    loss_train_val.append(loss_train)
                    accs_train_val.append(results_train['acc'])
                    f1w_train_val.append(results_train['f1_weighted'])
                    f1m_train_val.append(results_train['f1_mean'])

                    # Validation
                    # Calling the val() function with the current network and criterion
                    del train_batch_v, noise
                    results_val, loss_val = self.validate(network_obj, criterion)

                    elapsed_time_val = time.time() - start_time_val

                    # Appending the val metrics
                    losses_val.append(loss_val)
                    accs_val.append(results_val['acc'])
                    f1w_val.append(results_val['f1_weighted'])
                    f1m_val.append(results_val['f1_mean'])

                    # print statistics
                    logging.info('\n')
                    logging.info(
                        '        Network_User:        Validating:    '
                        'epoch {} batch {} itera {} elapsed time {}, best itera {}'.format(e, b, itera,
                                                                                           elapsed_time_val,
                                                                                           best_itera))
                    logging.info(
                        '        Network_User:        Validating:    '
                        'acc {}, f1_weighted {}, f1_mean {}'.format(results_val['acc'], results_val['f1_weighted'],
                                                                    results_val['f1_mean']))
                    # Saving the network for the best iteration accuracy
                    if results_val['acc'] > best_acc_val:
                        network_config = {
                            'NB_sensor_channels': self.config['NB_sensor_channels'],
                            'sliding_window_length': self.config['sliding_window_length'],
                            'filter_size': self.config['filter_size'],
                            'num_filters': self.config['num_filters'],
                            'reshape_input': self.config['reshape_input'],
                            'network': self.config['network'],
                            'output': self.config['output'],
                            'num_classes': self.config['num_classes'],
                            'num_attributes': self.config['num_attributes'],
                            'fully_convolutional': self.config['fully_convolutional'],
                            'labeltype': self.config['labeltype']
                        }

                        logging.info('        Network_User:            Saving the network')

                        torch.save({'state_dict': network_obj.state_dict(),
                                    'network_config': network_config,
                                    'att_rep': self.attr_representation},
                                   self.config['folder_exp'] + 'network.pt')
                        best_acc_val = results_val['acc']
                        best_itera = itera

                # Computing metrics for current training batch
                if (itera) % self.config['train_show'] == 0:
                    # Metrics for training
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)

                    activaciones = []
                    metrics_list = []
                    accs_train.append(results_train['acc'])
                    f1w_train.append(results_train['f1_weighted'])
                    f1m_train.append(results_train['f1_mean'])
                    losses_train.append(loss_train)

                    # Plotting for now deprecated
                    if self.config['plotting']:
                        #For plotting
                        metrics_list.append(accs_train)
                        metrics_list.append(f1w_train)
                        metrics_list.append(f1m_train)
                        metrics_list.append(losses_train)
                        metrics_list.append(accs_train_val)
                        metrics_list.append(f1w_train_val)
                        metrics_list.append(f1m_train_val)
                        metrics_list.append(accs_val)
                        metrics_list.append(f1w_val)
                        metrics_list.append(f1m_val)
                        metrics_list.append(loss_train_val)
                        metrics_list.append(losses_val)
                        activaciones.append(train_batch_v.to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[0].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[1].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[2].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[3].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        self.plot(fig, axis_list, plot_list, metrics_list, activaciones,
                                  harwindow_batched["label"][:, 0][0].item(),
                                  torch.argmax(feature_maps[0], dim=0).item())

                    # print statistics
                    logging.info('        Network_User:            Dataset {} network {} lr {} '
                                 'lr_optimizer {} Reshape {} '.format(self.config["dataset"], self.config["network"],
                                                                      self.config["lr"],
                                                                      optimizer.param_groups[0]['lr'],
                                                                      self.config["reshape_input"]))
                    logging.info(
                        '        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} '
                        'elapsed time {} best itera {}'.format(e, self.config['epochs'], b, len(dataLoader_train),
                                                               itera, elapsed_time_batch, best_itera))
                    logging.info('        Network_User:    Train:    loss {}'.format(loss))
                    logging.info(
                        '        Network_User:    Train:    acc {}, '
                        'f1_weighted {}, f1_mean {}'.format(results_train['acc'], results_train['f1_weighted'],
                                                            results_train['f1_mean']))
                    logging.info(
                        '        Network_User:    Train:    '
                        'Allocated {} GB Cached {} GB'.format(round(torch.cuda.memory_allocated(0)/1024**3, 1),
                                                              round(torch.cuda.memory_cached(0)/1024**3, 1)))
                    logging.info('\n\n--------------------------')
            #Step of the scheduler
            scheduler.step()

        elapsed_time_train = time.time() - start_time_train

        logging.info('\n')
        logging.info(
            '        Network_User:    Train:    epoch {} batch {} itera {} '
            'Total training time {}'.format(e, b, itera, elapsed_time_train))

        # Storing the acc, f1s and losses of training and validation for the current training run
        np.savetxt(self.config['folder_exp'] + 'plots/acc_train.txt', accs_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_train.txt', f1m_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_train.txt', f1w_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_train.txt', loss_train_val, delimiter=",", fmt='%s')

        np.savetxt(self.config['folder_exp'] + 'plots/acc_val.txt', accs_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_val.txt', f1m_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_val.txt', f1w_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_val.txt', losses_val, delimiter=",", fmt='%s')

        del losses_train, accs_train, f1w_train, f1m_train
        del losses_val, accs_val, f1w_val, f1m_val
        del loss_train_val, accs_train_val, f1w_train_val, f1m_train_val
        del metrics_list, feature_maps
        del network_obj

        torch.cuda.empty_cache()

        if self.config['plotting']:
            plt.savefig(self.config['folder_exp'] + 'training_final.png')
            plt.close()

        return results_val, best_itera


    ##################################################
    ################  Validate  ######################
    ##################################################

    def validate(self, network_obj, criterion):
        '''
        Validating a network

        @param network_obj: network object
        @param criterion: torch criterion object
        @return results_val: dict with validation results
        @return loss: loss of the validation
        '''

        # Setting validation set and dataloader
        harwindows_val = HARWindows(csv_file=self.config['dataset_root'] + "val.csv",
                                    root_dir=self.config['dataset_root'])

        dataLoader_val = DataLoader(harwindows_val, batch_size=self.config['batch_size_val'])

        # Setting the network to eval mode
        network_obj.eval()

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs)
        loss_val = 0

        # One doesnt need the gradients
        with torch.no_grad():
            for v, harwindow_batched_val in enumerate(dataLoader_val):
                # Selecting batch
                test_batch_v = harwindow_batched_val["data"]
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_val["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_val["label"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_val["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_val["label"]

                # Creating torch tensors
                # test_batch_v = torch.from_numpy(test_batch_v)
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    # labels for crossentropy needs long type
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long)
                elif self.config['output'] == 'attribute':
                    # labels for binerycrossentropy needs float type
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)
                    # labels for crossentropy needs long type

                # forward
                if self.config["dataset"] == "virtual_quarter" or self.config["dataset"] == "mocap_quarter" or\
                        self.config["dataset"] == "mbientlab_quarter":
                    idx_frequency = np.arange(0, 100, 4)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                predictions = network_obj(test_batch_v)
                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                elif self.config['output'] == 'attribute':
                    loss = criterion(predictions, test_batch_l[:, 1:])
                loss_val = loss_val + loss.item()

                # Concatenating all of the batches for computing the metrics
                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    predictions_val = predictions
                    if self.config['output'] == 'softmax':
                        test_labels = harwindow_batched_val["label"][:, 0]
                        test_labels = test_labels.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels = harwindow_batched_val["label"]
                else:
                    predictions_val = torch.cat((predictions_val, predictions), dim=0)
                    if self.config['output'] == 'softmax':
                        test_labels_batch = harwindow_batched_val["label"][:, 0]
                        test_labels_batch = test_labels_batch.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels_batch = harwindow_batched_val["label"]
                    test_labels = torch.cat((test_labels, test_labels_batch), dim=0)

                sys.stdout.write("\rValidating: Batch  {}/{}".format(v, len(dataLoader_val)))
                sys.stdout.flush()

        print("\n")
        # Computing metrics of validation
        test_labels = test_labels.to(self.device, dtype=torch.float)
        results_val = metrics_obj.metric(test_labels, predictions_val)

        del test_batch_v, test_batch_l
        del predictions, predictions_val
        del test_labels_batch, test_labels

        torch.cuda.empty_cache()

        return results_val, loss_val / v




    ##################################################
    ###################  test  ######################
    ##################################################

    def test(self, ea_itera):
        '''
        Testing a network

        @param ea_itera: evolution iteration
        @return results_test: dict with testing results
        @return confusion matrix: confusion matrix of testing results
        '''
        logging.info('        Network_User:    Test ---->')

        # Setting the testing set
        logging.info('        Network_User:     Creating Dataloader---->')
        harwindows_test = HARWindows(csv_file=self.config['dataset_root'] + "test.csv",
                                     root_dir=self.config['dataset_root'])

        dataLoader_test = DataLoader(harwindows_test, batch_size=self.config['batch_size_train'], shuffle=False)

        # Creating a network and loading the weights for testing
        # network is loaded from saved file in the folder of experiment
        logging.info('        Network_User:    Test:    creating network')
        if self.config['network'] == 'cnn' or self.config['network'] == 'cnn_imu':
            network_obj = Network(self.config)

            #Loading the model
            network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt')['state_dict'])
            network_obj.eval()

            logging.info('        Network_User:    Test:    setting device')
            network_obj.to(self.device)

        # Setting loss, only for being measured. Network wont be trained
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Test:    setting criterion optimizer Softmax')
            criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Test:    setting criterion optimizer Attribute')
            criterion = nn.BCELoss()

        loss_test = 0

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs)

        logging.info('        Network_User:    Testing')
        start_time_test = time.time()
        # loop for testing
        with torch.no_grad():
            for v, harwindow_batched_test in enumerate(dataLoader_test):
                #Selecting batch
                test_batch_v = harwindow_batched_test["data"]
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_test["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_test["label"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_test["labels"]
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_test["label"]

                # Sending to GPU
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long)
                    # labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    # labels for binerycrossentropy needs float type
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)

                #forward
                if self.config["dataset"] == "virtual_quarter" or self.config["dataset"] == "mocap_quarter" or \
                        self.config["dataset"] == "mbientlab_quarter":
                    idx_frequency = np.arange(0, 100, 4)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                predictions = network_obj(test_batch_v)
                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                elif self.config['output'] == 'attribute':
                    loss = criterion(predictions, test_batch_l[:, 1:])
                loss_test = loss_test + loss.item()

                # Summing the loss
                loss_test = loss_test + loss.item()

                # Concatenating all of the batches for computing the metrics for the entire testing set
                # and not only for a batch
                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    predictions_test = predictions
                    if self.config['output'] == 'softmax':
                        test_labels = harwindow_batched_test["label"][:, 0]
                        test_labels = test_labels.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels = harwindow_batched_test["label"]
                else:
                    predictions_test = torch.cat((predictions_test, predictions), dim=0)
                    if self.config['output'] == 'softmax':
                        test_labels_batch = harwindow_batched_test["label"][:, 0]
                        test_labels_batch = test_labels_batch.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels_batch = harwindow_batched_test["label"]
                    test_labels = torch.cat((test_labels, test_labels_batch), dim=0)

                sys.stdout.write("\rTesting: Batch  {}/{}".format(v, len(dataLoader_test)))
                sys.stdout.flush()

        elapsed_time_test = time.time() - start_time_test

        #Computing metrics for the entire testing set
        test_labels = test_labels.to(self.device, dtype=torch.float)
        logging.info('            Train:    type targets vector: {}'.format(test_labels.type()))
        results_test = metrics_obj.metric(test_labels, predictions_test)

        # print statistics
        logging.info(
            '        Network_User:        Testing:    elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(
                elapsed_time_test, results_test['acc'], results_test['f1_weighted'], results_test['f1_mean']))

        #predictions_labels = torch.argmax(predictions_test, dim=1)
        predictions_labels = results_test['predicted_classes'].to("cpu", torch.double).numpy()
        test_labels = test_labels.to("cpu", torch.double).numpy()
        if self.config['output'] == 'softmax':
            test_labels = test_labels
        elif self.config['output'] == 'attribute':
            test_labels = test_labels[:, 0]

        # Computing confusion matrix
        confusion_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']))
        for cl in range(self.config['num_classes']):
            pos_tg = test_labels == cl
            pos_pred = predictions_labels[pos_tg]
            bincount = np.bincount(pos_pred.astype(int), minlength=self.config['num_classes'])
            confusion_matrix[cl, :] = bincount

        logging.info("        Network_User:        Testing:    Confusion matrix \n{}\n".format(confusion_matrix.astype(int)))

        percentage_pred = []
        for cl in range(self.config['num_classes']):
            pos_trg = np.reshape(test_labels, newshape=test_labels.shape[0]) == cl
            percentage_pred.append(confusion_matrix[cl, cl] / float(np.sum(pos_trg)))
        percentage_pred = np.array(percentage_pred)

        logging.info("        Network_User:        Validating:    percentage Pred \n{}\n".format(percentage_pred))

        #plot predictions
        if self.config["plotting"]:
            fig = plt.figure()
            axis_test = fig.add_subplot(111)
            plot_trg = axis_test.plot([], [],'-r',label='trg')[0]
            #plot_pred = axis_test.plot([], [],'-b',label='pred')[0]

            plot_trg.set_ydata(test_labels)
            plot_trg.set_xdata(range(test_labels.shape[0]))

            #plot_pred.set_ydata(predictions_labels)
            #plot_pred.set_xdata(range(predictions_labels.shape[0]))

            axis_test.relim()
            axis_test.autoscale_view()
            axis_test.legend(loc='best')

            fig.canvas.draw()
            plt.pause(2.0)
            axis_test.cla()

        del test_batch_v, test_batch_l
        del predictions, predictions_test
        del test_labels, predictions_labels
        del network_obj

        torch.cuda.empty_cache()

        return results_test, confusion_matrix.astype(int)



    ##################################################
    ############  evolution_evaluation  ##############
    ##################################################

    def evolution_evaluation(self, ea_iter, testing = False):
        '''
        Organises the evolution, training, testing or validating

        @param ea_itera: evolution iteration
        @param testing: Setting testing in training or only testing
        @return results: dict with validating/testing results
        @return confusion_matrix: dict with validating/testing results
        @return best_itera: best iteration for training
        '''

        logging.info('        Network_User: Evolution evaluation iter {}'.format(ea_iter))

        confusion_matrix = 0
        best_itera = 0
        if testing:
            logging.info('        Network_User: Testing')
            results, confusion_matrix = self.test(ea_iter)
        else:
            if self.config['usage_modus'] == 'train':
                logging.info('        Network_User: Training')

                results, best_itera = self.train(ea_iter)

            elif self.config['usage_modus'] == 'evolution':
                logging.info('        Network_User: Evolution')

            elif self.config['usage_modus'] == 'train_final':
                logging.info('        Network_User: Final Training')
                results, best_itera = self.train(ea_iter)

            elif self.config['usage_modus'] == 'fine_tuning':
                logging.info('        Network_User: Fine Tuning')
                results, best_itera = self.train(ea_iter)

            elif self.config['usage_modus'] == 'test':
                logging.info('        Network_User: Testing')

                results, confusion_matrix = self.test(ea_iter)

            else:
                logging.info('        Network_User: Not selected modus')

        return results, confusion_matrix, best_itera
