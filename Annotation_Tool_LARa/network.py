'''
Created on May 17, 2019

@author: Fernando Moya-Rueda
@email: fernando.moya@tu-dortmund.de

'''

from __future__ import print_function
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np



class Network(nn.Module):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        
        super(Network, self).__init__()
        
        logging.info('            Network: Constructor')
        
        self.config = config


        in_channels = 1
        Hx = self.config['NB_sensor_channels']
        Wx = self.config['sliding_window_length']


        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=0, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=0, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=0, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=0, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))

        # set the Conv layers
        if self.config["network"] == "cnn":
            self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=0)
            self.conv1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=0)
            self.conv2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=0)
            self.conv2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=0)

            if self.config["reshape_input"]:
                self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * int(self.config['NB_sensor_channels'] / 3),
                                     256)
            else:
                self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * self.config['NB_sensor_channels'], 256)

        # set the Conv layers
        if self.config["network"] == "cnn_imu":
            # later
            pass
        
        if self.config["network"] == "cnn":
            self.fc4 = nn.Linear(256, 256)
        elif self.config["network"] == "cnn_imu":
            self.fc4 = nn.Linear(256 * 5, 256)

        if self.config['output'] == 'softmax': 
            self.fc5 = nn.Linear(256, self.config['num_classes'])
        elif self.config['output'] == 'attribute':  
            self.fc5 = nn.Linear(256, self.config['num_attributes'])

        self.softmax = nn.Softmax(dim=1)
        
        self.sigmoid = nn.Sigmoid()

        return
    
    

    def forward(self, x):
        if self.config["reshape_input"]:
            x = x.permute(0, 2, 1, 3)
            x = x.view(x.size()[0], x.size()[1], int(x.size()[3] / 3), 3)
            x = x.permute(0, 3, 1, 2)

        if self.config["network"] == "cnn":
            x = F.relu(self.conv1_1(x))
            x = F.relu(self.conv1_2(x))
            #x12 = F.max_pool2d(x12, (2, 1))

            x = F.relu(self.conv2_1(x))
            x = F.relu(self.conv2_2(x))
            # x = F.max_pool2d(x, (2, 1))

            # view is reshape
            x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
            x = F.relu(self.fc3(x))

        if self.config["network"] == "cnn_imu":
            # later
            pass
        
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        
        if self.config['output'] == 'attribute':
            x = self.sigmoid(x)

        if not self.training:
            if self.config['output'] == 'softmax':
                x = self.softmax(x)

        return x
        #return x11.clone(), x12.clone(), x21.clone(), x22.clone(), x
    
    
    
    def init_weights(self):
        self.apply(Network._init_weights_orthonormal)
        
        return
    
    
    
    @staticmethod
    def _init_weights_orthonormal(m):
        if isinstance(m, nn.Conv2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):          
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        
        return

    
    
    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        
        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * P) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * P) / S[1]
        
        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]
                    
        return Wy, Hy