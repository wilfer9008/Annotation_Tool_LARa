'''
Created on 24.04.2020

@author: Erik
'''
import torch
from network import Network
import global_variables as g

# from cnn_imu import Network as Network_
import os

import numpy as np

# https://pytorch.org/tutorials/beginner/saving_loading_models.html

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")


def network_path(index):
    return g.networks_path + g.networks[index]['file_name']


def add_config(file, key, value, prefix):
    checkpoint = torch.load('..' + os.sep + 'networks' + os.sep + file, map_location=torch.device('cpu'))
    checkpoint['network_config'][key] = value
    torch.save(checkpoint, '..' + os.sep + 'networks' + os.sep + prefix + file)


def check_config_keys(file):
    checkpoint = torch.load('..' + os.sep + 'networks' + os.sep + file, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    print('Network_config:', checkpoint['network_config'])


def check_loading(file, model):
    checkpoint = torch.load('..' + os.sep + 'networks' + os.sep + file, map_location=torch.device('cpu'))

    config = checkpoint['network_config']
    network = model(config)

    state_dict = checkpoint['state_dict']
    network.load_state_dict(state_dict)
    return network


def load_network(index):
    """Loads the selected network"""
    try:
        checkpoint = torch.load(network_path(index), map_location=device)

        state_dict = checkpoint['state_dict']
        config = checkpoint['network_config']
        if 'att_rep' in checkpoint.keys():
            att_rep = checkpoint['att_rep']
        else:
            att_rep = None
        network = Network(config)
        network.load_state_dict(state_dict)
        network.eval()
        return network, config, att_rep

    except KeyError as e:
        network = None
        raise e
    except FileNotFoundError as e:
        raise e


class_network = 'class_network.pt'
attr_network = "attrib_network.pt"
cnn_imu_network = 'cnn_imu_network.pt'
cnn_imu_retrained_network = 'cnn_imu_retrained_network.pt'

# check_config_keys(class_network)
# check_config_keys(attr_network)
# check_config_keys(cnn_imu_network)

# add_config(class_network, "fully_convolutional", "FC",'1')
# add_config(attr_network, "fully_convolutional", "FC",'1')
# add_config(cnn_imu_network, "fully_convolutional", "FC",'1')


"""
check_loading('1'+class_network, Network_)
print("loading class network with Network_ successful")

check_loading('1'+attr_network, Network_)
print("loading attr network with Network_ successful")
"""
# check_loading(cnn_imu_network, Network)
# print("loading cnn imu network with Network successful")


"""
att_rep = np.loadtxt("atts_per_class_dataset.txt",dtype = np.int,delimiter = ",")

checkpoint = torch.load('..'+os.sep+'networks'+os.sep+attr_network, map_location=torch.device('cpu'))
checkpoint['att_rep'] = att_rep
torch.save(checkpoint, '..'+os.sep+'networks'+os.sep+attr_network)

checkpoint = torch.load('..'+os.sep+'networks'+os.sep+cnn_imu_network, map_location=torch.device('cpu'))
checkpoint['att_rep'] = att_rep
torch.save(checkpoint, '..'+os.sep+'networks'+os.sep+cnn_imu_network)

checkpoint = torch.load('..'+os.sep+'networks'+os.sep+cnn_imu_retrained_network, map_location=torch.device('cpu'))
checkpoint['att_rep'] = att_rep
torch.save(checkpoint, '..'+os.sep+'networks'+os.sep+cnn_imu_retrained_network)

att_rep = checkpoint['att_rep']
print(att_rep.shape)
print(type(att_rep))
print(np.unique(att_rep[:,1:]))
"""
check_config_keys(network_path(1))
load_network(1)

check_config_keys(network_path(2))
load_network(2)


load_network(3)
load_network(4)