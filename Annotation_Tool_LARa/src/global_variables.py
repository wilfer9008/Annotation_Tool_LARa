"""
Created on 27.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""

import os

import numpy as np

from data_management import DataProcessor, WindowProcessor, RetrievalData

data: DataProcessor = None
windows: WindowProcessor = None
retrieval: RetrievalData = None

settings = {'openFilePath': '..' + os.sep + 'unlabeled',
            'saveFinishedPath': '..' + os.sep + 'labeled',
            'stateFinishedPath': '..' + os.sep + 'states',
            'backUpPath': '..' + os.sep + 'backups',
            'floorGrid': False,
            'dynamicFloor': False,
            'annotator_id': 0,
            'fastSpeed': 20,
            'normalSpeed': 50,
            'slowSpeed': 100,
            'segmentationWindowStride': 200,
            'deep_threshold': 0.05}

settings_path = '..' + os.sep + 'settings.json'

with open(f'..{os.sep}labels{os.sep}class.txt', 'r') as f:
    classes = f.read().split(',')

with open(f'..{os.sep}labels{os.sep}attrib.txt', 'r') as f:
    attributes = f.read().split(',')

attribute_rep = np.loadtxt(f"..{os.sep}labels{os.sep}atts_per_class_dataset.txt", delimiter=",")

states = None


def get_states(file_name):
    global states
    scenario = file_name.split("_")[0]  # Lxy
    with open(f'..{os.sep}labels{os.sep}states{scenario}.txt', 'r') as file:
        states = file.read().split(',')
        #print(states)


networks_path = '..' + os.sep + 'networks' + os.sep

networks = {1: {'name': 'tCNN_softmax',
                'file_name': 'cnn_classes_network.pt',
                'annotator_id': 90,
                'attributes': False},
            2: {'name': 'tCNN_attribute',
                'file_name': 'cnn_attrib_network.pt',
                'annotator_id': 91,
                'attributes': True},
            3: {'name': 'tCNN-IMU_softmax',
                'file_name': 'cnn_imu_classes_network.pt',
                'annotator_id': 92,
                'attributes': False},
            4: {'name': 'tCNN-IMU_attrib',
                'file_name': 'cnn_imu_attrib_network.pt',
                'annotator_id': 93,
                'attributes': True},
            }

annotation_guide_link = "https://docs.google.com/document/d/1RNahPI2sCZdx1Iy0Gfp-ALjFgd_e-AKnU78_DubN7iU/edit"
network_download_link = "https://tu-dortmund.sciebo.de/s/YkpqlYOffFrmFr0"

version = "231"
