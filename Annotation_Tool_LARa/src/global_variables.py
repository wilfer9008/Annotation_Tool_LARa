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

networks = {1: {'name': 'Class Network',
                'file_name': 'class_network.pt',
                'annotator_id': 90,
                'attributes': False},
            2: {'name': 'Attribute Network',
                'file_name': 'attrib_network.pt',
                'annotator_id': 91,
                'attributes': True},
            3: {'name': 'CNN IMU Network',
                'file_name': 'cnn_imu_network.pt',
                'annotator_id': 92,
                'attributes': True},
            4: {'name': 'CNN IMU Network retrained',
                'file_name': 'cnn_imu_retrained_network.pt',
                'annotator_id': 93,
                'attributes': True},
            5: {'name': 'CNN IMU w50 s12',
                'file_name': 'network_w50_s12.pt',
                'annotator_id': 94,
                'attributes': True},
            6: {'name': 'CNN IMU w100 s25',
                'file_name': 'network_w100_s25.pt',
                'annotator_id': 95,
                'attributes': True},
            7: {'name': 'CNN IMU w150 s25',
                'file_name': 'network_w150_s25.pt',
                'annotator_id': 96,
                'attributes': True}
            }

annotation_guide_link = "https://docs.google.com/document/d/1RNahPI2sCZdx1Iy0Gfp-ALjFgd_e-AKnU78_DubN7iU/edit"
network_download_link = "https://tu-dortmund.sciebo.de/s/YkpqlYOffFrmFr0"

version = "230"
