'''
Created on 27.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''

import os

data = None


settings = {'openFilePath' : '..'+os.sep+'unlabeled',
            'saveFinishedPath':'..'+os.sep+'labeled',
            'floorGrid' : False,
            'dynamicFloor':False,
            'annotatorID':0,
            'fastSpeed':20,
            'normalSpeed':50,
            'slowSpeed':100,
            'segmentationWindowStride':200,
            'deep_threshold':0.05}

settings_path = '..'+os.sep+'settings.json'
networks_path = '..'+os.sep+'networks' + os.sep

networks = {1 : {'name' : 'Class Network',
                 'file_name' : 'class_network.pt',
                 'annotator_id':90,
                 'attributes':False},
            2 : {'name':'Attribute Network',
                 'file_name' : 'attrib_network.pt',
                 'annotator_id':91,
                 'attributes':True},
            3 : {'name':'CNN IMU Network',
                 'file_name' : 'cnn_imu_network.pt',
                 'annotator_id':92,
                 'attributes':True},
            4 : {'name':'CNN IMU Network retrained',
                 'file_name' : 'cnn_imu_retrained_network.pt',
                 'annotator_id':93,
                 'attributes':True},
            }


annotation_guide_link = "https://docs.google.com/document/d/1RNahPI2sCZdx1Iy0Gfp-ALjFgd_e-AKnU78_DubN7iU/edit"

version = "224"
