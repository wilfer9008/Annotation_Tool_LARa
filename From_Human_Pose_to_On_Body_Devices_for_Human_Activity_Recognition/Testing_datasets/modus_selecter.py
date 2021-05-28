'''
Created on Feb 27, 2019

@author: fmoya

Old network_selecter with caffe/theano implementations

'''

from __future__ import print_function
import logging

from network_user import Network_User
from attributes import Attributes

import time

import xml.etree.ElementTree as ET
from xml.dom import minidom

import sys
import os

import numpy as np


class Modus_Selecter(object):
    '''
    classdocs
    '''

    def __init__(self, config):
        '''
        Constructor
        '''

        logging.info('    Network_selecter: Constructor')
        self.config = config

        self.network = Network_User(config)
        self.attributes = Attributes(config)
        self.attrs_0 = None

        return

    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter, type_simple='training'):
        """
        Save the results of traiing and testing according to the configuration.
        As training is repeated several times, results are appended, and mean and std of all the repetitions
        are computed.

        @param acc_test: List of accuracies of val or testing
        @param f1_weighted_test: List of F1w of val or testing
        @param f1_mean_test: List of F1m of val or testing
        @param ea_iter: Iteration of evolution
        @param type_simple: Type of experiment
        """
        
        xml_file_path = self.config['folder_exp'] + self.config['file_suffix']

        xml_root = ET.Element("Experiment_{}".format(self.config["name_counter"]))
        child_network = ET.SubElement(xml_root, "network", dataset=str(self.config['network']))
        child_dataset = ET.SubElement(child_network, "dataset", dataset=str(self.config['dataset']))
        child = ET.SubElement(child_dataset, "usage_modus", usage_modus=str(self.config['usage_modus']))
        child = ET.SubElement(child_dataset, "dataset_finetuning",
                              dataset_finetuning=str(self.config['dataset_finetuning']))
        child = ET.SubElement(child_dataset, "percentages",
                              percentages_names=str(self.config['proportions']))
        child = ET.SubElement(child_dataset, "type_simple", type_simple=str(type_simple))
        child = ET.SubElement(child_dataset, "output", output=str(self.config['output']))
        child = ET.SubElement(child_dataset, "lr", lr=str(self.config['lr']))
        child = ET.SubElement(child_dataset, "epochs", epochs=str(self.config['epochs']))
        child = ET.SubElement(child_dataset, "reshape_input", reshape_input=str(self.config["reshape_input"]))
        child = ET.SubElement(child_dataset, "ea_iter", ea_iter=str(ea_iter))
        child = ET.SubElement(child_dataset, "freeze_options", freeze_options=str(self.config['freeze_options']))
        for exp in range(len(acc_test)):
            child = ET.SubElement(child_dataset, "metrics", acc_test=str(acc_test[exp]),
                                  f1_weighted_test=str(f1_weighted_test[exp]),
                                  f1_mean_test=str(f1_mean_test[exp]))
        child = ET.SubElement(child_dataset, "metrics_mean", acc_test_mean=str(np.mean(acc_test)),
                              f1_weighted_test_mean=str(np.mean(f1_weighted_test)),
                              f1_mean_test_mean=str(np.mean(f1_mean_test)))
        child = ET.SubElement(child_dataset, "metrics_std", acc_test_mean=str(np.std(acc_test)),
                              f1_weighted_test_mean=str(np.std(f1_weighted_test)),
                              f1_mean_test_mean=str(np.std(f1_mean_test)))

        xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)
        
        return

    def train(self, itera=1, testing=False):
        """
        Train method. Train network for a certain number of repetitions
        computing the val performance, Testing using test(), saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        """
        
        logging.info('    Network_selecter: Train')
        
        start_time_test = time.time()

        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []

        #There will be only one iteration
        #As there is not evolution
        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []

        for iter_evl in range(itera):
            logging.info('    Network_selecter:    Train iter 0')
            # Training the network and obtaining the validation results
            acc_train, f1_weighted_train, f1_mean_train = self.network.evolution_evaluation(ea_iter=iter_evl)

            # Appending results for later saving in results file
            acc_train_ac.append(acc_train)
            f1_weighted_train_ac.append(f1_weighted_train)
            f1_mean_train_ac.append(f1_mean_train)
            
            elapsed_time_test = time.time() - start_time_test
            
            logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(elapsed_time_test,
                                                             acc_train,
                                                             f1_weighted_train,
                                                             f1_mean_train))

            # Saving the results
            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, ea_iter=iter_evl)

            # Testing the network
            if testing:
                acc_test, f1_weighted_test, f1_mean_test = self.test(testing=True)
                acc_test_ac.append(acc_test)
                f1_weighted_test_ac.append(f1_weighted_test)
                f1_mean_test_ac.append(f1_mean_test)
                self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing')

        if testing:
            self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing')

        return

    def test(self, testing=False):
        """
        Test method. Testing the network , saving the performances

        @param testing: Enabling testing after training
        @return acc_test: accuracy of testing
        @return f1_weighted_test: f1 weighted of testing
        @return f1_mean_test: f1 mean of testing
        """
        
        start_time_test = time.time()

        # Testing the network in folder (according to the conf)
        acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=0, testing=testing)
        
        elapsed_time_test = time.time() - start_time_test
        
        logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, acc_test, f1_weighted_test, f1_mean_test))

        # Saving the results
        if not testing:
            self.save([acc_test], [f1_weighted_test], [f1_mean_test], ea_iter=0, type_simple='testing')
            return
        
        return acc_test, f1_weighted_test, f1_mean_test


    def net_modus(self):
        """
        Setting the training, validation, evolution and final training.
        """
        logging.info('    Network_selecter: Net modus: {}'.format(self.config['usage_modus']))
        if self.config['usage_modus'] == 'train':
            self.train(itera=5, testing=True)
        elif self.config['usage_modus'] == 'test':
            self.test()
        elif self.config['usage_modus'] == 'evolution':
            self.evolution()
        elif self.config['usage_modus'] == 'train_final':
            self.train(itera=1,  testing=True)
        elif self.config['usage_modus'] == 'fine_tuning':
            self.train(itera=3, testing=True)
        return


