'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import logging
import os

from network_user import Network_User
import numpy as np

import time

import xml.etree.ElementTree as ET
from xml.dom import minidom

class Modus_Selecter(object):
    '''
    classdocs
    '''


    def __init__(self, config, exp = None):
        '''
        Constructor
        '''
        
        logging.info('    Network_selecter: Constructor')
        self.config = config
        logging.info('    Network_selecter: \n{}'.format(config))

        self.exp = exp
        self.network = Network_User(config, self.exp)

        return


    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter, type_simple = 'training',
             confusion_matrix=0):

        xml_file_path = self.config['folder_exp'] + self.config['file_suffix']

        xml_root = ET.Element("Experiment_{}".format(self.config["name_counter"]))
        child_network = ET.SubElement(xml_root, "network", dataset=str(self.config['network']))
        child_dataset = ET.SubElement(child_network, "dataset", dataset=str(self.config['dataset']))
        child = ET.SubElement(child_dataset, "usage_modus", usage_modus=str(self.config['usage_modus']))
        child = ET.SubElement(child_dataset, "dataset_finetuning",
                              dataset_finetuning=str(self.config['dataset_finetuning']))
        child = ET.SubElement(child_dataset, "percentages_names",
                              percentages_names=str(self.config['percentages_names']))
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
        child = ET.SubElement(child_dataset, "confusion_matrix_last",
                              confusion_matrix_last=str(confusion_matrix))


        xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)

        return

    def train(self, itera=1, testing=False):

        logging.info('    Network_selecter: Train')

        start_time_test = time.time()

        #There will be only one iteration
        #As there is not evolution
        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []

        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []

        for iter_evl in range(itera):
            logging.info('    Network_selecter:    Train iter 0')
            acc_train, f1_weighted_train, f1_mean_train, _ = self.network.evolution_evaluation(ea_iter=iter_evl)

            acc_train_ac.append(acc_train)
            f1_weighted_train_ac.append(f1_weighted_train)
            f1_mean_train_ac.append(f1_mean_train)

            elapsed_time_test = time.time() - start_time_test

            logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(elapsed_time_test,
                                                             acc_train,
                                                             f1_weighted_train,
                                                             f1_mean_train))

            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Val", value=acc_train)
                self.exp.log_scalar("F1w_Val", value=f1_weighted_train)
                self.exp.log_scalar("F1m_Val", value=f1_mean_train)

            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, ea_iter=iter_evl)

            if testing:
                acc_test, f1_weighted_test, f1_mean_test, confusion_matrix_test = self.test(testing=True)
                acc_test_ac.append(acc_test)
                f1_weighted_test_ac.append(f1_weighted_test)
                f1_mean_test_ac.append(f1_mean_test)

                if self.config["sacred"]:
                    self.exp.log_scalar("Acc_Test", value=acc_test)
                    self.exp.log_scalar("F1w_Test", value=f1_weighted_test)
                    self.exp.log_scalar("F1m_Test", value=f1_mean_test)
                    self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl,
                              type_simple='testing', confusion_matrix=confusion_matrix_test)

        if testing:
            self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing',
                      confusion_matrix=confusion_matrix_test)

        if self.config["usage_modus"] == "train":
            logging.info('    Network_selecter:    Train:    eliminating network file')
            os.remove(self.config['folder_exp'] + 'network.pt')

        return


    def test(self, testing = False):

        start_time_test = time.time()

        acc_test, f1_weighted_test, f1_mean_test, confusion_matrix_test = self.network.evolution_evaluation(ea_iter=0,
                                                                                                            testing=testing)

        elapsed_time_test = time.time() - start_time_test

        logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, acc_test, f1_weighted_test, f1_mean_test))

        if not testing:
            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Test", value=acc_test)
                self.exp.log_scalar("F1w_Test", value=f1_weighted_test)
                self.exp.log_scalar("F1m_Test", value=f1_mean_test)

            self.save([acc_test], [f1_weighted_test], [f1_mean_test], ea_iter=0, type_simple='testing')
            return

        return acc_test, f1_weighted_test, f1_mean_test, confusion_matrix_test


    def net_modus(self):

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
