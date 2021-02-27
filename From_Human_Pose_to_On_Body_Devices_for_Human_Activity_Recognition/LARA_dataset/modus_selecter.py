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

from sacred import Experiment

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

    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter, type_simple='training', confusion_matrix=0,
             time_iter=0, precisions=0, recalls=0, best_itera=0):

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
        child = ET.SubElement(child_dataset, "time_iter", time_iter=str(time_iter))
        child = ET.SubElement(child_dataset, "best_itera", best_itera=str(best_itera))

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
        if type_simple == 'training':
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(precisions))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(recalls))
        else:
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(np.mean(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(np.std(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "recall_mean", recall_mean=str(np.mean(recalls, axis=0)))
            child = ET.SubElement(child_dataset, "recall_std", recall_std=str(np.std(recalls, axis=0)))

        xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)

        return

    def train(self, itera=1, testing=False):

        logging.info('    Network_selecter: Train')

        #There will be only one iteration
        #As there is not evolution
        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []
        precisions_test = []
        recalls_test = []


        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []

        for iter_evl in range(itera):
            start_time_train = time.time()
            logging.info('    Network_selecter:    Train iter 0')
            #acc_train, f1_weighted_train, f1_mean_train, _ = self.network.evolution_evaluation(ea_iter=iter_evl)
            results_train, confusion_matrix_train, best_itera = self.network.evolution_evaluation(ea_iter=iter_evl)

            acc_train_ac.append(results_train['acc'])
            f1_weighted_train_ac.append(results_train['f1_weighted'])
            f1_mean_train_ac.append(results_train['f1_mean'])

            time_train = time.time() - start_time_train

            logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(time_train,
                                                             results_train['acc'],
                                                             results_train['f1_weighted'],
                                                             results_train['f1_mean']))

            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Val", value=results_train['acc'])
                self.exp.log_scalar("F1w_Val", value=results_train['f1_weighted'])
                self.exp.log_scalar("F1m_Val", value=results_train['f1_mean'])

            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, ea_iter=iter_evl,
                      time_iter=time_train, precisions=results_train['precision'], recalls=results_train['recall'],
                      best_itera=best_itera)

            if testing:
                start_time_test = time.time()
                results_test, confusion_matrix_test = self.test(testing=True)
                acc_test_ac.append(results_test['acc'])
                f1_weighted_test_ac.append(results_test['f1_weighted'])
                f1_mean_test_ac.append(results_test['f1_mean'])
                precisions_test.append(results_test['precision'].numpy())
                recalls_test.append(results_test['recall'].numpy())

                time_test = time.time() - start_time_test

                if self.config["sacred"]:
                    self.exp.log_scalar("Acc_Test", value=results_test['acc'])
                    self.exp.log_scalar("F1w_Test", value=results_test['f1_weighted'])
                    self.exp.log_scalar("F1m_Test", value=results_test['f1_mean'])
                    self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl,
                              type_simple='testing', confusion_matrix=confusion_matrix_test, time_iter=time_test,
                              results=results_test)

        if testing:
            self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing',
                      confusion_matrix=confusion_matrix_test, time_iter=time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test))

        if self.config["usage_modus"] == "train":
            logging.info('    Network_selecter:    Train:    eliminating network file')
            os.remove(self.config['folder_exp'] + 'network.pt')

        return


    def test(self, testing = False):

        start_time_test = time.time()
        precisions_test = []
        recalls_test = []

        results_test, confusion_matrix_test, _ = self.network.evolution_evaluation(ea_iter=0, testing=testing)

        elapsed_time_test = time.time() - start_time_test

        precisions_test.append(results_test['precision'].numpy())
        recalls_test.append(results_test['recall'].numpy())

        logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, results_test['acc'],
                                                         results_test['f1_weighted'], results_test['f1_mean']))

        if not testing:
            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Test", value=results_test['acc'])
                self.exp.log_scalar("F1w_Test", value=results_test['f1_weighted'])
                self.exp.log_scalar("F1m_Test", value=results_test['f1_mean'])

            self.save([results_test['acc']], [results_test['f1_weighted']], [results_test['f1_mean']],
                      ea_iter=0, type_simple='testing', confusion_matrix=confusion_matrix_test,
                      time_iter=elapsed_time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test))
            return

        return results_test, confusion_matrix_test


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
            self.train(itera=5, testing=True)
        return
