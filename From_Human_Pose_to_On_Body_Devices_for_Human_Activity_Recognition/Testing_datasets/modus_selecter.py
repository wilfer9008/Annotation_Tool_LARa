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

    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter, type_simple = 'training'):
        
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
            acc_train, f1_weighted_train, f1_mean_train = self.network.evolution_evaluation(ea_iter=iter_evl)

            acc_train_ac.append(acc_train)
            f1_weighted_train_ac.append(f1_weighted_train)
            f1_mean_train_ac.append(f1_mean_train)
            
            elapsed_time_test = time.time() - start_time_test
            
            logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(elapsed_time_test,
                                                             acc_train,
                                                             f1_weighted_train,
                                                             f1_mean_train))
            
            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, ea_iter=iter_evl)

            if testing:
                acc_test, f1_weighted_test, f1_mean_test = self.test(testing=True)
                acc_test_ac.append(acc_test)
                f1_weighted_test_ac.append(f1_weighted_test)
                f1_mean_test_ac.append(f1_mean_test)
                self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing')

        if testing:
            self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing')

        return

    def test(self, testing = False):
        
        start_time_test = time.time()
        
        acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=0, testing=testing)
        
        elapsed_time_test = time.time() - start_time_test
        
        logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, acc_test, f1_weighted_test, f1_mean_test))

        if not testing:
            self.save([acc_test], [f1_weighted_test], [f1_mean_test], ea_iter=0, type_simple='testing')
            return
        
        return acc_test, f1_weighted_test, f1_mean_test


    def evolution(self):
        logging.info('    Network_selecter: Evolution')

        # Setting attribute population
        if os.path.isfile('../' + self.config['folder_exp'] + '/iters.txt'):
            best_attrs = self.attributes.load_attrs(0, name_file='best_attrs')
            self.attrs_0 = best_attrs[0]['attrs']
            self.network.set_attrs(self.attrs_0)
            init_iter = self.load_iters() + 1

            logging.info('    Network_selecter:     Loading previous training in iters {}...'.format(init_iter))
        else:
            self.attrs_0 = self.attributes.creating_init_population()
            init_iter = 0
            self.network.set_attrs(self.attrs_0)

            logging.info('    Network_selecter:     No Loading training in iters {}...'.format(init_iter))

        start_time_test = time.time()

        # initial evaluation of the population number 0
        acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=0)

        elapsed_time_test = time.time() - start_time_test

        logging.info(
            '    Network_selecter:     EA: elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(elapsed_time_test,
                                                                                                      acc_test,
                                                                                                      f1_weighted_test,
                                                                                                      f1_mean_test))
        #Save validation results
        self.save(acc_test, f1_weighted_test, f1_mean_test, ea_iter = 0)

        self.attributes.save_attrs(self.attrs_0, f1_weighted_test, init_iter, name_file='attrs')

        #Setting up the fitness
        best_fitness = f1_weighted_test
        best_attr = np.copy(self.attrs_0)

        fitness = []
        all_fitness = []
        all_acc = []
        iters = []

        fitness.append(f1_weighted_test)
        all_fitness.append(f1_weighted_test)
        all_acc.append(acc_test)
        iters.append(init_iter)

        np.savetxt(self.config["folder_exp"] + 'fitness.txt', fitness, fmt='%.5f')
        np.savetxt(self.config["folder_exp"] + 'iters.txt', iters, fmt='%d')
        np.savetxt(self.config["folder_exp"] + 'best_attributes.txt', best_attr, fmt='%d')
        np.savetxt(self.config["folder_exp"] + 'all_fitness.txt', all_fitness, fmt='%.5f')
        np.savetxt(self.config["folder_exp"] + 'all_accuracies.txt', all_acc, fmt='%.5f')


        # Starting the evolution
        epochs_training = self.config["epochs"]
        for ea_iter in range(1, self.config["evolution_iter"]):

            logging.info(
                '    Network_selecter:     EA: iter {} from {} with epochs {}...'.format(ea_iter,
                                                                                         self.config["evolution_iter"],
                                                                                         epochs_training))
            #Mutating the attributes
            # attr_new = self.mutation_nonlocal_percentage(best_attr, best_percentage, number_K = 8)
            # attr_new = self.mutation_local(best_attr)
            # attr_new = self.mutation_nonlocal(best_attr, number_K = 4)
            attr_new = self.attributes.mutation_global(best_attr)

            #Setting the new attributes to the network
            self.network.set_attrs(attr_new)

            #training and validating the network
            acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=ea_iter)

            logging.info('    Network_selecter:     EA: elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(
                elapsed_time_test,
                acc_test,
                f1_weighted_test,
                f1_mean_test))

            #Store the fitness
            all_fitness.append(f1_weighted_test)
            np.savetxt(self.config["folder_exp"] + 'all_fitness.txt', all_fitness, fmt='%.5f')

            self.save(acc_test, f1_weighted_test, f1_mean_test, ea_iter=ea_iter)

            all_acc.append(acc_test)
            np.savetxt(self.config["folder_exp"] + 'all_accuracies.txt', all_acc, fmt='%.5f')

            #save the attributes
            self.attributes.save_attrs(attr_new, f1_weighted_test, ea_iter, protocol_file='ab')

            #select if fitness improved, if so, update the fitness and save the network and attributes
            if f1_weighted_test > best_fitness:
                logging.info('    Network_selecter:     EA: Got best attrs with f1{}...'.format(f1_weighted_test))

                best_fitness = f1_weighted_test
                best_attr = np.copy(attr_new)

                fitness.append(f1_weighted_test)
                iters.append(ea_iter)

                #Saving the best attributes and its network
                self.attributes.save_attrs(attr_new, f1_weighted_test, ea_iter, name_file='best_attrs')
                self.network.save_network(ea_iter)

                np.savetxt(self.config["folder_exp"] + 'fitness.txt', fitness, fmt='%.5f')
                np.savetxt(self.config["folder_exp"] + 'iters.txt', iters, fmt='%d')
                np.savetxt(self.config["folder_exp"] + 'best_attributes.txt', best_attr, fmt='%d')

        return



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


