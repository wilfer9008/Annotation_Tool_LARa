'''
Created on Jan 27, 2020

@author: fmoya
'''

import os
import numpy as np
import torch
import torch.utils.data as data
import logging
from augmentations import ActivityAugmentation

from sliding_window import sliding_window
from resampling import Resampling


class OrderPicking(data.Dataset):


    def __init__(self, config, partition_modus = 'train'):
        self.config = config
        self.partition_modus = partition_modus

        self.X, self.y = self.load_data()
        #self.X, self.y, self.Y = self.opp_sliding_window(self.X, self.Y)

        self.X = np.reshape(self.X, [self.X.shape[0], 1, self.X.shape[1], self.X.shape[2]])
        self.X = self.X.astype(np.float32)

        return

    def __getitem__(self, idx):
        window_data = {"data": self.X[idx], "label": self.y[idx], "labels": self.y[idx]}
        return window_data

    def __len__(self):
        return self.X.shape[0]

    def get_labels(self):
        labels_dict = {0: "NULL", 1: "UNKNOWN", 2: "FLIP", 3: "WALK",
                       4: "SEARCH", 5: "PICK", 6: "SCAN", 7: "INFO",
                       8: "COUNT", 9: "CARRY", 10: "ACK"}

        return labels_dict

    def load_data(self, wr='_DO', test_id=3, train_or_test=False, all_labels=False):
        '''
        Loads image (np array) paths and annos
        '''
        dictz = {"_DO": {1: "004", 2: "011", 3: "017"}, "_NP": {1: "004", 2: "014", 3: "015"}}
        logging.info("Data: Load data for dataset: wr {}; test person {}".format(wr, test_id))

        train_ids = list((dictz[wr]).keys())
        train_ids.remove(test_id)
        train_list = [self.config['dataset_directory'] + "%s__%s_data_labels_every-frame_100.npz" % (
        wr, dictz[wr][train_ids[i]]) for i in [0, 1]]
        test_list = [self.config["dataset_directory"] +
                     "%s__%s_data_labels_every-frame_100.npz" % (wr, dictz[wr][test_id])]

        if self.partition_modus == 'train':
            set_list = train_list
        elif self.partition_modus == 'val' or self.partition_modus == 'test':
            set_list = test_list
        else:
            raise("        Dataloader: Error list set")

        train_vals = []
        train_labels = []
        logging.info("Data: Load train data...")

        for path in set_list:
            tmp = np.load(path)
            vals = tmp["arr_0"].copy()
            labels = tmp["arr_1"].copy()
            tmp.close()

            for i in range(len(labels)):
                train_vals.append(vals[i])

                if all_labels:
                    train_labels.append(labels[i])
                else:
                    if self.config['label_pos'] == "end":
                        # It takes the end value as label
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_arg = label_arg[-1]
                    elif self.config['label_pos'] == "middle":
                        # It takes the center value as label
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_arg = label_arg[int(label_arg.shape[0] / 2)]
                    elif self.config['label_pos'] == "mode":
                        # It takes the mode value as label
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_arg = np.bincount(label_arg, minlength=self.config['num_classes'])
                        label_arg = np.argmax(label_arg)
                    else:
                        raise RuntimeError("unkown annotype")
                    train_labels.append(label_arg)

        # Make train arrays a numpy matrix
        train_vals = np.array(train_vals)
        train_labels = np.array(train_labels)


        ##############################
        # Normalizing the data to be in range [0,1] following the paper
        for ch in range(train_vals.shape[2]):
            max_ch = np.max(train_vals[:, :, ch])
            min_ch = np.min(train_vals[:, :, ch])
            median_old_range = (max_ch + min_ch) / 2
            train_vals[:, :, ch] = (train_vals[:, :, ch] - median_old_range) / (max_ch - min_ch)  # + 0.5

        # calculate number of labels
        labels = set([])
        labels = labels.union(set(train_labels.flatten()))

        # Remove NULL class label -> should be ignored
        labels = sorted(labels)
        if labels[0] == 0:
            labels = labels[1:]

        #
        # Create a class dictionary and save it
        # It is a mapping from the original labels
        # to the new labels, due that the all the
        # labels dont exist in the warehouses
        #
        #
        class_dict = {}
        for i, label in enumerate(labels):
            class_dict[label] = i

        self.class_dict = class_dict

        logging.info("Data: class_dict {}".format(class_dict))
        logging.info("Data: Augmentation of the data...")

        # Print some statistics count before augmentation
        for l_i in labels:
            n_of_x_label = train_labels == l_i
            logging.info('{} samples for label {} before augmentation'.format(np.sum(n_of_x_label), l_i))

        if train_or_test == False:
            #
            # Create batches of train indices
            # Augment more samples for rare classes
            #
            NUM_SAMPLES = 100000
            if train_labels.shape[0] < NUM_SAMPLES:
                # First balance classes a bit nicer
                batch_train_idx, train_vals, train_labels = ActivityAugmentation.augment_by_ratio(train_vals,
                                                                                                  train_labels, labels,
                                                                                                  min_sample_ratio=0.2)
            else:
                batch_train_idx, train_vals, train_labels = ActivityAugmentation.augment_by_number(train_vals,
                                                                                                   train_labels, labels,
                                                                                                   number_target_samples=1)

            logging.info("Data: Augmentation of the data done")

            # Print some statistics count
            for l_i in labels:
                logging.info('{} samples for label {}'.format(batch_train_idx[l_i].shape[0], class_dict[l_i]))

        logging.info("Data: Creating final matrices with new labels and no Null label...")

        counter = 0
        train_vals_fl = []
        train_labels_fl = []
        for idx in range(train_labels.shape[0]):
            if train_or_test == False:
                if counter >= NUM_SAMPLES:
                    break
            item = np.copy(train_vals[idx])
            label = train_labels[idx]

            if label == 0:
                continue
            train_vals_fl.append(item)
            train_labels_fl.append(int(class_dict[label]))

            counter += 1

        train_vals_fl = np.array(train_vals_fl)
        train_labels_fl = np.array(train_labels_fl)
        del train_vals
        del train_labels

        #logging.info("Data: Randomizing the data...")

        #train_vals_fl, train_labels_fl = self.random_data(train_vals_fl, train_labels_fl)

        #logging.info("Data: Done creating final matrices with new labels and no Null label...")

        #train_v_b = np.array(self.prepare_data(np.array(train_vals_fl), batch_size=batch_size))
        #train_l_b = np.array(self.prepare_data(np.array(train_labels_fl), batch_size=batch_size))

        return train_vals_fl, train_labels_fl

    def prepare_data(self, data, batch_size=1):

        logging.info("Prepare: Preparing data with batch size {}".format(batch_size))
        data_batches = []
        batches = np.arange(0, data.shape[0], batch_size)

        for idx in range(batches.shape[0] - 1):
            batch = []
            for data_in_batch in data[batches[idx]: batches[idx + 1]]:
                channel = []
                channel.append(data_in_batch)
                batch.append(channel)
            data_batches.append(batch)

        return data_batches

    def random_data(self, data, label):
        if data.shape[0] != label.shape[0]:
            logging.error("Random: Data and label havent the same number of samples")
            raise RuntimeError('Random: Data and label havent the same number of samples')
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)

        data_s = data[idx]
        label_s = label[idx]
        return data_s, label_s
