'''
Created on Dec 02, 2019

@author: fmoya
'''

import os
import numpy as np
from pandas import Series
import torch.utils.data as data
import logging

from sliding_window import sliding_window

# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
OPPORTUNITY_DATA_FILES = ['dataset/S1-Drill.dat', #0
                          'dataset/S1-ADL1.dat',  #1
                          'dataset/S1-ADL2.dat',  #2
                          'dataset/S1-ADL3.dat',  #3
                          'dataset/S1-ADL4.dat',  #4
                          'dataset/S1-ADL5.dat',  #5
                          'dataset/S2-Drill.dat', #6
                          'dataset/S2-ADL1.dat',  #7
                          'dataset/S2-ADL2.dat',  #8
                          'dataset/S3-Drill.dat', #9
                          'dataset/S3-ADL1.dat',  #10
                          'dataset/S3-ADL2.dat',  #11
                          'dataset/S2-ADL3.dat',  #12
                          'dataset/S3-ADL3.dat',  #13
                          'dataset/S2-ADL4.dat',  #14
                          'dataset/S2-ADL5.dat',  #15
                          'dataset/S3-ADL4.dat',  #16
                          'dataset/S3-ADL5.dat'   #17
                          ]



# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]



class Opportunity(data.Dataset):


    def __init__(self, config, partition_modus = 'train'):

        self.config = config
        self.partition_modus = partition_modus

        self.X, self.Y = self.load_data()
        self.X, self.y, self.Y = self.opp_sliding_window(self.X, self.Y)

        self.X = np.reshape(self.X, [self.X.shape[0], 1, self.X.shape[1], self.X.shape[2]])

        return

    def __getitem__(self, idx):
        window_data = {"data": self.X[idx], "label": self.y[idx], "labels": self.Y[idx]}
        return window_data

    def __len__(self):
        return self.X.shape[0]

    def load_data(self):
        """Function to load the OPPORTUNITY challenge raw data and process all sensor channels

        :param dataset: string
            Path with original OPPORTUNITY zip file
        :param target_filename: string
            Processed file
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
            recognition modes of locomotion/postures and recognition of sporadic gestures.
        :return _train, y_train, X_val, y_val, X_test, y_test:

        """

        X = np.empty((0, self.config['NB_sensor_channels']))
        Y = np.empty((0))

        #zf = zipfile.ZipFile(self.config['dataset_directory'])

        if self.partition_modus == 'train':
            #idx_files = [ids for ids in range(0,12)]
            if self.config["proportions"] == 0.2:
                idx_files = [0, 4, 8, 10]
            elif self.config["proportions"] == 0.5:
                idx_files = [0, 2, 4, 6, 8, 10, 11]
            elif self.config["proportions"] == 1.0:
                idx_files = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif self.partition_modus == 'val':
            idx_files = [ids for ids in range(12,14)] #12,14
        elif self.partition_modus == 'test':
            idx_files = [ids for ids in range(14,18)] #14,18
        else:
            raise("Wrong Dataset partition settup")
        logging.info('        Dataloader: Processing dataset files ...')


        for idx_f in idx_files:
            try:
                logging.info('        Dataloader: Loading file...{0}'.format(OPPORTUNITY_DATA_FILES[idx_f]))
                raw_data = np.loadtxt(self.config['dataset_root'] + OPPORTUNITY_DATA_FILES[idx_f])
                x, y = self.process_dataset_file(raw_data)
                logging.info(x.shape)
                logging.info(y.shape)

                X = np.vstack((X, x))
                Y = np.concatenate([Y, y])
            except KeyError:
                logging.error('        Dataloader: ERROR: Did not find {0} in zip file'.format(OPPORTUNITY_DATA_FILES[idx_f]))

        logging.info("        Dataloader: Final dataset with size: | train {0}".format(X.shape))
        return X, Y


    def check_data(self, data_set):
        """Try to access to the file and checks if dataset is in the data directory
           In case the file is not found try to download it from original location

        :param data_set: String
                Path with original OPPORTUNITY zip file
        :return:
        """
        logging.info('        Dataloader: Checking dataset {0}'.format(data_set))
        data_dir, data_file = os.path.split(data_set)
        # When a directory is not provided, check if dataset is in the data directory
        if data_dir == "" and not os.path.isfile(data_set):
            new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
            if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
                data_set = new_path

        # When dataset not found, try to download it from UCI repository
        if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
            logging.info('        Dataloader: ... dataset path {0} not found'.format(data_set))
            import urllib
            origin = (
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
            )
            if not os.path.exists(data_dir):
                logging.info('        Dataloader: ... creating directory {0}'.format(data_dir))
                os.makedirs(data_dir)
            logging.info('        Dataloader: ... downloading data from {0}'.format(origin))
            urllib.urlretrieve(origin, data_set)

        return data_dir

    def process_dataset_file(self, raw_data):
        """Function defined as a pipeline to process individual OPPORTUNITY files

        :param raw_data: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :return: numpy integer matrix, numy integer array
            Processed sensor data, segmented into features (x) and labels (y)
        """
        #resamp = Resampling()
        # Select correct columns
        raw_data = self.select_columns_opp(raw_data)

        # Columns are segmented into features and labels
        data_t, data_x, data_y = self.divide_x_y(raw_data)
        #_, data_x, data_y = resamp.interpolate(data_t, data_x, data_y, 50)

        data_y = self.adjust_idx_labels(data_y)
        data_y = data_y.astype(int)

        # Perform linear interpolation
        data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

        # Remaining missing data are converted to zero
        data_x[np.isnan(data_x)] = 0

        # All sensor channels are normalized
        data_x = self.normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

        return data_x, data_y

    def select_columns_opp(self, raw_data):
        """Selection of the 113 columns employed in the OPPORTUNITY challenge

        :param raw_data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """

        #                     included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(raw_data, features_delete, 1)

    def divide_x_y(self, raw_data):
        """Segments each sample into features and label

        :param raw_data: numpy integer matrix
            Sensor data
        :param task: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numpy integer array
            Recording time, Features encapsulated into a matrix and labels as an array
        """

        try:
            data_t = raw_data[:, 0]
            data_x = raw_data[:, 1:114]
            if self.config['dataset'] not in ['locomotion', 'gesture']:
                raise RuntimeError("Invalid label: '%s'" % self.config['dataset'])
            if self.config['dataset'] == 'locomotion':
                logging.info("        Dataloader: Locomotion")
                data_y = raw_data[:, 114]  # Locomotion label
            elif self.config['dataset'] == 'gesture':
                logging.info("        Dataloader: Gestures")
                data_y = raw_data[:, 115]  # Gestures label
        except KeyError:
            logging.error(KeyError)

        return data_t, data_x, data_y

    def adjust_idx_labels(self, data_y):
        """Transforms original labels into the range [0, nb_labels-1]

        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """

        try:
            if self.config['dataset'] == 'locomotion':  # Labels for locomotion are adjusted
                data_y[data_y == 4] = 3
                data_y[data_y == 5] = 4
            elif self.config['dataset'] == 'gesture':  # Labels for gestures are adjusted
                data_y[data_y == 406516] = 1
                data_y[data_y == 406517] = 2
                data_y[data_y == 404516] = 3
                data_y[data_y == 404517] = 4
                data_y[data_y == 406520] = 5
                data_y[data_y == 404520] = 6
                data_y[data_y == 406505] = 7
                data_y[data_y == 404505] = 8
                data_y[data_y == 406519] = 9
                data_y[data_y == 404519] = 10
                data_y[data_y == 406511] = 11
                data_y[data_y == 404511] = 12
                data_y[data_y == 406508] = 13
                data_y[data_y == 404508] = 14
                data_y[data_y == 408512] = 15
                data_y[data_y == 407521] = 16
                data_y[data_y == 405506] = 17
        except KeyError:
            logging.error(KeyError)
        return data_y

    def normalize(self, raw_data, max_list, min_list):
        """Normalizes all sensor channels

        :param data: numpy integer matrix
            Sensor data
        :param max_list: numpy integer array
            Array containing maximums values for every one of the 113 sensor channels
        :param min_list: numpy integer array
            Array containing minimum values for every one of the 113 sensor channels
        :return:
            Normalized sensor data
        """
        max_list, min_list = np.array(max_list), np.array(min_list)
        diffs = max_list - min_list
        for i in np.arange(raw_data.shape[1]):
            raw_data[:, i] = (raw_data[:, i] - min_list[i]) / diffs[i]
        #     Checking the boundaries
        raw_data[raw_data > 1] = 0.99
        raw_data[raw_data < 0] = 0.00
        return raw_data


    ##################################################
    #############  opp_sliding_window  ###############
    ##################################################

    def opp_sliding_window(self, data_x, data_y):
        '''
        Performs the sliding window approach on the data and the labels

        return three arrays.
        - data, an array where first dim is the windows
        - labels per window according to end, middle or mode
        - all labels per window

        @param data_x: ids for train
        @param data_y: ids for train
        @return data_x: Sequence train inputs [windows, C, T]
        @return data_y_labels: Activity classes [windows, 1]
        @return data_y_all: Activity classes for samples [windows, 1, T]
        '''

        ws = self.config['sliding_window_length']
        ss = self.config['sliding_window_step']

        logging.info('        Dataloader: Sliding window with ws {} and ss {}'.format(ws, ss))

        # Segmenting the data with labels taken from the end of the window
        data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
        if self.config['label_pos'] == 'end':
            data_y_labels = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
        elif self.config['label_pos'] == 'middle':
            # Segmenting the data with labels from the middle of the window
            data_y_labels = np.asarray([[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
        elif self.config['label_pos'] == 'mode':
            data_y_labels = []
            for sw in sliding_window(data_y, ws, ss):
                count_l = np.bincount(sw.astype(int), minlength=self.config['num_classes'])
                idy = np.argmax(count_l)
                data_y_labels.append(idy)
            data_y_labels = np.asarray(data_y_labels)

        # Labels of each sample per window
        data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])

        logging.info('        Dataloader: Sequences are segmented')

        return data_x.astype(np.float32), \
               data_y_labels.reshape(len(data_y_labels)).astype(np.uint8), \
               data_y_all.astype(np.uint8)

