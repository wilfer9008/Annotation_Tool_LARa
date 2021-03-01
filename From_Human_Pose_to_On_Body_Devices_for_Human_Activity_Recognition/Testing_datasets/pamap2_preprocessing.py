'''
Created on Dec 03, 2019

@author: fmoya
'''

import numpy as np
import torch.utils.data as data
import logging

from sliding_window import sliding_window



# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat', #0
                     'PAMAP2_Dataset/Optional/subject101.dat', #1
                     'PAMAP2_Dataset/Protocol/subject102.dat', #2
                     'PAMAP2_Dataset/Protocol/subject103.dat', #3
                     'PAMAP2_Dataset/Protocol/subject104.dat', #4
                     'PAMAP2_Dataset/Protocol/subject107.dat', #5
                     'PAMAP2_Dataset/Protocol/subject108.dat', #6
                     'PAMAP2_Dataset/Optional/subject108.dat', #7
                     'PAMAP2_Dataset/Protocol/subject109.dat', #8
                     'PAMAP2_Dataset/Optional/subject109.dat', #9
                     'PAMAP2_Dataset/Protocol/subject105.dat', #10
                     'PAMAP2_Dataset/Optional/subject105.dat', #11
                     'PAMAP2_Dataset/Protocol/subject106.dat', #12
                     'PAMAP2_Dataset/Optional/subject106.dat', #13
                      ]


NORM_MAX_THRESHOLDS = [202.0, 35.5, 47.6314, 155.532, 157.76, 45.5484, 62.2598, 61.728, 21.8452,
                       13.1222, 14.2184, 137.544, 109.181, 100.543, 38.5625, 26.386, 153.582,
                       37.2936, 23.9101, 61.9328, 36.9676, 15.5171, 5.97964, 2.94183, 80.4739,
                       39.7391, 95.8415, 35.4375, 157.232, 157.293, 150.99, 61.9509, 62.0461,
                       60.9357, 17.4204, 13.5882, 13.9617, 91.4247, 92.867, 146.651]

NORM_MIN_THRESHOLDS = [0., 0., -114.755, -104.301, -73.0384, -61.1938, -61.8086, -61.4193, -27.8044,
                       -17.8495, -14.2647, -103.941, -200.043, -163.608, 0., -29.0888, -38.1657, -57.2366,
                       -32.9627, -39.7561, -56.0108, -10.1563, -5.06858, -3.99487, -70.0627, -122.48,
                       -66.6847, 0., -155.068, -155.617, -156.179, -60.3067, -61.9064, -62.2629, -14.162,
                       -13.0401, -14.0196, -172.865, -137.908, -102.232]





class Pamap2(data.Dataset):


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
        """Function to load the Pamap2 challenge raw data and process all sensor channels

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

        if self.partition_modus == 'train':
            # dx_files = [ids for ids in range(0,10)]
            if self.config["proportions"] == 0.2:
                idx_files = [0, 9]
            elif self.config["proportions"] == 0.5:
                idx_files = [0, 2, 4, 6, 8]
            elif self.config["proportions"] == 1.0:
                idx_files = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif self.partition_modus == 'val':
            # idx_files = [ids for ids in range(10,12)]
            idx_files = [10, 11]
        elif self.partition_modus == 'test':
            # idx_files = [ids for ids in range(12,14)]
            idx_files = [12, 13]
        else:
            raise("Wrong Dataset partition settup")
        logging.info('Processing dataset files ...')
        for idx_f in idx_files:
            try:
                logging.info('Loading file...{0}'.format(PAMAP2_DATA_FILES[idx_f]))
                raw_data = np.loadtxt(self.config['dataset_root'] + PAMAP2_DATA_FILES[idx_f])
                x, y = self.process_dataset_file(raw_data)
                logging.info(x.shape)
                logging.info(y.shape)

                X = np.vstack((X, x))
                Y = np.concatenate([Y, y])
            except KeyError:
                logging.error('ERROR: Did not find {0} in zip file'.format(PAMAP2_DATA_FILES[idx_f]))

        logging.info("Final dataset with size: | train {0}".format(X.shape))
        return X, Y

    def process_dataset_file(self, raw_data):
        """Function defined as a pipeline to process individual OPPORTUNITY files

        :param data: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numy integer array
            Processed sensor data, segmented into features (x) and labels (y)
        """

        # Colums are segmentd into features and labels
        data_t, data_x, data_y = self.divide_x_y(raw_data)
        data_t, data_x, data_y = self.del_labels(data_t, data_x, data_y)

        data_y = self.adjust_idx_labels(data_y)
        data_y = data_y.astype(int)

        # Select correct columns
        data_x = self.select_columns_opp(data_x)

        if data_x.shape[0] != 0:
            HR_no_NaN = self.complete_HR(data_x[:, 0])
            data_x[:, 0] = HR_no_NaN

            data_x[np.isnan(data_x)] = 0
            # All sensor channels are normalized
            data_x = self.normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

        data_t, data_x, data_y = self.downsampling(data_t, data_x, data_y)

        return data_x, data_y


    def select_columns_opp(self, raw_data):
        """Selection of the columns employed in the Pamap2 dataset

        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """

        #                     included-excluded
        features_delete = np.arange(14, 18)
        features_delete = np.concatenate([features_delete, np.arange(31, 35)])
        features_delete = np.concatenate([features_delete, np.arange(48, 52)])

        return np.delete(raw_data, features_delete, 1)


    def divide_x_y(self, raw_data):
        """Segments each sample into features and label

        :param data: numpy integer matrix
            Sensor data
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numpy integer array
            Features encapsulated into a matrix and labels as an array
        """
        data_t = raw_data[:, 0]
        data_y = raw_data[:, 1]
        data_x = raw_data[:, 2:]

        return data_t, data_x, data_y

    def del_labels(self, data_t, data_x, data_y):

        idy = np.where(data_y == 0)[0]
        labels_delete = idy

        idy = np.where(data_y == 8)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 9)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 10)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 11)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 18)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 19)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 20)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y,
                                                                                                   labels_delete, 0)

    def adjust_idx_labels(self, data_y):
        """Transforms original labels into the range [0, nb_labels-1]

        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """

        data_y[data_y == 24] = 0
        data_y[data_y == 12] = 8
        data_y[data_y == 13] = 9
        data_y[data_y == 16] = 10
        data_y[data_y == 17] = 11

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

    def complete_HR(self, raw_data):

        pos_NaN = np.isnan(raw_data)
        idx_NaN = np.where(pos_NaN == False)[0]
        data_no_NaN = raw_data * 0
        for idx in range(idx_NaN.shape[0] - 1):
            data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = raw_data[idx_NaN[idx]]

        data_no_NaN[idx_NaN[-1]:] = raw_data[idx_NaN[-1]]

        return data_no_NaN

    def downsampling(self, data_t, data_x, data_y):

        idx = np.arange(0, data_t.shape[0], 3)

        return data_t[idx], data_x[idx], data_y[idx]


    ##################################################
    #############  opp_sliding_window  ###############
    ##################################################

    def opp_sliding_window(self, data_x, data_y):
        ws = self.config['sliding_window_length']
        ss = self.config['sliding_window_step']

        logging.info('        Network_User: Sliding window with ws {} and ss {}'.format(ws, ss))

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
                count_l = np.bincount(sw, minlength=self.config['num_classes'])
                idy = np.argmax(count_l)
                data_y_labels.append(idy)
            data_y_labels = np.asarray(data_y_labels)

        # Labels of each sample per window
        data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])

        logging.info('        Network_User: Sequences are segmented')

        return data_x.astype(np.float32), \
               data_y_labels.reshape(len(data_y_labels)).astype(np.uint8), \
               data_y_all.astype(np.uint8)