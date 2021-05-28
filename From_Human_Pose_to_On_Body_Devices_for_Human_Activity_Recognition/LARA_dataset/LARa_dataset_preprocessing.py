'''
Created on May 18, 2019

@author: fernando moya rueda
@email: fernando.moya@tu-dortmund.de
'''

import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle


import matplotlib.pyplot as plt

from scipy.stats import norm, mode


#folder path of the dataset
FOLDER_PATH = "path_to_dataset_LARa_Mocap"

# Hardcoded number of sensor channels employed in the MoCap dataset
NB_SENSOR_CHANNELS = 134



NUM_CLASSES = 8
NUM_ATTRIBUTES = 19

NORM_MAX_THRESHOLDS = [392.85,    345.05,    311.295,    460.544,   465.25,    474.5,     392.85,
                       345.05,    311.295,   574.258,   575.08,    589.5,     395.81,    503.798,
                       405.9174,  322.9,     331.81,    338.4,     551.829,   598.326,   490.63,
                       667.5,     673.4,     768.6,     560.07,    324.22,    379.405,   193.69,
                       203.65,    159.297,   474.144,   402.57,    466.863,   828.46,    908.81,
                       99.14,    482.53,    381.34,    386.894,   478.4503,  471.1,     506.8,
                       420.04,    331.56,    406.694,   504.6,     567.22,    269.432,   474.144,
                       402.57,    466.863,   796.426,   863.86,    254.2,     588.38,    464.34,
                       684.77,    804.3,     816.4,     997.4,     588.38,    464.34,    684.77,
                       889.5,     910.6,    1079.7,     392.0247,  448.56,    673.49,    322.9,
                       331.81,    338.4,     528.83,    475.37,    473.09,    679.69,    735.2,
                       767.5,     377.568,   357.569,   350.501,   198.86,    197.66,    114.931,
                       527.08,    412.28,    638.503,   691.08,    666.66,    300.48,    532.11,
                       426.02,    423.84,    467.55,    497.1,     511.9,     424.76,    348.38,
                       396.192,   543.694,   525.3,     440.25,    527.08,    412.28,    638.503,
                       729.995,   612.41,    300.33,    535.94,    516.121,   625.628,   836.13,
                       920.7,     996.8,     535.94,    516.121,   625.628,   916.15,   1009.5,
                       1095.6,    443.305,   301.328,   272.984,   138.75,    151.84,    111.35]

NORM_MIN_THRESHOLDS = [-382.62, -363.81, -315.691, -472.2, -471.4, -152.398,
                       -382.62, -363.81, -315.691, -586.3, -581.46, -213.082,
                       -400.4931, -468.4, -409.871, -336.8, -336.2, -104.739,
                       -404.083, -506.99, -490.27, -643.29, -709.84, -519.774,
                       -463.02, -315.637, -405.5037, -200.59, -196.846, -203.74,
                       -377.15, -423.992, -337.331, -817.74, -739.91, -1089.284,
                       -310.29, -424.74, -383.529, -465.34, -481.5, -218.357,
                       -442.215, -348.157, -295.41, -541.82, -494.74, -644.24,
                       -377.15, -423.992, -337.331, -766.42, -619.98, -1181.528,
                       -521.9, -581.145, -550.187, -860.24, -882.35, -645.613,
                       -521.9, -581.145, -550.187, -936.12, -982.14, -719.986,
                       -606.395, -471.892, -484.5629, -336.8, -336.2, -104.739,
                       -406.6129, -502.94, -481.81, -669.58, -703.12, -508.703,
                       -490.22, -322.88, -322.929, -203.25, -203.721, -201.102,
                       -420.154, -466.13, -450.62, -779.69, -824.456, -1081.284,
                       -341.5005, -396.88, -450.036, -486.2, -486.1, -222.305,
                       -444.08, -353.589, -380.33, -516.3, -503.152, -640.27,
                       -420.154, -466.13, -450.62, -774.03, -798.599, -1178.882,
                       -417.297, -495.1, -565.544, -906.02, -901.77, -731.921,
                       -417.297, -495.1, -565.544, -990.83, -991.36, -803.9,
                       -351.1281, -290.558, -269.311, -159.9403, -153.482, -162.718]

headers = ["sample", "label", "head_RX", "head_RY", "head_RZ", "head_TX", "head_TY", "head_TZ", "head_end_RX",
           "head_end_RY", "head_end_RZ", "head_end_TX", "head_end_TY", "head_end_TZ", "L_collar_RX", "L_collar_RY",
           "L_collar_RZ", "L_collar_TX", "L_collar_TY", "L_collar_TZ", "L_elbow_RX", "L_elbow_RY", "L_elbow_RZ",
           "L_elbow_TX", "L_elbow_TY", "L_elbow_TZ", "L_femur_RX", "L_femur_RY", "L_femur_RZ", "L_femur_TX",
           "L_femur_TY", "L_femur_TZ", "L_foot_RX", "L_foot_RY", "L_foot_RZ", "L_foot_TX", "L_foot_TY", "L_foot_TZ",
           "L_humerus_RX", "L_humerus_RY", "L_humerus_RZ", "L_humerus_TX", "L_humerus_TY", "L_humerus_TZ", "L_tibia_RX",
           "L_tibia_RY", "L_tibia_RZ", "L_tibia_TX", "L_tibia_TY", "L_tibia_TZ", "L_toe_RX", "L_toe_RY", "L_toe_RZ",
           "L_toe_TX", "L_toe_TY", "L_toe_TZ", "L_wrist_RX", "L_wrist_RY", "L_wrist_RZ", "L_wrist_TX", "L_wrist_TY",
           "L_wrist_TZ", "L_wrist_end_RX", "L_wrist_end_RY", "L_wrist_end_RZ", "L_wrist_end_TX", "L_wrist_end_TY",
           "L_wrist_end_TZ", "R_collar_RX", "R_collar_RY", "R_collar_RZ", "R_collar_TX", "R_collar_TY", "R_collar_TZ",
           "R_elbow_RX", "R_elbow_RY", "R_elbow_RZ", "R_elbow_TX", "R_elbow_TY", "R_elbow_TZ", "R_femur_RX",
           "R_femur_RY", "R_femur_RZ", "R_femur_TX", "R_femur_TY", "R_femur_TZ", "R_foot_RX", "R_foot_RY", "R_foot_RZ",
           "R_foot_TX", "R_foot_TY", "R_foot_TZ", "R_humerus_RX", "R_humerus_RY", "R_humerus_RZ", "R_humerus_TX",
           "R_humerus_TY", "R_humerus_TZ", "R_tibia_RX", "R_tibia_RY", "R_tibia_RZ", "R_tibia_TX", "R_tibia_TY",
           "R_tibia_TZ", "R_toe_RX", "R_toe_RY", "R_toe_RZ", "R_toe_TX", "R_toe_TY", "R_toe_TZ", "R_wrist_RX",
           "R_wrist_RY", "R_wrist_RZ", "R_wrist_TX", "R_wrist_TY", "R_wrist_TZ", "R_wrist_end_RX", "R_wrist_end_RY",
           "R_wrist_end_RZ", "R_wrist_end_TX", "R_wrist_end_TY", "R_wrist_end_TZ", "root_RX", "root_RY", "root_RZ",
           "root_TX", "root_TY", "root_TZ"]

annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
             "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
             "S13": "A08", "S14": "A06", "S15": "A05", "S16": "A05"}

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

#scenario = ['S01']
persons = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
           "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
repetition = ["N01", "N02"]
annotator_S01 = ["A17", "A12"]

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}


def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #included-excluded
    features_delete = np.arange(68, 74)
    
    return np.delete(data, features_delete, 1)

def divide_x_y(data):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]
    

    return data_t, data_x, data_y

def normalize(data):
    """
    Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    try:
        max_list, min_list = np.array(NORM_MAX_THRESHOLDS), np.array(NORM_MIN_THRESHOLDS)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i]-min_list[i])/diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
    except:
        raise("Error in normalization")
        
    return data


def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    @return data_x: Sequence train inputs [Batch,1, C, T]
    @return data_y_labels: Activity classes [B, 1]
    @return data_y_all: Activity classes for samples [Batch,1,T]
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:
            # Label according to mode
            try:
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                    labels = np.zeros((20)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:, 1:], axis=0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)



def compute_max_min(ids):
    '''
    Compute the max and min values for normalizing the data.
    
    
    print max and min.
    These values will be computed only once and the max min values
    will be place as constants
    
    @param ids: ids for train
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
    
    max_values_total = np.zeros((132))
    min_values_total = np.ones((132)) * 1000000
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'S03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"

                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("Files loaded")

                        data_t, data_x, data_y = divide_x_y(data)
                        del data_t
                        del data_y

                        max_values = np.max(data_x, axis = 0)
                        min_values = np.min(data_x, axis = 0)

                        max_values_total = np.max((max_values, max_values_total), axis = 0)
                        min_values_total = np.min((min_values, min_values_total), axis = 0)

                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_norm))
    
    print("Max values \n{}".format(max_values_total))
    print("Min values \n{}".format(min_values_total))
    
    return




def compute_min_num_samples(ids, boolean_classes=True, attr=0):
    '''
    Compute the minimum duration of a sequences with the same classes or attribute
    
    This value will help selecting the best sliding window size
    
    @param ids: ids for train
    @param boolean_classes: selecting between classes or attributes
    @param attr: ids for attribute
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    if boolean_classes:
        NUM_CLASSES = 8
    else:
        NUM_CLASSES = 2

    #min_durations = np.ones((NUM_CLASSES)) * 10000000
    min_durations = np.empty((0,NUM_CLASSES))
    hist_classes_all = np.zeros((NUM_CLASSES))
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file,N)

                    try:
                        data = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        labels = data[:,attr]
                        print("Files loaded")

                        min_duration = np.zeros((1,NUM_CLASSES))
                        for c in range(NUM_CLASSES):

                            #indexes per class
                            idxs = np.where(labels == c)[0]
                            counter = 0
                            min_counter = np.Inf
                            #counting if continuity in labels
                            for idx in range(idxs.shape[0] - 1):
                                if idxs[idx + 1] - idxs[idx] == 1:
                                    counter += 1
                                else:
                                    if counter < min_counter:
                                        min_counter = counter
                                        counter = 0
                            if counter < min_counter:
                                min_counter = counter
                                counter = 0
                            min_duration[0,c] = min_counter

                            print("class  {} counter size {}".format(c, min_counter))

                        min_durations = np.append(min_durations, min_duration, axis = 0)
                        #Statistics

                        hist_classes = np.bincount(labels.astype(int), minlength = NUM_CLASSES)
                        hist_classes_all += hist_classes

                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_label))
    
    min_durations[min_durations == 0] = np.Inf
    print("Minimal duration per class \n{}".format(min_durations))
    
    print("Number of samples per class {}".format(hist_classes_all))
    print("Number of samples per class {}".format(hist_classes_all / np.float(np.sum(hist_classes_all)) * 100))
    
    return np.min(min_durations, axis = 0)



def compute_statistics_samples(ids, boolean_classes=True, attr=0):
    '''
    Compute some statistics of the duration of the sequences data:

    print:
    Max and Min durations per class or attr
    Mean and Std durations per class or attr
    Lower whiskers durations per class or attr
    1st quartile of durations per class or attr
    Histogram of proportion per class or attr
    
    @param ids: ids for train
    @param boolean_classes: selecting between classes or attributes
    @param attr: ids for attribute
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_list_class = {}

    if boolean_classes:
        NUM_CLASSES = 8
    else:
        NUM_CLASSES = 2
    
    for cl in range(NUM_CLASSES):
        counter_list_class[cl] = []
    
    hist_classes_all = np.zeros((NUM_CLASSES))
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        data = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        labels = data[:,attr]
                        print("Files loaded")

                        for c in range(NUM_CLASSES):

                            #indexes per class
                            idxs = np.where(labels == c)[0]
                            counter = 0

                            #counting if continuity in labels
                            for idx in range(idxs.shape[0] - 1):
                                if idxs[idx + 1] - idxs[idx] == 1:
                                    counter += 1
                                else:
                                    counter_list_class[c].append(counter)
                                    counter = 0

                                if (idx+1) == (idxs.shape[0] - 1):
                                    counter_list_class[c].append(counter)
                                    counter = 0
                        #Statistics

                        hist_classes = np.bincount(labels.astype(int), minlength = NUM_CLASSES)
                        hist_classes_all += hist_classes
                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_label))

    fig = plt.figure()
    axis_list = []
    axis_list.append(fig.add_subplot(421))
    axis_list.append(fig.add_subplot(422))
    axis_list.append(fig.add_subplot(423))
    axis_list.append(fig.add_subplot(424))
    axis_list.append(fig.add_subplot(425))
    axis_list.append(fig.add_subplot(426))
    axis_list.append(fig.add_subplot(427))
    axis_list.append(fig.add_subplot(428))
    
    fig2 = plt.figure()
    axis_list_2 = []
    axis_list_2.append(fig2.add_subplot(111))

    fig3 = plt.figure()
    axis_list_3 = []
    axis_list_3.append(fig3.add_subplot(421))
    axis_list_3.append(fig3.add_subplot(422))
    axis_list_3.append(fig3.add_subplot(423))
    axis_list_3.append(fig3.add_subplot(424))
    axis_list_3.append(fig3.add_subplot(425))
    axis_list_3.append(fig3.add_subplot(426))
    axis_list_3.append(fig3.add_subplot(427))
    axis_list_3.append(fig3.add_subplot(428))  

    colours = {0 : 'b', 1 : 'g', 2 : 'r', 3 : 'c', 4 : 'm', 5 : 'y', 6 : 'k', 7 : 'greenyellow'}
    
    mins = []
    mus = []
    sigmas = []
    min_1_data = []
    min_2_data = []
    min_3_data = []
    medians = []
    lower_whiskers = []
    Q1s = []
    for cl in range(NUM_CLASSES):
        mu = np.mean(np.array(counter_list_class[cl]))
        sigma = np.std(np.array(counter_list_class[cl]))
        
        mus.append(mu)
        sigmas.append(sigma)
        min_1_data.append(- 1 * sigma + mu)
        min_2_data.append(- 2 * sigma + mu)
        min_3_data.append(- 3 * sigma + mu)
        mins.append(np.min(np.array(counter_list_class[cl])))
        medians.append(np.median(np.array(counter_list_class[cl])))
        
        x = np.linspace(-3 * sigma + mu, 3 * sigma + mu, 100)
        
        axis_list[cl].plot(x, norm.pdf(x,mu,sigma) / np.float(np.max(norm.pdf(x,mu,sigma))),
                           '-b', label='mean:{}_std:{}'.format(mu, sigma))
        axis_list[cl].plot(counter_list_class[cl], np.ones(len(counter_list_class[cl])) , 'ro')
        result_box = axis_list[cl].boxplot(counter_list_class[cl], vert=False)
        lower_whiskers.append(result_box['whiskers'][0].get_data()[0][0])
        Q1s.append(result_box['whiskers'][0].get_data()[0][1])
        
        axis_list_2[0].plot(x, norm.pdf(x,mu,sigma) /  np.float(np.max(norm.pdf(x,mu,sigma))),
                            '-b', label='mean:{}_std:{}'.format(mu, sigma), color = colours[cl])
        axis_list_2[0].plot(counter_list_class[cl], np.ones(len(counter_list_class[cl])) , 'ro')
                            #color = colours[cl], marker='o')
                            
                            
        axis_list_3[cl].boxplot(counter_list_class[cl])

        axis_list_2[0].relim()
        axis_list_2[0].autoscale_view()
        axis_list_2[0].legend(loc='best')

        fig.canvas.draw()
        fig2.canvas.draw()
        plt.pause(2.0)
    
    print("Mins {} Min {} Argmin {}".format(mins, np.min(mins), np.argmin(mins)))
    print("Means {} Min {} Argmin {}".format(mus, np.min(mus), np.argmin(mus)))
    print("Stds {} Min {}".format(sigmas, sigmas[np.argmin(mus)]))
    print("Medians {} Min {} Argmin {}".format(medians, np.min(medians), np.argmin(medians)))
    print("Lower Whiskers {} Min {} Argmin {}".format(lower_whiskers, np.min(lower_whiskers), np.argmin(lower_whiskers)))
    print("Q1s {} Min {} Argmin {}".format(Q1s, np.min(Q1s), np.argmin(Q1s)))
    
    
    print("1sigma from mu {}".format(min_1_data))
    print("2sigma from mu {}".format(min_2_data))
    print("3sigma from mu {}".format(min_3_data))
    
    print("Min 1sigma from mu {}".format(np.min(min_1_data)))
    print("Min 2sigma from mu {}".format(np.min(min_2_data)))
    print("Min 3sigma from mu {}".format(np.min(min_3_data)))
    
    print("Number of samples per class {}".format(hist_classes_all))
    print("Number of samples per class {}".format(hist_classes_all / np.float(np.sum(hist_classes_all)) * 100))
    
    return


################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None, half=False,
                  identity_bool=False, usage_modus='train'):
    '''
    creates files for each of the sequences, which are extracted from a file
    following a sliding window approach
    
    returns
    Sequences are stored in given path
    
    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    @param half: using the half of the recording frequency
    @param identity_bool: selecting for identity experiment
    @param usage_modus: selecting Train, Val or testing
    '''

    if identity_bool:
        if usage_modus == 'train':
            recordings = ['R{:02d}'.format(r) for r in range(1, 21)]
        elif usage_modus == 'val':
            recordings = ['R{:02d}'.format(r) for r in range(21, 26)]
        elif usage_modus == 'test':
            recordings = ['R{:02d}'.format(r) for r in range(26, 31)]
    else:
        recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_seq = 0
    hist_classes_all = np.zeros(NUM_CLASSES)

    for P in persons:
        if P not in ids:
            print("\nNo Person in expected IDS {}".format(P))
        else:
            if P == 'S11':
                # Selecting the proportions of the train, val or testing according to the quentity of
                # recordings per subject, as there are not equal number of recordings per subject
                # see dataset for checking the recording files per subject
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 10)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(10, 12)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(12, 15)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            elif P == 'S12':
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 25)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(25, 28)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(28, 31)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            else:
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 21)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(21, 26)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(26, 31)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            for R in recordings:
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[R]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[R] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"

                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        #getting data
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_norm))
                        data = select_columns_opp(data)
                        print("Columns selected")
                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        #Getting labels and attributes
                        labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data = np.delete(data, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)

                        # halving the frequency
                        if half:
                            downsampling = range(0, data.shape[0], 2)
                            data = data[downsampling]
                            labels = labels[downsampling]
                            data_t, data_x, data_y = divide_x_y(data)
                            del data_t
                        else:
                            data_t, data_x, data_y = divide_x_y(data)
                            del data_t

                    except:
                        print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        # checking if annotations are consistent
                        data_x = normalize(data_x)
                        if np.sum(data_y == labels[:, 0]) == data_y.shape[0]:

                            # Sliding window approach
                            print("Starting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, labels.astype(int),
                                                             sliding_window_length,
                                                             sliding_window_step, label_pos_end = False)
                            print("Windows are extracted")

                            # Statistics
                            hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                            hist_classes_all += hist_classes
                            print("Number of seq per class {}".format(hist_classes_all))

                            for f in range(X.shape[0]):
                                try:

                                    sys.stdout.write('\r' + 'Creating sequence file '
                                                            'number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)

                                    # Storing the sequences
                                    obj = {"data": seq, "label": y[f], "labels": y_all[f],
                                           "identity": labels_persons[P]}
                                    f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                                    f.close()

                                    counter_seq += 1
                                except:
                                    raise('\nError adding the seq')

                            print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_norm))

                            del data
                            del data_x
                            del data_y
                            del X
                            del labels
                            del class_labels

                        else:
                            print("\nNot consisting annotation in  {}".format(file_name_norm))
                            continue

                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
            
    return



def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return

def general_statistics(ids):
    '''
    Computing min duration of activity classes

    @param ids: IDS for subjects in the dataset.
    '''
    #compute_max_min(ids)
    attr_check = 19
    min_durations = compute_min_num_samples(ids, boolean_classes=False, attr=attr_check)

    compute_statistics_samples(ids, boolean_classes=False, attr=attr_check)

    print("Minimum per class {}".format(min_durations))
    print("Minimum ordered {}".format(np.sort(min_durations)))
    return


def create_dataset(half=False):
    '''
    create dataset
    - Segmentation
    - Storing sequences

    @param half: set for creating dataset with half the frequence.
    '''

    train_ids = ["S01", "S02", "S03", "S04", "S07", "S08", "S09", "S10", "S15", "S16"]
    train_final_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10" "S11", "S12", "S15", "S16"]
    val_ids = ["S05", "S11", "S12"]
    test_ids = ["S06", "S13", "S14"]

    all_data = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    #general_statistics(train_ids)

    if half:
        "Path to the segmented sequences"
        base_directory = '/path_where_sequences_will_ve_stored/MoCap_dataset_half_freq/'
        sliding_window_length = 100
        sliding_window_step = 12
    else:
        "Path to the segmented sequences"
        base_directory = '/path_where_sequences_will_ve_stored/MoCap_dataset/'
        sliding_window_length = 200
        sliding_window_step = 25

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_train, half=half, usage_modus='train')
    generate_data(val_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_val, half=half)
    generate_data(test_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_test, half=half)

    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return



if __name__ == '__main__':
    # Creating dataset for LARA Mocap 200Hz or LARA Mocap 100Hz
    # Set the path to where the segmented windows will be located
    # This path will be needed for the main.py

    # Dataset (extracted segmented windows) will be stored in a given folder by the user,
    # However, inside the folder, there shall be the subfolders (sequences_train, sequences_val, sequences_test)
    # These folders and subfolfders gotta be created manually by the user
    # This as a sort of organisation for the dataset
    # MoCap_dataset/sequences_train
    # MoCap_dataset/sequences_val
    # MoCap_dataset/sequences_test

    create_dataset(half=False)

    print("Done")
