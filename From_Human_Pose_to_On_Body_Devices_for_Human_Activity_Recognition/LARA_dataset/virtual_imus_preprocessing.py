'''
Created on March 08, 2020

@author: fmoya
'''


import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle
import scipy.interpolate

import csv

#from HARwindows import HARWindows

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import DataLoader

from scipy.stats import norm, mode

# Hardcoded number of sensor channels employed in the MoCap dataset
NB_SENSOR_CHANNELS = 134

NUM_CLASSES = 8
NUM_ATTRIBUTES = 19

NORM_MAX_THRESHOLDS = [3631.08295833,  4497.89608551,  3167.75032512,  7679.5730537,
                       7306.54182726,  5053.64124207,  3631.08295833,  4497.89608551,
                       3167.75032512,  7520.1195731,   5866.25362466,  4561.74563579,
                       8995.09142766, 10964.53262598,  9098.53329506,  5449.80983967,
                       5085.72851705,  3473.14411695, 21302.63337367, 18020.90101605,
                       16812.03779666, 15117.39792079, 18130.13729404, 20435.92835743,
                       11904.27678521, 10821.65838575, 12814.3751877,   6207.89358209,
                       6207.66099742,  5744.44264059, 18755.48551252, 13983.17089764,
                       15044.68926247, 16560.57636066, 16249.15177318, 50523.97223427,
                       13977.42954035, 11928.06755543, 14040.39506186, 12506.74970729,
                       11317.24558625, 18513.54602435, 17672.82964044, 12698.55673051,
                       10791.77877743, 21666.83178164, 24842.59483402, 17764.63662927,
                       18755.48551252, 13983.17089764, 15044.68926247, 31417.17595547,
                       25543.65023889, 41581.37446104, 17854.61535493, 17888.99265748,
                       18270.05893641, 21940.59554685, 28842.57380555, 26487.70921625,
                       17854.61535493, 17888.99265748, 18270.05893641, 26450.29857186,
                       29322.63199557, 33366.25283016, 14493.32082359,  7227.39740818,
                       13485.20175734,  5449.80983967,  5085.72851705,  3473.14411695,
                       18651.75669658, 23815.12584062, 21579.39921193, 32556.06051458,
                       19632.56510825, 38223.1285115,  23864.06687714,  8143.79426126,
                       10346.98996386, 10062.37072833, 14465.46088042,  6363.84691989,
                       15157.19066909, 14552.35778047, 17579.07446021, 28342.70805834,
                       30250.26765128, 38822.32871634, 11675.2974133,  21347.97047233,
                       9688.26485256, 15386.32967605,  6973.26742725,  8413.29172314,
                       17034.91758251, 13001.19959282, 11449.20721127, 28191.66779469,
                       23171.19896564, 18113.64230323, 15157.19066909, 14552.35778047,
                       17579.07446021, 33429.96810456, 24661.64883529, 42107.2934863,
                       17288.1377315,  27332.9219541,  18660.71577123, 38596.98925921,
                       34977.05772231, 60992.88899744, 17288.1377315, 27332.9219541,
                       18660.71577123, 48899.45655,    45458.05180034, 70658.90456363,
                       13302.12166712,  6774.86630402,  5359.33623984,  4589.29865259,
                       5994.63947534,  3327.83594399]

NORM_MIN_THRESHOLDS = [-2948.30406222,  -5104.92518724,  -3883.21490707,  -8443.7668244,
                       -8476.79530575,  -6094.99248245,  -2948.30406222,  -5104.92518724,
                       -3883.21490707,  -7638.48719528,  -7202.24599521,  -6357.05970106,
                       -7990.36654959, -12559.35302651, -11038.18468099,  -5803.12768035,
                       -6046.67001691,  -4188.84645697, -19537.6152259,  -19692.04664185,
                       -18548.37708291, -18465.90530892, -22119.96935856, -16818.45181433,
                       -10315.80425454, -11129.02825056, -10540.85493494,  -5590.78061688,
                       -6706.54838582,  -7255.845227,   -20614.03216916, -15124.01287142,
                       -14418.97715774, -15279.30518515, -15755.49700464, -52876.67430633,
                       -17969.48805632, -11548.41807713, -12319.34970371, -15331.29246293,
                       -13955.81324322, -15217.97507736, -17828.76429939, -12235.18670802,
                       -10508.19455787, -21039.94502811, -23382.9517919,  -16289.47810937,
                       -20614.03216916, -15124.01287142, -14418.97715774, -29307.81700117,
                       -27385.77923632, -39251.64863972, -18928.76957804, -20399.72095829,
                       -17417.32884474, -18517.86724449, -23566.88454454, -22782.77912723,
                       -18928.76957804, -20399.72095829, -17417.32884474, -23795.85006226,
                       -29914.3625062,  -27826.42086488, -11998.09109479,  -7978.46422461,
                       -12388.18397068,  -5803.12768035,  -6046.67001691,  -4188.84645697,
                       -20341.80941731, -23459.72733752, -20260.81953868, -33146.50450199,
                       -18298.11527347, -41007.64090081, -20861.12016565, -10084.98928355,
                       -12620.01970423,  -8183.86583411, -11868.40952478,  -6055.26285391,
                       -12839.53720997, -14943.34999686, -19473.17909211, -26832.66919396,
                       -28700.83598723, -31873.10404748, -14363.94407474, -19923.81826152,
                       -10022.64019372, -12509.07807217,  -8077.30383941,  -7964.13296659,
                       -14361.88766202, -13910.99623182, -13936.53426527, -34833.81498682,
                       -28282.58885647, -19432.17640984, -12839.53720997, -14943.34999686,
                       -19473.17909211, -30943.62377719, -26322.96645724, -44185.31075859,
                       -18305.69197916, -29297.10158134, -18929.46219474, -35741.42796924,
                       -37958.82196459, -65424.30589802, -18305.69197916, -29297.10158134,
                       -18929.46219474, -45307.70596895, -48035.24434893, -75795.15002644,
                       -11224.17786938,  -6928.28917534,  -4316.037138,    -4770.62854206,
                       -7629.61899295,  -4021.62984035]


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
           "S10", "S11", "S12", "S13", "S14"]
#persons = ["P01", "P02", "P03", "P04", "P05", "P06"]
repetition = ["N01", "N02"]

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}


def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    @param data: numpy integer matrix
        Sensor data (all features)
    @return: numpy integer matrix
        Selection of features
    """

    # included-excluded
    features_delete = np.arange(68, 74)

    return np.delete(data, features_delete, 1)


def save_data_csv(data, filename):
    '''
    Saves a recording into a CSV file

    @param data: recording
    @param filename: path for storing the recording
    '''
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(headers)
        for d in data:
            spamwriter.writerow(d)
    return


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



def interpolate(data):
    '''
    Interpolates the sequences per channel for generating spline approximation funtions of a short period of time
    The spline approximation functions are derivated for generating virtual IMUs

    @param data: recording
    @return derivatives: @nd derivative of local spline interpolations of the data per channel
    '''

    """
    g, ax_x = plt.subplots(4, sharex=False)
    line1, = ax_x[0].plot([], [], '-b', label='blue')
    line2, = ax_x[1].plot([], [], '-b', label='blue')
    line3, = ax_x[2].plot([], [], '-b', label='blue')
    line4, = ax_x[3].plot([], [], '-b', label='blue')
    """

    prima2_data = np.zeros(data.shape)
    prima2_data[:, 0] = data[:, 0]
    prima2_data[:, 1] = data[:, 1]

    # int_data = np.zeros(data.shape)
    # int_data[:, 0] = data[:, 0]
    # int_data[:, 1] = data[:, 1]

    # prima1_data = np.zeros(data.shape)
    # prima1_data[:, 0] = data[:, 0]
    # prima1_data[:, 1] = data[:, 1]

    data_ext = np.zeros((data.shape[0] + 12, data.shape[1]))
    data_ext[6:-6] = data[:]
    time_x = range(-6, data_ext.shape[0] + 6)
    for ms in range(6, data.shape[0]):
        #tck = scipy.interpolate.CubicSpline(data[:, 0], data[:, sid])
        #derivatives[:, sid] = tck(data[:, 0], 2)

        sys.stdout.write("\rFrame {}".format(ms))
        sys.stdout.flush()

        x = time_x[ms - 6: ms + 6]
        for sid in range(2, data.shape[1]):
            # sys.stdout.write('\r' + 'x size ' + str(len(x)) + 'y size {}' + str(len(y[:, ms])))
            # sys.stdout.flush()
            tck = scipy.interpolate.splrep(x, data_ext[ms - 6: ms + 6, sid], s=0, k=5)
            # y_new = scipy.interpolate.splev(ms, tck)
            # y_new_prima_1 = scipy.interpolate.splev(ms, tck, der=1)
            y_new_prima_2 = scipy.interpolate.splev(ms, tck, der=2)
            # int_data[ms, sid] = y_new
            # prima1_data[ms, sid] = y_new_prima_1
            prima2_data[ms, sid] = y_new_prima_2

    """
    # Graphic Vals for X in T
    line1.set_xdata(data[:, 0])
    line1.set_ydata(data[:, 2])

    #line2.set_xdata(int_data[:, 0])
    #line2.set_ydata(int_data[:, 2])

    #line3.set_xdata(prima1_data[:, 0])
    #line3.set_ydata(prima1_data[:, 2])

    line4.set_xdata(prima2_data[:, 0])
    line4.set_ydata(prima2_data[:, 2])

    ax_x[0].relim()
    ax_x[0].autoscale_view()
    ax_x[1].relim()
    ax_x[1].autoscale_view()
    ax_x[2].relim()
    ax_x[2].autoscale_view()
    ax_x[3].relim()
    ax_x[3].autoscale_view()

    plt.draw()
    plt.pause(2.0)
    """

    return prima2_data


def generate_derivatives(ids):
    '''
    Generate the files containing the derivatives of the sequences, what will be called virtual IMUs

    THe functions will store files with the derivatives of the Mocap data for the subjects, specified with IDs,
    and stored the files under the same name of the MoCAP recordings, keeping up the same structure of the LARA
    dataset

    @param ids: IDS of the subjects for which derivatives will be computed
    '''

    FOLDER_PATH = '/path_to_theLARa_MOCAP_dataset/'
    folder_derivative = "path_to_theLARa_Virtual_dataset/"

    # Recording names, refer to the naming of the files in LARa dataset
    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    for P in persons:
        if P not in ids:
            print("\nNo Person in expected IDS {}".format(P))
        else:
            for r, R in enumerate(recordings):
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
                    if P == "S07" and SCENARIO[R] == "L01":
                        annotator_file = "A03"
                    if P == "S14" and SCENARIO[R] == "L03":
                        annotator_file = "A19"
                    if P == "S11" and SCENARIO[R] == "L01":
                        annotator_file = "A03"
                    if P == "S11" and R in ["R04", "R08", "R09", "R10", "R11", "R12", "R13", "R15"]:
                        annotator_file = "A02"
                    if P == "S13" and R in ["R28"]:
                        annotator_file = "A01"
                    if P == "S13" and R in ["R29", "R30"]:
                        annotator_file = "A11"
                    if P == "S09" and R in ["R28", "R29"]:
                        annotator_file = "A01"
                    if P == "S09" and R in ["R21", "R22", "R23", "R24", "R25"]:
                        annotator_file = "A11"

                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_derivative = "{}/{}_{}_{}_{}_{}_der_data.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        # getting data
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("\nFiles loaded")
                        data = select_columns_opp(data)
                        print("Columns selected")
                    except:
                        print("\n In generating data, selecting Columns\nNo file {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        # Interpolating
                        print("Interpolating")
                        data = interpolate(data)
                    except:
                        print("\n In generating data, Interpolatin the data {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        print("\nsaving")
                        save_data_csv(data, folder_derivative + file_name_derivative)
                    except:
                        print(
                            "\n In generating data, Error Saving \n"
                            "Error getting the data {}".format(folder_derivative + file_name_derivative))
                        continue
    return


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


def compute_max_min(ids):
    '''
    Compute the max and min values for normalizing the data.

    print max and min.
    These values will be computed only once and the max min values
    will be place as constants

    @param ids: ids for train
    '''

    FOLDER_PATH = "path_to_theLARa_Virtual_dataset/"

    # Recording names, refer to the naming of the files in LARa dataset
    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    max_values_total = np.zeros((126))
    min_values_total = np.ones((126)) * 1000000

    accumulator_mean_measurements = np.empty((0, 126))
    accumulator_std_measurements = np.empty((0, 126))

    for P in persons:
        if P in ids:
            accumulator_measurements = np.empty((0, 126))
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["P01", "P02", "P03", "P04", "P05", "P06"]:
                    S = "S01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'P07' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'P14' and SCENARIO[r] == 'S03':
                        annotator_file = "A19"
                    if P == 'P11' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'P11' and r in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    file_name_norm = "{}/{}_{}_{}_{}_{}_der_data.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("Files loaded")
                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_norm))

                    try:
                        print("Getting the max and min")
                        data_t, data_x, data_y = divide_x_y(data)
                        del data_t
                        del data_y

                        max_values = np.max(data_x, axis=0)
                        min_values = np.min(data_x, axis=0)

                        max_values_total = np.max((max_values, max_values_total), axis=0)
                        min_values_total = np.min((min_values, min_values_total), axis=0)

                        accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
                        print("Accumulated")
                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_norm))

            mean_values = np.mean(accumulator_measurements, axis=0)
            std_values = np.std(accumulator_measurements, axis=0)

            accumulator_mean_measurements = np.append(accumulator_mean_measurements, [mean_values], axis=0)
            accumulator_std_measurements = np.append(accumulator_std_measurements, [std_values], axis=0)

    try:
        mean_values = np.mean(accumulator_mean_measurements, axis=0)
        std_values = np.max(accumulator_std_measurements, axis=0)
        mean_values = np.around(mean_values, decimals=4)
        std_values = np.around(std_values, decimals=5)
        print("Max values \n{}".format(max_values_total))
        print("Min values \n{}".format(min_values_total))
        print("Mean values \n{}".format(mean_values))
        print("Std values \n{}".format(std_values))
    except:
        print("Error computing statistics")
    return

def normalize(data):
    """
    Max-Min Normalization of all sensor channels

    @param data: numpy integer matrix
        Sensor data
    @return:
        Normalized sensor data
    """
    try:
        max_list, min_list = np.array(NORM_MAX_THRESHOLDS), np.array(NORM_MIN_THRESHOLDS)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
    except:
        raise ("Error in normalization")

    return data



def norm_mean_std(data):
    """
    Zero Mean and Unit variance Normalization of all sensor channels

    @param data: numpy integer matrix
        Sensor data
    @return:
        Normalized sensor data
    """
    mean_values = np.array([-1.200e-03,  3.000e-04, -1.000e-04, -3.000e-03, -9.300e-03, -1.377e-01,
                            -1.200e-03,  3.000e-04, -1.000e-04, -2.000e-03, -1.320e-02, -1.730e-01,
                            2.050e-02, -6.800e-03, -7.500e-03, -2.100e-03, -6.500e-03, -9.690e-02,
                            3.000e-02,  1.850e-02, -2.420e-02, -5.010e-02,  1.310e-02, -2.460e-02,
                            -2.000e-03, -5.000e-03, -5.700e-03, -1.690e-02,  7.600e-03,  3.770e-02,
                            9.500e-03, -9.700e-03,  2.300e-03, -2.980e-02, -1.700e-03,  3.090e-01,
                            -2.800e-03, -3.170e-02,  3.100e-02, -2.860e-02,  7.100e-03, -1.106e-01,
                            -3.600e-03, -5.300e-03, -5.400e-03, -2.310e-02,  8.200e-03,  1.768e-01,
                            9.500e-03, -9.700e-03,  2.300e-03, -1.220e-02,  3.100e-02,  3.072e-01,
                            3.600e-02,  1.650e-02, -3.220e-02, -3.870e-02,  2.660e-02,  3.740e-02,
                            3.600e-02,  1.650e-02, -3.220e-02, -4.330e-02,  3.800e-02,  5.610e-02,
                            7.500e-03,  1.290e-02, -1.320e-02, -2.100e-03, -6.500e-03, -9.690e-02,
                            1.720e-02,  2.680e-02, -3.160e-02,  3.990e-02, -2.910e-02, -1.980e-02,
                            1.070e-02, -7.100e-03, -7.600e-03,  1.580e-02, -1.130e-02,  3.690e-02,
                            -2.300e-03,  7.700e-03, -7.500e-03,  1.480e-02, -2.230e-02,  3.061e-01,
                            2.560e-02, -2.870e-02,  2.720e-02,  2.540e-02, -2.010e-02, -1.071e-01,
                            7.300e-03, -7.100e-03, -3.900e-03,  2.570e-02, -1.520e-02,  1.755e-01,
                            -2.300e-03,  7.700e-03, -7.500e-03,  3.170e-02,  9.600e-03,  3.067e-01,
                            9.600e-03,  2.170e-02, -2.500e-02,  4.540e-02, -1.590e-02,  4.590e-02,
                            9.600e-03,  2.170e-02, -2.500e-02,  5.600e-02, -1.300e-02,  6.340e-02,
                            -2.900e-03, -5.000e-04,  7.000e-04, -5.000e-04, -1.900e-03,  3.730e-02])

    mean_values = np.reshape(mean_values, [1, 126])

    std_values = np.array([39.67105,  31.09914,  44.66373, 104.49076,  74.05434,  98.78674,  39.67105,
                           31.09914,  44.66373,  96.31244,  66.80233,  89.38161,  74.54627,  72.66876,
                           91.49264,  73.4471,   52.05317,  69.44388, 237.04959, 182.36846, 212.70275,
                           163.11063, 127.97767, 111.61515,  89.99005, 122.80086, 115.70241, 132.34554,
                           127.97426, 132.24224, 286.37277, 169.39669, 225.1807,  199.15293, 166.19869,
                           208.88805,  73.65657,  77.88164,  75.9377,   94.48174,  71.24829, 100.96765,
                           100.61155,  87.64971, 118.83619, 283.77666, 392.68112, 272.8422,  286.37277,
                           169.39669, 225.1807,  808.57494, 536.05044, 440.01609, 337.49112, 265.74831,
                           269.40528, 362.01252, 271.71638, 378.73486, 337.49112, 265.74831, 269.40528,
                           429.14616, 311.88314, 523.12388, 265.27743, 170.50911, 248.23945,  73.4471,
                           52.05317,  69.44388, 453.75003, 490.27469, 433.02178, 425.69749, 314.67207,
                           370.20253, 178.01386, 124.30709, 129.96297, 115.79145, 115.28095, 128.83746,
                           341.88668, 308.62789, 342.54978, 439.41588, 457.241,   375.0191,  303.78202,
                           239.78097, 251.88022, 171.28521, 132.10398, 204.81751, 214.33376, 141.84135,
                           137.5641,  551.03716, 507.66082, 595.38634, 341.88668, 308.62789, 342.54978,
                           856.33941, 637.96056, 929.2445,  390.08437, 406.15986, 416.71797, 460.93649,
                           448.14604, 532.80464, 390.08437, 406.15986, 416.71797, 576.30111, 530.26703,
                           591.98005,  86.23326,  65.583,    57.84249, 103.79382,  80.50423, 78.304])

    std_values = np.reshape(std_values, [1, 126])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0

    #data_norm = (data - mean_array) / std_array

    return data_norm


################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None):
    '''
    creates files for each of the sequences extracted from a file
    following a sliding window approach

    returns a numpy array

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    '''


    FOLDER_PATH = '/path_to_LARa_Mocap_for_annotations/'
    folder_derivative = "/path_to_LARa_Mocap_for_annotations/"

    # Recording names, refer to the naming of the files in LARa dataset
    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_seq = 0
    hist_classes_all = np.zeros(NUM_CLASSES)

    for P in persons:
        if P not in ids:
            print("\nNo Person in expected IDS {}".format(P))
        else:
            for r, R in enumerate(recordings):
                # Selecting the proportions of the train, val or testing according to the quentity of
                # recordings per subject, as there are not equal number of recordings per subject
                # see dataset for checking the recording files per subject
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[R]
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

                    file_name_norm = "{}/{}_{}_{}_{}_{}_der_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        # getting data
                        data = csv_reader.reader_data(folder_derivative + file_name_norm)
                        print("\nFiles loaded")
                    except:
                        print("\n In generating data, No file {}".format(folder_derivative + file_name_norm))
                        continue

                    try:
                        # Getting labels and attributes
                        labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        class_labels = np.where(labels[:, 0] == 7)[0]
                        print("\nGet labels")

                        # Deleting rows containing the "none" class
                        data = np.delete(data, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)
                        print("\nDeleting none rows")

                        # halving the frequency, as Mbientlab or MotionMiners sensors use 100Hz
                        downsampling = range(0, data.shape[0], 2)
                        data = data[downsampling]
                        labels = labels[downsampling]
                        data_t, data_x, data_y = divide_x_y(data)
                        del data_t
                        print("\nDownsampling")

                    except:
                        print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        # checking if annotations are consistent
                        data_x = norm_mean_std(data_x)
                        if np.sum(data_y == labels[:, 0]) == data_y.shape[0]:

                            # Sliding window approach
                            print("Starting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, labels.astype(int), sliding_window_length,
                                                             sliding_window_step, label_pos_end=False)
                            print("Windows are extracted")

                            # Statistics
                            hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                            hist_classes_all += hist_classes
                            print("Number of seq per class {}".format(hist_classes_all))

                            for f in range(X.shape[0]):
                                try:

                                    sys.stdout.write(
                                        '\r' + 'Creating sequence file number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)

                                    # Storing the sequences
                                    obj = {"data": seq, "label": y[f], "labels": y_all[f]}
                                    f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                                    f.close()

                                    counter_seq += 1

                                except:
                                    raise ('\nError adding the seq')

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

    return f


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

    return f



def create_dataset():
    '''
    create dataset
    - Segmentation
    - Storing sequences

    '''
    train_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10"]
    train_final_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12"]
    val_ids = ["S05", "S11", "S12"]
    test_ids = ["S06", "S13", "S14"]

    all_data = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    # general_statistics(train_ids)

    base_directory = '/data/fmoya/HAR/datasets/Virtual_IMUs/'

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train)
    generate_data(val_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val)
    generate_data(test_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test)

    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return


if __name__ == '__main__':
    # Creating dataset for LARa virtual IMUs
    # Set the path to where the segmented windows will be located
    # This path will be needed for the main.py

    # Dataset (extracted segmented windows) will be stored in a given folder by the user,
    # However, inside the folder, there shall be the subfolders (sequences_train, sequences_val, sequences_test)
    # These folders and subfolfders gotta be created manually by the user
    # This as a sort of organisation for the dataset
    # Virtual_IMUs/sequences_train
    # Virtual_IMUs/sequences_val
    # Virtual_IMUs/sequences_test


    train_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10"]
    train_final_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12"]
    val_ids = ["S05", "S11", "S12"]
    test_ids = ["S06", "S13", "S14"]

    #all_data = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    all_data = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    # compute_max_min(train_final_ids)

    # Generate derivatives of the Mocap dataset. Files will be stored following Mocap Structure
    # generate_derivatives(all_data)

    # Segments sequences from the recordings and dump them in individual annotated files for Torch
    # create_dataset()

    print("Done")
