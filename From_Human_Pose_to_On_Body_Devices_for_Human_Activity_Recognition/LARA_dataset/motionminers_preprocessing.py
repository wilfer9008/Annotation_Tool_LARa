'''
Created on June 18, 2020

@author: fmoya
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import csv_reader
import csv_save

from sliding_window import sliding_window
import pickle

from IMUSequenceContainer import IMUSequenceContainer


dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/Data_Real/DFG-Data/"

headers_annotated = ['Time', 'Class', 'AccX_L', 'AccY_L', 'AccZ_L', 'GyrX_L', 'GyrY_L', 'GyrZ_L',
           'MagX_L', 'MagY_L', 'MagZ_L', 'AccX_T', 'AccY_T', 'AccZ_T', 'GyrX_T', 'GyrY_T',
           'GyrZ_T', 'MagX_T', 'MagY_T', 'MagZ_T', 'AccX_R', 'AccY_R', 'AccZ_R', 'GyrX_R',
           'GyrY_R', 'GyrZ_R', 'MagX_R', 'MagY_R', 'MagZ_R']

headers = ['Time', 'AccX_L', 'AccY_L', 'AccZ_L', 'GyrX_L', 'GyrY_L', 'GyrZ_L',
           'MagX_L', 'MagY_L', 'MagZ_L', 'AccX_T', 'AccY_T', 'AccZ_T', 'GyrX_T', 'GyrY_T',
           'GyrZ_T', 'MagX_T', 'MagY_T', 'MagZ_T', 'AccX_R', 'AccY_R', 'AccZ_R', 'GyrX_R',
           'GyrY_R', 'GyrZ_R', 'MagX_R', 'MagY_R', 'MagZ_R']

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}

NUM_CLASSES = 8
NUM_ATTRIBUTES = 19

def visualize(data):
    fig = plt.figure()
    axis_list = []
    plot_list = []
    axis_list.append(fig.add_subplot(311))
    axis_list.append(fig.add_subplot(312))
    axis_list.append(fig.add_subplot(313))

    plot_list.append(axis_list[0].plot([], [], '-r', label='T', linewidth=0.15)[0])
    plot_list.append(axis_list[0].plot([], [], '-b', label='L', linewidth=0.20)[0])
    plot_list.append(axis_list[0].plot([], [], '-g', label='R', linewidth=0.20)[0])

    plot_list.append(axis_list[1].plot([], [], '-r', label='T', linewidth=0.15)[0])
    plot_list.append(axis_list[1].plot([], [], '-b', label='LR', linewidth=0.30)[0])

    plot_list.append(axis_list[2].plot([], [], '-r', label='T', linewidth=0.15)[0])
    plot_list.append(axis_list[2].plot([], [], '-b', label='LR', linewidth=0.30)[0])

    #  AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ
    # data [T, 28] with L [:, 1:10] T [:, 10:19] R [:, 19:]
    # 1,    2,      3,    4,    5,    6,    7,   8,     9
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    # 10,    11,   12,    13,   14,  15,   16,   17,   18
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    # 19,    20,   21,    22,   23,  24,   25,   26,   27
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    data_range_init = 0
    data_range_end = data.shape[0]
    time_x = np.arange(data_range_init, data_range_end)

    print("Range init {} end {}".format(data_range_init, data_range_end))
    T = np.linalg.norm(data[data_range_init:data_range_end, 10:13], axis=1)
    L = np.linalg.norm(data[data_range_init:data_range_end, [1, 2, 3]], axis=1)
    R = np.linalg.norm(data[data_range_init:data_range_end, [19, 20, 21]], axis=1)

    Arms = (L + R) / 2
    plot_list[0].set_data(time_x, T)
    plot_list[1].set_data(time_x, L)
    plot_list[2].set_data(time_x, R)

    plot_list[3].set_data(time_x, T)
    plot_list[4].set_data(time_x, Arms)

    plot_list[5].set_data(time_x, T)
    plot_list[6].set_data(time_x, Arms)

    axis_list[0].relim()
    axis_list[0].autoscale_view()
    axis_list[0].legend(loc='best')

    axis_list[1].relim()
    axis_list[1].autoscale_view()
    axis_list[1].legend(loc='best')

    axis_list[2].relim()
    axis_list[2].autoscale_view()
    axis_list[2].legend(loc='best')

    fig.canvas.draw()
    plt.show()
    # plt.pause(2.0)

    return



def read_extracted_data(path, skiprows = 1):
    '''
    gets data from csv file
    data contains 3 columns, start, end and label

    returns a numpy array

    @param path: path to file
    '''

    annotation_original = np.loadtxt(path, delimiter=',', skiprows=skiprows)
    return annotation_original


def save_data(data, filename, headers_bool=False, seq_annotated=False):

    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if headers_bool:
            if seq_annotated:
                spamwriter.writerow(headers_annotated)
            else:
                spamwriter.writerow(headers)
        for d in data:
            spamwriter.writerow(d)

    return


def extract_data_flw(IMUcontainer):
    #dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/Dock_data_organized/"
    #new_dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/flw_data/"
    #dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/"
    dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/flw_data/"
    new_dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/flw_data/"
    sensors_persons_mapping = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    sensors_persons_mapping = ["S14"]

    for skey in sensors_persons_mapping:
        print("Sensor session {}: Person ".format(skey))
        imu_file_path = dataset_path + skey + "/"

        ser_file = IMUSequenceContainer.read_exls3(path=imu_file_path)

        # timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ, Q0,Q1,Q2,Q3, Vbat
        # data [L, T, R]
        print("Loading Data")
        data = IMUSequenceContainer.get_data()
        print("Data Loaded")

        save_data(data, new_dataset_path + skey + "/" + skey + '_data.csv')

    return data


def read_data_flw():
    #dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/Dock_data_organized/"
    dataset_path = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/Extracted_singleRecording_per_subject/"
    #dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/flw_data/"
    #dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/"
    #dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/flw_data/"
    #dataset_path = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/LARa_dataset/" \
    #               "MotionMiners_FLW/flw_data/"
    #new_dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/motionminers/flw_data/"
    sensors_persons_mapping = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    sensors_persons_mapping = ["S10"]

    for skey in sensors_persons_mapping:
        print("Sensor session {}: Person ".format(skey))
        imu_file_path = dataset_path + skey + "/" + skey + "_data.csv"

        # timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ, Q0,Q1,Q2,Q3, Vbat
        # data [L, T, R]
        print("Loading Data")
        data = read_extracted_data(imu_file_path, skiprows=0)
        print("Data Loaded")
        # timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ, Q0,Q1,Q2,Q3, Vbat
        # data [L, T, R]

        visualize(data)

    return data


def visualise_sequences_flw():
    #dataset_path = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/MotionMiners_FLW/flw_sequences/"
    #dataset_path = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_1st_cut/"
    dataset_path = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_data_recordings/"
    sensors_persons_mapping = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    sensors_persons_mapping = ["S10"]
    recording = "R20"
    scenario = SCENARIO[recording]

    for skey in sensors_persons_mapping:
        print("Sensor session {}: Person Recording {}".format(skey, recording))
        imu_file_path = dataset_path + "{}/{}_{}_{}.csv".format(skey, scenario, skey, recording)

        print("Loading Data")
        data = read_extracted_data(imu_file_path, skiprows=1)
        print("Data Loaded")
        visualize(data)

    return


def correct_headers_data_sequences_flw():
    dataset_path = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_12000/"
    #dataset_path = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/MotionMiners_FLW/" \
    #               "flw_recordings_12000/"
    sensors_persons_mapping = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]


    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
    for skey in sensors_persons_mapping:
        for recording in recordings:
            scenario = SCENARIO[recording]
            print("Sensor session {}: Person Recording {}".format(skey, recording))
            imu_file_path = dataset_path + "{}/{}_{}_{}.csv".format(skey, scenario, skey, recording)

            try:
                print("Loading Data")
                data = read_extracted_data(imu_file_path, skiprows=1)

                save_data(data, imu_file_path, headers_bool=True, seq_annotated=True)
            except:
                print("Error in {}".format(imu_file_path))

    return


def create_sequences_flw():
    #dataset_path = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/" \
    #               "MotionMiners_FLW/flw_sequences/"
    #dataset_path_sequences = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/" \
    #                         "MotionMiners_FLW/flw_sequences_12000/"

    dataset_path = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/Extracted_singleRecording_per_subject/"
    dataset_path_sequences = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_data_recordings/"

    sensors_persons_mapping = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    sensors_persons_mapping = ["S10"]

    #start_sequences = [538, 559, 542, 560, 564, 552, 546, 554, 594, 550, 730, 545, 559,
    #                   552, 583, 605, 554, 538, 1022, 551, 533, 537, 556, 0, 544, 539,
    #                   524, 535, 0, 0]

    start_sequences = [17221, 31321, 148237, 181438, 194992, 209130, 223295, 236877, 250409,
                       264214, 278358, 293233, 306902, 320090, 333644, 347400, 542562, 556673,
                       569673, 603247, 617189, 681908, 718504, 948339, 961545, 974743, 987731,
                       1001630, 1015420, 1029560]

    stop_sequences = [30020, 44158, 161047, 194274, 207779, 221852, 236258, 249697, 263246, 277149,
                      291353, 306272, 319771, 332932, 346508, 360275, 555372, 569521, 583042, 616111,
                      629993, 694778, 731411, 961182, 974384, 987647, 1000560, 1014550, 1028160, 1042380]

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
    for skey in sensors_persons_mapping:
        for ridx, recording in enumerate(recordings):
            scenario = SCENARIO[recording]
            print("Sensor session {}: Person Recording {}".format(skey, recording))
            imu_file_path = dataset_path + skey + "/" + skey + "_data.csv"
            #imu_file_path = dataset_path + "{}/{}_{}_{}.csv".format(skey, scenario, skey, recording)

            try:
                print("Loading Data")
                data = read_extracted_data(imu_file_path, skiprows=1)
                print("Data Loaded")

                recordings_counter = 0
                file_name_data = dataset_path_sequences + "{}/{}_{}_{}.csv".format(skey, scenario, skey, recording)
                #sequence = data[start_sequences[ridx]:start_sequences[ridx]+12000]
                sequence = data[start_sequences[ridx]:stop_sequences[ridx]]
                sequence[:, 0] = sequence[:, 0] - sequence[0, 0]
                save_data(sequence, file_name_data, headers_bool=True)
            except:
                print("Error in Sensor session {}: Person Recording {}".format(skey, recording))

    return


def create_annotated_sequences():
    dir_dataset_mocap = "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/15_Annotated_Dataset_Corrected/"
    dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_data_recordings/"
    dataset_path_imu_sequences = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_annotated_revised_2/"
    #persons = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    persons = ["S10"]

    annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
                 "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
                 "S13": "A08", "S14": "A06"}

    wframes = {'S07': [535, 525, 544, 0, 535,
                       520, 563, 550, 556, 539,
                       540, 535, 545, 565, 545,
                       776, 558, 536, 540, 570,
                       553, 543, 528, 523, 520,
                       558, 558, 514, 534, 544],
               'S08': [552, 535, 550, 621, 500,
                       545, 540, 544, 500, 530,
                       560, 536, 525, 557, 531,
                       558, 555, 559, 574, 546,
                       550, 549, 537, 575, 541,
                       535, 559, 544, 536, 524],
               'S09': [561, 558, 534, 543, 545,
                       550, 539, 586, 544, 542,
                       557, 536, 532, 539, 561,
                       555, 550, 538, 559, 560,
                       0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0],
               'S10': [538, 559, 542, 560, 564,
                       552, 546, 554, 594, 550,
                       730, 545, 559, 552, 583,
                       605, 554, 538, 1022, 551,
                       533, 537, 556, 0, 544, 539,
                       524, 535, 0, 0],
               'S11': [516, 528, 549, 540, 534,
                       560, 563, 546, 531, 539,
                       537, 553, 551, 500, 513,
                       500, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0],
               'S12': [0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 539, 548, 521, 551,
                       535, 520, 513, 503, 564,
                       486, 515, 623, 569, 542,
                       796, 520, 505, 711, 516,
                       511],
               'S13': [530, 523, 535, 513, 546,
                       578, 534, 866, 549, 551,
                       561, 512, 500, 501, 544,
                       500, 530, 506, 533, 585,
                       511, 563, 550, 502, 504,
                       506, 538, 582, 506, 500],
               'S14': [510, 513, 521, 505, 500,
                       516, 520, 512, 535, 576,
                       480, 500, 518, 508, 474,
                       493, 496, 500, 501, 506,
                       496, 519, 503, 497, 498,
                       503, 509, 508, 515, 514]
               }

    wframe_mocap = {'S07': [351, 448, 484, 0, 468, 324, 410, 392, 442, 445, 450, 371, 315, 388, 452,
                            274, 525, 422, 387, 389, 332, 378, 240, 515, 357, 430, 239, 349, 400, 260],
                    'S08': [281, 171, 264, 314, 0, 272, 197, 287, 0, 148, 193, 152, 149, 181, 204,
                            219, 185, 279, 235, 243, 307, 298, 146, 276, 197, 368, 139, 159, 144, 184],
                    'S09': [537, 265, 268, 185, 209, 167, 280, 181, 187, 228, 239, 268, 192, 193, 209,
                            210, 426, 247, 294, 241, 231, 204, 242, 237, 191, 186, 196, 111, 261, 0],
                    'S10': [345, 266, 410, 387, 369, 467, 261, 411, 417, 274, 356, 309, 416, 403, 393,
                            500, 311, 439, 352, 298, 410, 440, 297, 0, 410, 277, 366, 278, 0, 0],
                    'S11': [423, 349, 284, 381, 357, 314, 205, 222, 188, 239, 281, 229, 318, 0, 242,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'S12': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 213, 204, 189, 184, 147, 316, 269, 207,
                            492, 674, 160, 268, 1284, 184, 143, 411, 197, 248, 143, 147],
                    'S13': [253, 241, 310, 243, 180, 195, 194, 208, 179, 163, 277, 175, 211, 232,
                            216, 259, 180, 198, 275, 209, 242, 224, 209, 234, 266, 257, 198, 138, 162, 243],
                    'S14': [393, 295, 302, 480, 249, 288, 273, 157, 257, 178, 314, 425, 341, 297, 354,
                            298, 322, 317, 356, 239, 209, 280, 314, 287, 262, 321, 246, 232, 633, 223]
                    }

    #start_sequences_imus = [535, 525, 544, 0, 535, 520, 563, 550, 556, 539, 540, 535, 545, 565,
    #                        545, 776, 558, 536, 540, 570, 553, 543, 528, 523, 520, 558, 558,
    #                        514, 534, 544]

    start_sequences_imus = wframes[persons[0]]

    #start_sequences_mocap = [351, 448, 484, 0, 468, 324, 410, 392, 442, 445, 450, 371,
    #                         315, 388, 452, 274, 525, 422, 387, 389, 332, 378, 240, 515,
    #                         357, 430, 239, 349, 400, 260]

    start_sequences_mocap = wframe_mocap[persons[0]]

    repetition = ["N01", "N02"]

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    for idp, P in enumerate(persons):
        for counter, r in enumerate(recordings):
            for N in repetition:
                print("\n\n------------------------------------------------------------------")
                annotator_file = annotator[P]
                if P == 'S07' and SCENARIO[r] == 'L01':
                    annotator_file = "A03"
                if P == 'S09' and r in ['R28', 'R29']:
                    annotator_file = "A01"
                if P == 'S09' and r in ['R21', 'R22', 'R23', 'R24', 'R25']:
                    annotator_file = "A11"
                if P == 'S11' and SCENARIO[r] == 'L01':
                    annotator_file = "A03"
                if P == 'S11' and r in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                    annotator_file = "A02"
                if P == 'S13' and r in ['R28']:
                    annotator_file = "A01"
                if P == 'S13' and r in ['R29', 'R30']:
                    annotator_file = "A11"
                if P == 'S14' and SCENARIO[r] == 'L03':
                    annotator_file = "A19"

                file_name_imu = '{}/{}_{}_{}.csv'.format(P, SCENARIO[r], P, r)
                file_name_imu_attr = '{}/{}_{}_{}_labels.csv'.format(P, SCENARIO[r], P, r)
                file_name_mocap_attr = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, SCENARIO[r], P, r, annotator_file, N)

                print("\n{}\n".format(file_name_imu))

                if not os.path.exists(dir_dataset_mocap + file_name_mocap_attr):
                    print("1 - No file in {}".format(dir_dataset_mocap + file_name_mocap_attr))
                    continue

                try:
                    data_labels = csv_reader.reader_labels(dir_dataset_mocap + file_name_mocap_attr)
                    data_labels = data_labels[start_sequences_mocap[counter]:]
                except:
                    print("2 - Error getting annotated labels in {}".format(dir_dataset_mocap + file_name_mocap_attr))
                    continue

                try:
                    data_imu = read_extracted_data(dataset_path_imu + file_name_imu, skiprows=1)
                    print("\nFiles loaded\n")
                except:
                    print("3 - Error getting annotated labels in {}".format(dataset_path_imu + file_name_imu))
                    continue

                try:
                    idxs_labels = np.arange(0, data_labels.shape[0], 2)
                    data_labels = data_labels[idxs_labels]

                    sequence = data_imu[start_sequences_imus[counter]: start_sequences_imus[counter] +
                                                                       data_labels.shape[0]]
                    sequence[:, 0] = sequence[:, 0] - sequence[0, 0]

                    annotated_sequence = np.zeros((sequence.shape[0], sequence.shape[1] + 1))
                    annotated_sequence[:, 0] = sequence[:, 0]
                    annotated_sequence[:, 1] = data_labels[:, 0]
                    annotated_sequence[:, 2:] = sequence[:, 1:]

                except:
                    print("4 - Error getting annotated labels in {}".format(dataset_path_imu + file_name_imu))
                    continue

                try:
                    if annotated_sequence.shape[0] == data_labels.shape[0]:
                        save_data(annotated_sequence, dataset_path_imu_sequences + file_name_imu, headers_bool=True,
                                  seq_annotated=True)
                        csv_save.save_attr_csv(data_labels.astype(int),
                                               filename=dataset_path_imu_sequences + file_name_imu_attr)
                except:
                    print("4 - Error saving atts in {}".format(dataset_path_imu_sequences + file_name_imu_attr))

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

    count_l = 0
    idy = 0
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:

        # Label from the middle
        if False:
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

def statistics_measurements():
    '''
    Computes some statistics over the channels for the entire training data

    returns a max_values, min_values, mean_values, std_values
    '''

    #dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_12000/"
    dataset_path_imu = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/" \
                             "MotionMiners_FLW/flw_recordings_12000/"

    train_final_ids = ["S07", "S08", "S09", "S10", "S11", "S12"]

    persons = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    accumulator_measurements = np.empty((0, 27))
    for P in persons:
        if P not in train_final_ids:
            print("\n6 No Person in expected IDS {}".format(P))
        else:
            for r, R in enumerate(recordings):
                S = SCENARIO[R]
                file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
                file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                print("------------------------------\n{}\n{}".format(file_name_data, file_name_label))
                try:
                    # getting data
                    data = read_extracted_data(dataset_path_imu + file_name_data, skiprows=1)
                    data_x = data[:, 2:]
                    accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
                    print("\nFiles loaded")
                except:
                    print("\n1 In loading data,  in file {}".format(dataset_path_imu + file_name_data))
                    continue

    try:
        max_values = np.max(accumulator_measurements, axis=0)
        min_values = np.min(accumulator_measurements, axis=0)
        mean_values = np.mean(accumulator_measurements, axis=0)
        std_values = np.std(accumulator_measurements, axis=0)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    return max_values, min_values, mean_values, std_values



def norm_mbientlab(data):

    mean_values = np.array([-1983.7241, 1437.0780, 1931.1834, -1.7458, 10.9168,
                            -15.1849, 2804.7062, -2651.8638, -1798.9489, 74.8857,
                            -3773.9069, -1022.3879, -7.3275, 0.8866, -2.8524,
                            -180.2325, 1760.7540, 679.8919, 1687.7334, 1292.4192,
                            2271.0754, 15.0217, -10.7949, 18.0103, -1965.7448,
                            -2079.8198, -2488.2345])
    mean_values = np.reshape(mean_values, [1, 27])
    std_values = np.array([1647.4232, 1973.3758, 1433.9222, 786.4119, 1020.2077,
                           929.3821, 2272.4711, 2725.6133, 1977.3836, 694.1986,
                           614.7125, 1207.0166, 298.7505, 588.4253, 266.5462,
                           808.6021, 383.0109, 1194.2341, 1802.0538, 1999.7361,
                           1440.8832, 891.9623, 1048.0871, 961.4529, 2577.3050,
                           2819.8947, 1793.8098])
    std_values = np.reshape(std_values, [1, 27])

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
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None,
                  identity_bool = False, usage_modus = 'train'):
    '''
    creates files for each of the sequences extracted from a file
    following a sliding window approach


    returns a numpy array

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    '''

    dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/LARa_dataset_motionminers/"
    #dataset_path_imu = "/Users/fernandomoyarueda/Documents/Doktorado/DFG_project/Data/" \
    #                         "MotionMiners_FLW/flw_recordings_12000/"

    persons = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    idx_train = {"S07": 21, "S08": 21, "S09": 21, "S10": 11, "S11": 11, "S12": 23, "S13": 21, "S14": 21}
    idx_val = {"S07": 26, "S08": 26, "S09": 26, "S10": 16, "S11": 13, "S12": 26, "S13": 26, "S14": 26}
    idx_test = {"S07": 31, "S08": 31, "S09": 31, "S10": 24, "S11": 16, "S12": 31, "S13": 31, "S14": 31}

    counter_seq = 0
    hist_classes_all = np.zeros((NUM_CLASSES))
    counter_file_label = -1

    #g, ax_x = plt.subplots(2, sharex=False)
    #line3, = ax_x[0].plot([], [], '-b', label='blue')
    #line4, = ax_x[1].plot([], [], '-b', label='blue')
    for P in persons:
        if P not in ids:
            print("\n6 No Person in expected IDS {}".format(P))
        else:
            if identity_bool:
                if usage_modus == 'train':
                    recordings = ['R{:02d}'.format(rec) for rec in range(1, idx_train[P])]
                elif usage_modus == 'val':
                    recordings = ['R{:02d}'.format(rec) for rec in range(idx_train[P], idx_val[P])]
                elif usage_modus == 'test':
                    recordings = ['R{:02d}'.format(rec) for rec in range(idx_val[P], idx_test[P])]
            else:
                recordings = ['R{:02d}'.format(rec) for rec in range(1, 31, 1)]
                # recordings = ['R{:02d}'.format(r) for r in range(1, 31, 2)]
            print("\nModus {} \n{}".format(usage_modus, recordings))
            for R in recordings:
                try:
                    S = SCENARIO[R]
                    file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
                    file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                    print("\n{}\n{}".format(file_name_data, file_name_label))
                    try:
                        # getting data
                        data = read_extracted_data(dataset_path_imu + file_name_data, skiprows=1)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
                        data_x = data[:, 2:]
                        print("\nFiles loaded")
                    except:
                        print("\n1 Error In loading data,  in file {}".format(dataset_path_imu + file_name_data))
                        continue

                    try:
                        # Getting labels and attributes
                        labels = csv_reader.reader_labels(dataset_path_imu + file_name_label)
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data_x = np.delete(data_x, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)

                        #data_t, data_x, data_y = divide_x_y(data)
                        #del data_t

                    except:
                        print(
                            "2 In generating data, Error getting the data {}".format(dataset_path_imu
                                                                                       + file_name_data))
                        continue

                    try:
                        # Graphic Vals for X in T
                        #line3.set_ydata(data_x[:, 0].flatten())
                        #line3.set_xdata(range(len(data_x[:, 0].flatten())))
                        #ax_x[0].relim()
                        #ax_x[0].autoscale_view()
                        #plt.draw()
                        #plt.pause(2.0)

                        data_x = norm_mbientlab(data_x)

                        #line4.set_ydata(data_x[:, 0].flatten())
                        #line4.set_xdata(range(len(data_x[:, 0].flatten())))
                        #ax_x[1].relim()
                        #ax_x[1].autoscale_view()
                        #plt.draw()
                        #plt.pause(2.0)
                    except:
                        print("\n3  In generating data, Plotting {}".format(dataset_path_imu + file_name_data))
                        continue

                    try:
                        # checking if annotations are consistent
                        if data_x.shape[0] == labels.shape[0]:
                            # Sliding window approach
                            print("\nStarting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, labels.astype(int), sliding_window_length,
                                                             sliding_window_step, label_pos_end=False)
                            print("\nWindows are extracted")

                            # Statistics

                            hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                            hist_classes_all += hist_classes
                            print("\nNumber of seq per class {}".format(hist_classes_all))

                            counter_file_label += 1

                            for f in range(X.shape[0]):
                                try:

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=float)


                                    obj = {"data": seq, "label": y[f], "labels": y_all[f],
                                           "identity": labels_persons[P], "label_file": counter_file_label}
                                    file_name = open(os.path.join(data_dir,
                                                                  'seq_{0:07}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                                    counter_seq += 1

                                    sys.stdout.write(
                                        '\r' +
                                        'Creating sequence file number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    file_name.close()

                                except:
                                    raise ('\nError adding the seq {} from {} \n'.format(f, X.shape[0]))

                            print("\nCorrect data extraction from {}".format(dataset_path_imu + file_name_data))

                            del data
                            del data_x
                            del X
                            del labels
                            del class_labels

                        else:
                            print("\n4 Not consisting annotation in  {}".format(file_name_data))
                            continue
                    except:
                        print("\n5 In generating data, No created file {}".format(dataset_path_imu + file_name_data))
                    print("-----------------\n{}\n{}\n-----------------".format(file_name_data, file_name_label))
                except KeyboardInterrupt:
                    print('\nYou cancelled the operation.')

    return


def generate_CSV(csv_dir, type_file, data_dir):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')

    return f


def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return f

def create_dataset(identity_bool = False):
    train_ids = ["S07", "S08", "S09", "S10"]
    #train_ids = ["S10"]
    train_final_ids = ["S07", "S08", "S09", "S10", "S11", "S12"]
    val_ids = ["S11", "S12"]
    test_ids = ["S13", "S14"]

    all_data = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    base_directory = '/data/fmoya/HAR/datasets/motionminers_flw/'

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    if identity_bool:
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train,
                      identity_bool=identity_bool, usage_modus='train')
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val,
                      identity_bool=identity_bool, usage_modus='val')
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test,
                      identity_bool=identity_bool, usage_modus='test')
    else:
        generate_data(train_ids, sliding_window_length=150, sliding_window_step=12, data_dir=data_dir_train)
        generate_data(val_ids, sliding_window_length=150, sliding_window_step=12, data_dir=data_dir_val)
        generate_data(test_ids, sliding_window_length=150, sliding_window_step=12, data_dir=data_dir_test)
        print("done")

    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return


if __name__ == '__main__':
    #IMUSequenceContainer = IMUSequenceContainer()

    # FLW
    #data = extract_data_flw(IMUSequenceContainer)
    #visualize(data)

    # Getting and visualising sequences out of the entire recordings
    #data = read_data_flw()

    # Segmenting the 30 recordings
    #create_sequences_flw()

    # Visualising for refinement of the sequences length
    visualise_sequences_flw()

    # Created annotated sequences
    #create_annotated_sequences()
    # correct_headers_data_sequences_flw()

    #Computing Statistics of data
    #max_values, min_values, mean_values, std_values = statistics_measurements()

    # Generate segmented sequences
    #create_dataset(identity_bool=False)



    print("Done")
