import numpy as np
import os
import sys
import time
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import fnmatch
from symbol import except_clause
import traceback
from _functools import reduce

body_segments= {
            -1:'none',
            0:'head',
            1:'head end',
            2:'L collar',       12:'R collar',
            6:'L humerus',      16:'R humerus',
            3:'L elbow',        13:'R elbow',
            9:'L wrist',        19:'R wrist',
            10:'L wrist end',   20:'R wrist end',
            11:'lower back',    
            21:'root',
            4:'L femur',        14:'R femur',
            7:'L tibia',        17:'R tibia',
            5:'L foot',         15:'R foot',
            8:'L toe',          18:'R toe'}
    

def load_data(path:str) -> np.array:
        """loads and normalizes mocap data stored in path
        
        Arguments:
        ---------
        path : str
            path to the unlabeled motion capture data
        ---------
        
        Returns: 
        ---------
        array : numpy.array
            2D array with normalized motioncapture data.
            1st dimension is the time
            2nd dimension is the location and rotation data of each bodysegment
            shape should be (t,132) with t as number of timesteps in the data
        ---------
        """
        array = np.loadtxt(path, delimiter=',', skiprows=5)
        array = array[:,2:]
        array = normalize_data(array)
        return array
    
def normalize_data(array:np.array) -> np.array:
    """normalizes the mocap data array
        
    The data gets normalized by subtraction of the lower backs data from every bodysegment.
    That way the lowerback is in the origin.
        
    Arguments:
    ---------
    array : numpy.array
        2D array with normalized motioncapture data.
        1st dimension is the time
        2nd dimension is the location and rotation data of each bodysegment
        shape should be (t,132) with t as number of timesteps in the data
    ---------
        
    Returns: 
    ---------
    array : numpy.array
        2D array with normalized motioncapture data.
        1st dimension is the time
        2nd dimension is the location and rotation data of each bodysegment
        shape should be (t,132) with t as number of timesteps in the data
    ---------
    """
        
    normalizing_vector = array[:,66:72]#66:72 are the columns for lowerback
    for _ in range(21):
        normalizing_vector = np.hstack((normalizing_vector,array[:,66:72]))    
    array = np.subtract(array,normalizing_vector) 
    return array

def genWindows(data):
    """"""
    windows = []
    classes = data[:,0].astype(int)
    attributes = data[:,1:].astype(int)
        
    samples = data.shape[0]
        
    current_class = classes[0]
    current_attributes = attributes[0].tolist()
    start = 0
    for i in range(1,samples):
        
        a_and_b = [a == b for a,b in zip(current_attributes,attributes[i].tolist())]
        same_attributes = reduce(lambda a,b: a and b, a_and_b) 
        
        if not ((classes[i] == current_class) and same_attributes):
            end = i
            windows.append((start,end,current_class,current_attributes))
            start = end
            current_class = classes[i]
            current_attributes = attributes[i].tolist()
    end = samples
    windows.append((start,end,current_class,current_attributes))
    return windows

def saveWindows(path,windows):
    """Rewrites all windows into the file
        
    Should be called when at least one already existing window gets changed.
    """
    #rewrites the file file with the new change
    file = open(path, 'wt')
    for window in windows:
        file.write(str(window)+'\n')
        file.flush()
    file.close()

app = QtWidgets.QApplication(sys.argv)

data_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory with the raw mocap data files')
label_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory where the labeled data is stored')
new_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory where the updated data should be stored')

QTimer.singleShot(100, app.quit)
app.exec_()

if data_directory is not '' and label_directory is not '' and new_directory is not '':
    (_, _, data_filenames) = next(os.walk(data_directory))
    (_, _, label_filenames) = next(os.walk(label_directory))
    
    #S02_P10_R03.csv
    #S02_P10_R03_A18_N1_norm_data.csv 
    
    
    label_filenames = fnmatch.filter(label_filenames,"*labels*")#sort out norm_data files.
    
    
    for label_f in label_filenames:
        print("\n\n\n\nprocessing file: "+label_f)
        
        #S03_P10_R27_A20_N01_attrib.csv
        name_fragments = label_f.split('_')
        
        raw_data_name = name_fragments[0]
        for fragment in name_fragments[1:3]: 
            raw_data_name += "_"+fragment
        raw_data_name += ".csv" #S03_P10_R27.csv
        
        if raw_data_name in data_filenames:
            #open label_f and raw_data_name
            #normalize raw_data_name
            #overwrite label_f's data
            #save label_f
            
            #opening label_f
            try:
                
                #----------------------------------------making names
                files_name_core = name_fragments[0]
                for fragment in name_fragments[1:-1]: 
                    files_name_core += "_"+fragment
                #S03_P10_R27_A20_N01
                
                label_name = files_name_core+"_labels.csv"
                old_window_name = files_name_core+"_windows.txt"
                old_data_name = files_name_core+"_norm_data.csv"
                
                #------------------------------------make labels
                
                labels = np.loadtxt(label_directory+os.sep+label_f, delimiter=',', skiprows=1)
                #labels = np.vstack((labels[0],labels[0],labels))
                labels_header_list = np.loadtxt(label_directory+os.sep+label_f, delimiter=',', max_rows=1,dtype='str').tolist()
                labels_header = reduce(lambda s1,s2: s1+","+s2, labels_header_list)
                #print(labels_header)
                
                
                
                #-------------------------------------make normdata
                try:
                    header = "sample,label"
                    for i in range(22):
                        bodysegment = body_segments[i]
                        for coordinate in ['RX','RY','RZ','TX','TY','TZ']:
                            header += "," + bodysegment+'_'+coordinate
                
                    normalized_data = load_data(data_directory+os.sep+raw_data_name)
                    #normalized_data = np.vstack((normalized_data[0],normalized_data[0],normalized_data))


                    print("Old Labels size {}".format(labels.shape))
                    print("OLd Data size {}".format(normalized_data.shape))
                    
                    if labels.shape[0] == 23998:
                        labels = np.vstack((labels[0],labels[0],labels))
                        print("Labels size {}".format(labels.shape))
                        print("Data size {}".format(normalized_data.shape))

                    if labels.shape[0] == 24002:
                        labels = labels[:24000]
                        print("Labels size {}".format(labels.shape))
                        print("Data size {}".format(normalized_data.shape))

                    if normalized_data.shape[0] == 23998:
                        normalized_data = np.vstack((normalized_data[0],normalized_data[0],normalized_data))
                        print("Labels size {}".format(labels.shape))
                        print("Data size {}".format(normalized_data.shape))

                    print("New Labels size {}".format(labels.shape))
                    print("New Data size {}".format(normalized_data.shape))
                
                    classes = np.expand_dims(labels[:,0],1)                
                    samples = np.expand_dims(np.arange(1,classes.shape[0]+1), 0).T
                    
                    print(samples.shape, classes.shape, normalized_data.shape)
                    norm_data_contents = np.hstack((samples,classes,normalized_data))

                    print("Final Labels size {}".format(labels.shape))
                    print("FinalData size {}".format(norm_data_contents.shape))

                    
                    norm_data_name = name_fragments[0]
                    for fragment in name_fragments[1:-1]: 
                        norm_data_name += "_"+fragment
                    norm_data_name += "_norm_data.csv" #S03_P10_R27_A20_N01_norm_data.csv
                
                    np.savetxt(new_directory+os.sep+norm_data_name, norm_data_contents, delimiter=',', header=header, comments='')
                    np.savetxt(new_directory+os.sep+label_name, labels, delimiter=',', header=labels_header, comments='')

                except Exception as e: 
                    print("Error couldnt save normdata")
                    traceback.print_exc()
                    raise


                #-------------------------------------make windows
                try:
                    windows_file_name = name_fragments[0]
                    for fragment in name_fragments[1:-1]: 
                        windows_file_name += "_"+fragment
                    windows_file_name += "_windows.txt" #S03_P10_R27_A20_N01_windos.txt
                
                    windows = genWindows(labels)
                
                    saveWindows(new_directory+os.sep+windows_file_name, windows)
                except Exception: 
                    print("Error couldnt save windows")
                    traceback.print_exc()
                    raise
                
                #try: 
                #    os.remove(label_directory+os.sep+label_f)
                #    os.remove(label_directory+os.sep+old_window_name)
                #    os.remove(label_directory+os.sep+old_data_name)
                #except:
                #    print("Error: Deleting old Files belonging to: " + label_f)
                
                
                
            except Exception:
                print("Error: Couldn't process: " + label_f)
                traceback.print_exc()
            
        else:
            print("Error: Couldn't find original data file with name: '" + raw_data_name +
                   "' to correspond to labeled file: '" + label_f + "' moving onto next file")
    print("finished all files")
        
else:
    print("at least one chosen directory was ''")
    









