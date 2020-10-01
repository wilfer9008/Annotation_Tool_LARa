import numpy as np
import os
import sys
import time
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import fnmatch

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

'''
def saveResults(self,directory:str,annotatorID:int,tries:int):
        """Saves the finished labels, normalized data, and windows in 3 files in the provided directory
        
        Arguments:
        ----------
        directory : str
            path to the directory where the results should be saved.
        annotatorID : int
            ID of the person annonating the dataset.
        tries : int
            number indicating how often the currently worked on file 
            was beeing annotated by the same annotator.
        ----------
        
        """
        annotator_suffix = '_A'+str(annotatorID)+'_N'+str(tries)
        header = "sample,classlabel"
        for i in range(22):
            bodysegment = self.body_segments[i]
            for coordinate in ['RX','RY','RZ','TX','TY','TZ']:
                header += "," + bodysegment+'_'+coordinate
        
        data = np.zeros((number_samples, 2+mocap_data.shape[1])) #samples+class+ datacolumns
        for start,end,class_index,attributes in windows:
            for i in range(start,end):
                data[i,1] = class_index
        data[:,0] = range(self.number_samples)
        data[:,2:] = self.mocap_data[:,:]   
                
                
        file_name = file_name.split('.')[0] +annotator_suffix+ "_norm_data.csv"
        np.savetxt(directory+os.sep+file_name, data, delimiter=',', header=header, comments='')
'''

app = QtWidgets.QApplication(sys.argv)

data_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory with the raw mocap data files')
label_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory where the wrong normalized data is be stored')
new_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select the directory where the new normalized data should be stored')

QTimer.singleShot(100, app.quit)
app.exec_()

if data_directory is not '' and label_directory is not '' and new_directory is not '':
    (_, _, data_filenames) = next(os.walk(data_directory))
    (_, _, label_filenames) = next(os.walk(label_directory))
    
    #S02_P10_R03.csv
    #S02_P10_R03_A18_N1_norm_data.csv 
    
    
    label_filenames = fnmatch.filter(label_filenames,"*norm*")#sort out norm_data files.
    
    print("found data files")
    print(data_filenames)
    print("found labeled files")
    print(label_filenames)
    
    for label_f in label_filenames:
        print("processing file: "+label_f)        
        #get their classlabels and annotator_suffix
        #overwrite data in the file if data_filenames contains corresponding .csv file.
        
        #name_fragments = label_f.split('_')[:3]
        #S03_P10_R27_A20_N01_norm_data.csv
        name_fragments = label_f.split('_')
        
        raw_data_name = name_fragments[0]
        for fragment in name_fragments[1:3]: 
            raw_data_name += "_"+fragment
        raw_data_name += ".csv" #S03_P10_R27.csv
        
        #print("opening file: "+raw_data_name)
        if raw_data_name in data_filenames:
            #open label_f and raw_data_name
            #normalize raw_data_name
            #overwrite label_f's data
            #save label_f
            
            #opening label_f
            try:
                old_csv = np.loadtxt(label_directory+os.sep+label_f, delimiter=',', skiprows=1,usecols=(0,1))
                if not old_csv.shape[0] == 24000:
                    old_csv = np.loadtxt(label_directory+os.sep+label_f, delimiter=',', skiprows=0, usecols=(0,1))
                    
                #first_2_columns = old_csv[:,:2]
                first_2_columns = old_csv
                header = "sample,label"
                for i in range(22):
                    bodysegment = body_segments[i]
                    for coordinate in ['RX','RY','RZ','TX','TY','TZ']:
                        header += "," + bodysegment+'_'+coordinate
            
                #normalizing raw_data_name
                normalized_data = load_data(data_directory+os.sep+raw_data_name)
            
                #combining norm_data and first 2 columns
                #print(first_2_columns.shape)
                #print(normalized_data.shape)
                csv = np.hstack((first_2_columns,normalized_data))
                
                #saving csv to label_f
                np.savetxt(new_directory+os.sep+label_f, csv, delimiter=',', header=header, comments='')
                #print("Normalizing done. Moving onto next file")
                
                #moving label.csv and windows.txt
                
                file_name_core = name_fragments[0]
                for fragment in name_fragments[1:-2]: 
                    file_name_core += "_"+fragment
                file_name_label = file_name_core + "_labels.csv" #S03_P10_R27_A20_N01_labels.csv
                file_name_windows = file_name_core + "_windows.txt"
                print(file_name_label)
                print(file_name_windows)
                try:
                    os.replace(label_directory+os.sep+file_name_label, new_directory+os.sep+file_name_label)
                    os.replace(label_directory+os.sep+file_name_windows, new_directory+os.sep+file_name_windows)
                    os.remove(label_directory+os.sep+label_f)
                except:
                    print("Error Moving windows and labelfile belonging to: " + label_f)
                
                
                
            except:
                print("Error: Couldn't open: " + label_f)
                
            
        else:
            print("Error: Couldn't find original file with name: '" + raw_data_name +
                   "' to correspond to labeled file: '" + label_f + "' moving onto next file")
    print("finished all files")
        
else:
    print("at least one chosen directory was ''")
    









