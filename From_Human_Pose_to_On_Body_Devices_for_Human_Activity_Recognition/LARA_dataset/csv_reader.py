'''
Created on May 18, 2019

@author: fmoya
'''

import numpy as np
import csv 
import sys


#def reader_data(path: str) -> np.array:
def reader_data(path):
    '''
    gets data from csv file
    data contains 134 columns
    the first column corresponds to sample
    the second column corresponds to class label
    the rest 132 columns corresponds to all of the joints (x,y,z) measurements

    returns a numpy array

    @param path: path to file
    '''
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data



def reader_data_2(path):
    '''
    gets data from csv file
    data contains 134 columns
    the first column corresponds to sample
    the second column corresponds to class label
    the rest 132 columns corresponds to all of the joints (x,y,z) measurements
    
    returns a numpy array
    
    @param path: path to file
    '''

    counter = 0
    data = np.empty((0,134))
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                if spamreader.line_num == 1:
                    print('\n')
                    print(', '.join(row))
                else:
                    frame = list(map(float, row))
                    frame = np.array(frame)
                    frame = frame[:134]
                    frame = np.reshape(frame, newshape = (1, 134))
                    data = np.append(data, frame, axis = 0)
                    sys.stdout.write('\r' + 'In {} Number of seq {}'.format(path, len(data)) )
                    sys.stdout.flush()

                #if counter == 5000:
                #    break

                counter += 1
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    return data


def reader_labels(path):
    '''
    gets labels and attributes from csv file
    data contains 20 columns
    the first column corresponds to class label
    the rest 19 columns corresponds to all of the attributes

    returns a numpy array

    @param path: path to file
    '''

    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data


def reader_labels_2(path):

    '''
    gets labels and attributes from csv file
    data contains 20 columns
    the first column corresponds to class label
    the rest 19 columns corresponds to all of the attributes
    
    returns a numpy array
    
    @param path: path to file
    '''
    
    counter = 0

    data = np.empty((0,20))
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                if spamreader.line_num == 1:
                    print('\n')
                    print(', '.join(row))
                else:
                    frame = list(map(float, row))
                    frame = np.array(frame)
                    frame = frame[:20]
                    frame = np.reshape(frame, newshape = (1, 20))
                    data = np.append(data, frame, axis = 0)
                    sys.stdout.write('\r' + 'In {} Number of seq {}'.format(path, len(data)) )
                    sys.stdout.flush()
                #if counter == 5000:
                #    break

                counter += 1
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')
     
    return data


if __name__ == '__main__':
    

    #pathFile = '/vol/corpora/har/DFG_Project/2019/MoCap/recordings_2019_06/14_Annotated_Dataset/P01/' +\
    #           'S01_P01_R01_A17_N01_labels.csv'
    
    #labels = reader_labels(pathFile)
    

    #pathFile = '/vol/corpora/har/DFG_Project/2019/MoCap/recordings_2019_06/14_Annotated_Dataset/P01/' +\
    #           'S01_P01_R01_A17_N01_norm_data.csv'
    
    #data = reader_data(pathFile)

    print("Done")
    
    