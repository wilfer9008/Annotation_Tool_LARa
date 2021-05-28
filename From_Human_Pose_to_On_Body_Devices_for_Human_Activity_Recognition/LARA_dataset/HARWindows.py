'''
Created on May 18, 2019

@author: fmoya
'''

import os

from torch.utils.data import Dataset


import pandas as pd
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class HARWindows(Dataset):
    '''
    classdocs
    '''


    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with list of annotated sequences.
            root_dir (string): Directory with all the sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.harwindows = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.harwindows)

    def __getitem__(self, idx):
        '''
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.harwindows.iloc[idx, 0])

        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()

        X = data['data']
        y = data['label']
        Y = data['labels']

        if 'identity' in data.keys():
            i = data['identity']
        
            window_data = {"data": X, "label": y, "labels": Y, "identity": i}
        else:
            window_data = {"data": X, "label": y, "labels": Y}

        return window_data
