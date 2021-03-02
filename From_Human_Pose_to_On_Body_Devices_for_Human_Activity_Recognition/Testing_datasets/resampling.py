'''
Created on Dec 05, 2019

@author: fmoya
'''

import sys
import numpy as np
import math
import scipy.interpolate
import time
import torch.utils.data as data
import logging
import scipy.interpolate



class Resampling():


    def __init__(self):


        return


    def interpolate(self, data_t, data_x, data_y, nsr):
        """Function to interpolate the sequence according to the given sampling rate

        :param data_t: numpy integer matrix
            Vector containing  the sample time in ms
        :param data_x: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :param data_y: numpy integer matrix
            Vector containing the labels
        :param nsr: int
            New sampling rate
        :return data_t_new, data_x_news, data_y_new:

        """

        ms = 1000 / nsr
        int_time = np.arange(math.ceil(data_t[1]), math.floor(data_t[-2]), ms)

        data_x_new = [np.empty((0)) for ss in range(data_x.shape[1])]
        data_y_new = np.empty((0))
        data_t_new = np.empty((0))
        for tm in range(0, int_time.shape[0] - 3, 3):
            rb = [np.argmin(np.abs(data_t - int_time[tm + tmx])) for tmx in range(3)]

            y_new = data_y[rb]
            data_y_new = np.append(data_y_new, y_new, axis=0)

            rt = rb[-1] + 2
            rb = rb[0] - 1
            if rt > data_t.shape[0]:
                rt = data_t.shape[0]

            t_new = int_time[tm: tm + 3]
            if rt - rb < 3:
                break

            for ms in range(data_x.shape[1]):
                tck = scipy.interpolate.splrep(data_t[rb: rt], data_x[rb: rt, ms], s=0, k=2)
                x_new = scipy.interpolate.splev(t_new, tck, der=0)
                data_x_new[ms] = np.append(data_x_new[ms], x_new, axis=0)

            data_t_new = np.append(data_t_new, t_new, axis=0)

        data_x_news = np.zeros((data_x_new[0].shape[0], data_x.shape[1]))
        for ss in range(data_x.shape[1]):
            data_x_news[:, ss] = data_x_new[ss]

        return data_t_new, data_x_news, data_y_new

