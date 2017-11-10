"""
data_utils.py: A script with some utils to process data

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


import numpy as np


def read_data(file):

    type = file.split('/')[1].split('_')[0]

    if type == 'linear':
        parameters = 2
    elif type == 'quadratic':
        parameters = 3

    f = open(file)
    data = f.readlines()
    data = data[1:]
    i = 0

    for element in data:
        element = element.split('\n')[0].split(']')[0].split('[')[1].split(',')
        data[i] = np.array(element[parameters:], 'int')
        i = i+1

    return data


def reshape_data(data):

    for sample in enumerate(data):
        points = sample[1]

        if sample[0] == 0:
            dataX = points[0:-1]
            dataY = points[points.size - 1]

        else:
            dataX = np.vstack([dataX, points[0:-1]])
            dataY = np.append(dataY, points[points.size - 1])

    return dataX,dataY
