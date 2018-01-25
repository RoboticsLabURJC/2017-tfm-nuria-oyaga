"""
data_utils.py: A script with some utils to process data

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


import numpy as np
import os
import shutil


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def read_data(file):
    """
    Function to read the data according how thy were created (sequence_generator.py)

    :param file: File where the data are saved
    :return: The data
    """

    f = open(file)
    data = f.readlines()
    data = data[1:]

    parameters = [data[i].split('][')[0].split(' ')[1:-1] for i in range(len(data))]

    samples = [np.array(data[i].split('][')[1].split(' ')[1:-2], dtype=np.float32) for i in range(len(data))]

    return parameters, samples


def reshape_data(data):
    """
    Function to get the data in a correct shape to use with a Keras network

    :param data: Data to reshape(sample)
    :return: Data provided, data to predict
    """

    for sample in enumerate(data):
        points = sample[1]  # Number the points in a sample (known + to predict)

        if sample[0] == 0:
            dataX = points[0:-1]  # Get known elements
            dataY = points[points.size - 1]  # Get to predict element

        else:
            dataX = np.vstack([dataX, points[0:-1]])  # Stack all the known elements (each sample)
            dataY = np.append(dataY, points[points.size - 1])  # Add all to predict element (each shample)

    return dataX,dataY


def draw_data(fig, data, gap):
    """
    Function to draw a sample

    :param fig: Figure where draw
    :param data: Data to draw
    :param gap: Gap between know data and data to predict
    """

    # For know elements
    last_know = len(data[0])
    x = list(range(last_know))

    fig.scatter(x, data[0], 5, color='green')
    # For to predict element
    fig.scatter(last_know - 1 + gap, data[1], 5, color='green')
