"""
data_utils.py: A script with some utils to process data

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


import numpy as np


def read_data(file):

    f = open(file)
    data = f.readlines()
    data = data[1:]

    parameters = [data[i].split('][')[0].split(' ')[1:-1] for i in range(len(data))]

    samples = [np.array(element.split('][')[1].split(' ')[1:-2], dtype=np.float32) for element in data]

    print(samples[0].size)
    return parameters, samples


def reshape_data(data):

    for sample in enumerate(data):
        points = sample[1]

        if sample[0] == 0:
            dataX = points[0:-1]
            dataY = points[points.size - 1]

        else:
            dataX = np.vstack([dataX, points[0:-1]])
            dataY = np.append(dataY, points[points.size - 1])

    print(dataX.shape)

    return dataX,dataY


def draw_data(fig, data, gap):
    last_know = len(data[0])
    x = list(range(last_know))

    fig.scatter(x, data[0], 5, color='green')
    fig.scatter(last_know - 1 + gap, data[1], 5, color='green')