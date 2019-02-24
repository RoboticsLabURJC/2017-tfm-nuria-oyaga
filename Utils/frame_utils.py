from Utils import utils

import pandas as pd
import numpy as np
import cv2


def read_frame_data(f_path):
    parameters_path = f_path.replace('samples', 'parameters.txt')
    samples_paths = utils.get_dirs(f_path)

    samples = []
    for p in samples_paths:
        samples.append([cv2.imread(p + '/' + str(i) + '.png', 0) for i in range(21)])

    parameters = pd.read_csv(parameters_path, sep=' ')

    return parameters, samples


def reshape_frame_data(data, channels=False):
    dataX = []
    dataY = []
    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(i)

        dataX.append(sample[:][0:-1])
        y_image = np.array(sample[:][-1])
        dataY.append(y_image.reshape(y_image.size))

    dataX = np.array(dataX, dtype="float") / 255
    dataY = np.array(dataY, dtype="float") / 255

    if channels:
        dataX = np.expand_dims(dataX, axis=-1)

    return dataX, dataY


def draw_frame(fig, real_data, pred_data, dim):
    bw_image_real = real_data.reshape(dim)
    bw_image_real = bw_image_real.astype(np.uint8) * 255
    color_image_real = np.dstack([bw_image_real, bw_image_real, bw_image_real])

    pred_data = np.round(pred_data)
    bw_image_pred = pred_data.reshape(dim)
    bw_image_pred = bw_image_pred.astype(np.uint8) * 255
    color_image_pred = np.dstack([bw_image_pred, np.zeros(dim, np.uint8), np.zeros(dim, np.uint8)])

    color_image = color_image_pred + color_image_real
    fig.imshow(color_image)

    return fig
