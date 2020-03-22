from Utils import utils

import numpy as np
import pandas as pd
import cv2


def read_vector_data(data_path):
    parameters_path = data_path.replace('samples', 'parameters.txt')
    images_paths = utils.get_images(data_path)

    parameters = pd.read_csv(parameters_path, sep=' ')
    images = [cv2.imread(img_path, 0) for img_path in images_paths]

    return parameters, images


def reshape_vector_data(data):
    dataX = []
    dataY = []
    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(i)

        dataX.append(sample[:][0:-1])
        dataY.append(sample[:][-1])

    dataX = np.array(dataX, dtype="float") / 255
    dataY = np.array(dataY, dtype="float") / 255

    return dataX, dataY


def get_positions(predictions, real):
    predict_pos = []
    real_pos = []
    maximum = []
    for i, p in enumerate(predictions):
        predict_pos.append(np.argmax(p))
        real_pos.append(np.argmax(real[i]))
        maximum.append(len(p))

    return np.array(predict_pos), np.array(real_pos), np.array(maximum)


def draw_vector(fig, real_data, pred_data, gap):
    gap_image = np.zeros((gap - 1, real_data[0].shape[1]))
    bw_image = np.concatenate((real_data[0], gap_image, real_data[1].reshape(1, real_data[0].shape[1])))
    bw_image = bw_image.astype(np.uint8) * 255

    pred_pos = np.where(pred_data == 1)[0]
    bw_image_with_pred = bw_image.copy()
    bw_image_with_pred[-1][pred_pos] = 255
    color_image = np.dstack([bw_image_with_pred, bw_image, bw_image])

    fig.imshow(color_image)

    return fig
