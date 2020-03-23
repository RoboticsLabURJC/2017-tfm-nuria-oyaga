from Utils import utils

import pandas as pd
import numpy as np
import cv2


def read_frame_data(f_path, sample_type, channels=False):
    if sample_type not in f_path:
        f_path = f_path + sample_type

    parameters_path = f_path.replace(sample_type, 'parameters.txt')

    if sample_type == "raw_samples":
        samples_paths = utils.get_dirs(f_path)
        dataX, dataY = get_samples(samples_paths, channels)
    else:
        samples_paths = utils.get_files(f_path)
        dataX, dataY = get_modeled_samples(samples_paths)

    parameters = pd.read_csv(parameters_path, sep=' ')

    return parameters, dataX, dataY


def get_images_per_sample(sample_dir):
    images = utils.get_images(sample_dir)

    return len(images) - 1


def read_batch_data(samples, idx, batch_size, channels):
    sub_samples = samples[idx * batch_size: (idx * batch_size) + batch_size]
    dataX, dataY = get_samples(sub_samples, channels)

    return dataX, dataY


def get_samples(samples_paths, channels):

    dataX = []
    dataY = []

    for p in samples_paths:
        dataX.append([cv2.imread(p + '/' + str(i) + '.png', 0) for i in range(20)])
        y_image = np.array(cv2.imread(p + '/20.png', 0))
        dataY.append(y_image.reshape(y_image.size))

    dataX = np.array(dataX, dtype="float") / 255
    dataY = np.array(dataY, dtype="float") / 255

    if channels:
        dataX = np.expand_dims(dataX, axis=-1)

    return (dataX, dataY)


def get_modeled_samples(samples_paths):
    dataX = []
    dataY = []

    for p in samples_paths:
        sample = pd.read_csv(p)
        dataX.append(np.fliplr(sample.iloc[:-1, :].values))
        dataY.append(sample.iloc[-1, :].values[::-1])

    return np.array(dataX), np.array(dataY)


def batch_generator(samples, batch_size, steps, channels):
     idx = 1
     while True:
        yield read_batch_data(samples, idx-1, batch_size, channels)
        if idx < steps:
            idx += 1
        else:
            idx = 1


def get_positions(predictions, real, dim, raw):
    predict_pos = []
    real_pos = []
    maximum = []
    for i, p in enumerate(predictions):
        if raw:
            p = p.reshape(dim)
            predict_pos.append(np.unravel_index(p.argmax(), p.shape))
            r = real[i].reshape(dim)
            real_pos.append(np.unravel_index(r.argmax(), r.shape))
        else:
            predict_pos.append(p)
            real_pos.append(real[i])

        maximum.append(np.linalg.norm(np.array((0, 0)) - np.array(dim)))

    return np.array(predict_pos), np.array(real_pos), np.array(maximum)


def draw_frame(fig, real_data, pred_data, dim):
    if len(real_data) > 2:
        bw_image_real = real_data.reshape(dim)
        bw_image_real = bw_image_real.astype(np.uint8) * 255

        pred_data[np.argmax(pred_data)] = 1
        pred_data = np.round(pred_data)
        bw_image_pred = pred_data.reshape(dim)
        bw_image_pred = bw_image_pred.astype(np.uint8) * 255

    else:
        bw_image_real = np.zeros(dim, np.uint8)
        bw_image_real[real_data[0], real_data[1]] = 255

        bw_image_pred = np.zeros(dim, np.uint8)
        bw_image_pred[int(pred_data[0]), int(pred_data[1])] = 255

    color_image_real = np.dstack([bw_image_real, bw_image_real, bw_image_real])
    color_image_pred = np.dstack([bw_image_pred, np.zeros(dim, np.uint8), np.zeros(dim, np.uint8)])
    color_image = color_image_pred + color_image_real
    fig.imshow(color_image)

    return fig

