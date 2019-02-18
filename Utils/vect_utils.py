import numpy as np
from matplotlib import pyplot as plt
import cv2


def get_positions(predictions, real):
    predict_pos = []
    real_pos = []
    maximum = []
    for i, p in enumerate(predictions):
        predict_pos.append(np.where(np.round(p[0]) == 1)[0][0])
        real_pos.append(np.where(real[i][0] == 1)[0][0])
        maximum.append(len(p[0]))

    return np.array(predict_pos), np.array(real_pos), np.array(maximum)


def draw_vector(fig, real_data, pred_data, gap):
    gap_image = np.zeros((gap-1, real_data[0].shape[1]))
    bw_image = np.concatenate((real_data[0], gap_image, real_data[1]))
    bw_image = bw_image.astype(np.uint8) * 255
    color_image = np.dstack([bw_image, bw_image, bw_image])

    color_image[bw_image.shape[0] - 1][pred_data][-1] = 255

    fig.imshow(color_image)

    return fig
