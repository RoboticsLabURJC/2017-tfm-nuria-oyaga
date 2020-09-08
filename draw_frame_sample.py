from Utils import frame_utils
import numpy as np
import cv2
import random

if __name__ == '__main__':
    # Get sample
    type_data = "URM"
    data_dir = "/home/docker/data/Frames_dataset/" + type_data + "_point_255_fix_1000_80_120/" + type_data + \
               "_10_[None]_train/"
    parameters, known, predict = frame_utils.read_frame_data(data_dir, 'modeled_samples')
    to_draw = random.randint(0, known.shape[0] - 1)
    sample_to_draw_known = known[to_draw]
    sample_to_draw_predict = predict[to_draw]

    # Create images
    h = 80
    w = 120
    bw_image_known = np.zeros((h, w),  np.uint8)
    bw_image_predict = np.zeros((h, w), np.uint8)

    # Draw positions
    for pos in sample_to_draw_known:
        print(pos)
        bw_image_known[int(pos[0])][int(pos[1])] = 1
    bw_image_predict[int(sample_to_draw_predict[0])][int(sample_to_draw_predict[1])] = 1
    print(sample_to_draw_predict)

    # Make RGB image
    color_image_known = np.dstack([bw_image_known * 255, bw_image_known * 255, bw_image_known * 255])
    color_image_predict = np.dstack([bw_image_predict * 255, bw_image_predict * 249, bw_image_predict * 126])
    color_image = color_image_predict + color_image_known

    # Show image
    cv2.imwrite("samples/" + type_data + "_sample.png", color_image)
