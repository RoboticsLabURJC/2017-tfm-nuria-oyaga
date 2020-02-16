from Utils import utils

import random
import cv2
import os


if __name__ == '__main__':

    samples_folder = "/home/nuria/Desktop/Data/Frames_dataset/linear_point_255_var_10000/linear_10_[None]_train/raw_samples/"
    video_name = '/home/nuria/Desktop/Videos/frames_linear_point_y0_sample.avi'
    front_name = video_name.replace('.avi', '.png')

    # samples_paths = utils.get_dirs(samples_folder)
    # to_draw = random.randint(0, len(samples_paths) - 1)
    # sample_to_draw = samples_paths[to_draw]
    sample_to_draw = "/home/nuria/Desktop/Data/Frames_dataset/linear_point_255_var_10000/linear_10_[None]_train/raw_samples/sample57"
    print(sample_to_draw)
    sample = [sample_to_draw + '/' + str(i) + '.png' for i in range(21)]

    front = cv2.imread(sample[0])
    cv2.imwrite(front_name, front)
    height, width, layers = front.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for frame_path in sample:
        print(frame_path)

        image = cv2.imread(frame_path)
        cv2.imshow('frame', image)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
