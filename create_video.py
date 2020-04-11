from Utils import utils

import random
import cv2


if __name__ == '__main__':

    data_type = "parabolic"
    samples_folder = "/home/docker/Generated_Data/Frames_dataset/" + data_type + "_point_255_fix_10000_80_120/" +\
                     data_type + "_10_[None]_train/raw_samples/"
    # front_name = video_name.replace('.avi', '.png')
    video_dir = '/home/docker/Videos/'
    utils.check_dirs(video_dir)
    samples_paths = utils.get_dirs(samples_folder)
    to_draw = random.randint(0, len(samples_paths) - 1)
    sample_to_draw = samples_paths[to_draw]
    video_name = video_dir + data_type + '_' + sample_to_draw.split("/")[-1] + '.avi'
    print(sample_to_draw)
    # sample = [sample_to_draw + '/' + str(i) + '.png' for i in range(21)]
    sample = [sample_to_draw + '/' + str(i) + '.png' for i in range(21)]

    front = cv2.imread(sample[0])
    # cv2.imwrite(front_name, front)
    height, width, layers = front.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

    for frame_path in sample:
        print(frame_path)

        image = cv2.imread(frame_path)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
