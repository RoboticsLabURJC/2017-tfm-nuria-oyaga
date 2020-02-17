import cv2
import numpy as np


if __name__ == '__main__':

    raw_sample_to_draw = "/home/nuria/Desktop/Data/Frames_dataset/linear_point_255_var_10000/linear_10_[None]_train/raw_samples/sample57"
    raw_sample = [raw_sample_to_draw + '/' + str(i) + '.png' for i in range(21)]

    modeled_sample_to_draw = "/home/nuria/Desktop/Data/sample57/sample57_"
    modeled_sample = [modeled_sample_to_draw + str(i+1) + '.PNG' for i in range(21)]

    raw_front = cv2.imread(raw_sample[0])
    r_height, r_width, r_layers = raw_front.shape
    r_h = int(r_height * 4)
    r_w = int(r_width * 4)

    modeled_front = cv2.imread(modeled_sample[0])
    m_height, m_width, m_layers = modeled_front.shape

    diff = abs(r_h - m_height)

    for i, frame_path in enumerate(raw_sample):
        print(frame_path)

        raw_image = cv2.imread(frame_path)
        raw_image = cv2.resize(raw_image, (r_w, r_h), interpolation=cv2.INTER_AREA)
        raw_image = cv2.copyMakeBorder(raw_image, int(diff/2), int(diff/2) + 1, 10, 10, cv2.BORDER_CONSTANT, None, [255, 255, 255])
        modeled_image = cv2.imread(modeled_sample[i])
        modeled_image = cv2.resize(modeled_image, (m_width, m_height), interpolation=cv2.INTER_AREA)

        image = np.concatenate((raw_image, modeled_image), axis=1)
        cv2.imwrite("/home/nuria/Desktop/Videos/images/image" + str(i) + ".png", image)
