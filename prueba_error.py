import numpy as np
import cv2


def get_samples(samples_paths):

    data = []

    for p in samples_paths:
        y_image = np.array(cv2.imread(p, 0))
        data.append(y_image.reshape(y_image.size))

    data = np.array(data, dtype="float") / 255

    return data


def get_positions(predictions, real, dim):
    predict_pos = []
    real_pos = []
    maximum = []
    for i, p in enumerate(predictions):
        p = p.reshape(dim)
        predict_pos.append(np.unravel_index(p.argmax(), p.shape))
        r = real[i].reshape(dim)
        real_pos.append(np.unravel_index(r.argmax(), r.shape))
        maximum.append(np.linalg.norm(np.array((0, 0)) - np.array(dim)))

    return np.array(predict_pos), np.array(real_pos), np.array(maximum)


def calculate_error(real, prediction, maximum):
    # Calculate error
    if len(real.shape) > 1:
        error = np.array([np.linalg.norm(np.array(real[i]) - np.array(prediction[i]))
                          for i in range(real.shape[0] - 1)])
    else:
        error = np.array([abs(real[i] - prediction[i]) for i in range(real.size - 1)])

    # Calculate relative error
    relative_error = np.zeros(error.size)
    for i in range(relative_error.size - 1):
        relative_error[i] = np.array((error[i] / abs(maximum[i])) * 100)

    return error, relative_error


if __name__ == '__main__':
    target = ["/home/docker/2017-tfm-nuria-oyaga/prueba_error_images/I_target.png" for i in range(13)]
    images = ["/home/docker/2017-tfm-nuria-oyaga/prueba_error_images/I_" + str(i+1) + ".png" for i in range(13)]
    prediction_data = get_samples(images)
    real_data = get_samples(target)
    predict_positions, real_positions, maximum_val = get_positions(prediction_data, real_data, (80, 120))
    abs_error, rel_error = calculate_error(real_positions, predict_positions, maximum_val)

    print()
