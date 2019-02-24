from Utils import func_utils, vect_utils, frame_utils

from matplotlib import pyplot as plt
import numpy as np


def calculate_error(real, prediction, maximum):
    # Calculate error
    error = np.array([abs(real[i] - prediction[i]) for i in range(real.size - 1)])

    # Calculate relative error
    relative_error = np.zeros(error.size)
    for i in range(relative_error.size - 1):
        relative_error[i] = np.array((error[i] / abs(maximum[i])) * 100)

    return error, relative_error


def get_errors_statistics(error, relative_error):
    # Max error
    max_error_index = np.argmax(error)
    max_error = round(float(error[max_error_index]), 3)

    # Locate nan in relative error
    nan_index = np.argwhere(np.isnan(relative_error) == 1)

    # Replace Nan to 0 for calculate max
    for i in nan_index:
        relative_error[i] = 0

    max_rel_error_index = np.argmax(relative_error)
    max_rel_error = round(float(relative_error[max_rel_error_index]), 3)

    # Calculate error mean
    error_mean = np.sum(error) / error.size

    # Remove nan to calculate relative error mean
    relative_error = np.delete(relative_error, nan_index)

    relative_error_mean = np.sum(relative_error) / relative_error.size

    return [error_mean, [max_error_index, max_error]],\
           [relative_error_mean, [max_rel_error_index, max_rel_error]]


def error_histogram(error):
    hist = [sum((error >= 0) & (error < 10)),
            sum((error >= 10) & (error < 20)),
            sum((error >= 20) & (error < 30)),
            sum((error >= 30) & (error < 40)),
            sum((error >= 40) & (error < 50)),
            sum((error >= 50) & (error < 60)),
            sum((error >= 60) & (error < 70)),
            sum((error >= 70) & (error < 80)),
            sum((error >= 80) & (error < 90)),
            sum((error >= 90) & (error < 100)),
            sum((error >= 100))]

    labels = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)',
              '[70, 80)', '[80, 90)', '[90, 100)', '>100']

    plt.figure()
    x = range(len(hist))
    y = hist

    plt.bar(x, y, color='red')

    plt.xticks(x, labels)
    plt.ylabel("Number errors")
    plt.xlabel('Relative error intervals')
    plt.title('Relative error histogram')


def draw_max_error_samples(test_x, test_y, predict, gap, error_stats, rel_error_stats, data_type):

    if data_type == "Functions_dataset":
        f, (s1, s2) = plt.subplots(1, 2, sharey='all', sharex='all')
        func_utils.draw_function(s1, [test_x[error_stats[1][0]], test_y[error_stats[1][0]]],
                                 predict[error_stats[1][0]], gap)
        func_utils.draw_function(s2, [test_x[rel_error_stats[1][0]], test_y[rel_error_stats[1][0]]],
                                 predict[rel_error_stats[1][0]], gap)
        s2.set_xlim([0, 40])
    elif data_type == "Vectors_dataset":
        f, (s1, s2) = plt.subplots(2, 1, sharey='all', sharex='all')
        vect_utils.draw_vector(s1, [test_x[error_stats[1][0]], test_y[error_stats[1][0]]],
                               np.round(predict[error_stats[1][0]]), gap)
        vect_utils.draw_vector(s2, [test_x[rel_error_stats[1][0]], test_y[rel_error_stats[1][0]]],
                               np.round(predict[rel_error_stats[1][0]]), gap)

    else:  # data_type == "Frames_dataset"
        f, (s1, s2) = plt.subplots(1, 2, sharey='all', sharex='all')
        frame_dim = (test_x.shape[2], test_x.shape[3])
        frame_utils.draw_frame(s1, test_y[error_stats[1][0]], np.round(predict[error_stats[1][0]]), frame_dim)
        frame_utils.draw_frame(s2, test_y[rel_error_stats[1][0]], np.round(predict[rel_error_stats[1][0]]), frame_dim)

    s1.set_title(
        'Sample ' + str(error_stats[1][0]) + '\n' + 'Max. absolute error = ' + str(error_stats[1][1]) + '\n' +
        'Error mean = ' + "{0:.4f}".format(error_stats[0]))

    s2.set_title(
        'Sample ' + str(rel_error_stats[1][0]) + '\n' + 'Max. relative error = ' + str(rel_error_stats[1][1]) +
        '%' + '\n' + 'Relative error mean = ' + "{0:.4f}".format(rel_error_stats[0]) + '%')

    plt.show()
