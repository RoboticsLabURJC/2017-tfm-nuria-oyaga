from Utils import func_utils, vect_utils, frame_utils, utils

from matplotlib import pyplot as plt
import numpy as np


def calculate_error(real, prediction, maximum):
    # Calculate error
    x_error = []
    y_error = []
    if len(real.shape) > 1:
        error = np.array([np.linalg.norm(np.array(real[i]) - np.array(prediction[i]))
                          for i in range(real.shape[0])])
        y_error = np.array([abs(real[i][0] - prediction[i][0]) for i in range(real.shape[0])])
        x_error = np.array([abs(real[i][1] - prediction[i][1]) for i in range(real.shape[0])])

    else:
        error = np.array([abs(real[i] - prediction[i]) for i in range(real.size)])
        x_error = None
        y_error = None
        x_rel_error = None
        y_rel_error = None

    # Calculate relative error
    relative_error = np.array([((error[i] / abs(maximum[i])) * 100) for i in range(error.size)])

    return error, x_error, y_error, relative_error


def get_errors_statistics(error, x_error, y_error, relative_error, dim):
    # Max error
    max_error_index, max_error = utils.calculate_max(error)

    # Relative error
    # Locate nan in relative error
    # nan_index = np.argwhere(np.isnan(relative_error) == 1)

    # Replace Nan to 0 for calculate max
    """for i in nan_index:
        relative_error[i] = 0"""
    max_rel_error_index, max_rel_error = utils.calculate_max(relative_error)

    # Calculate error mean
    error_mean = utils.calculate_mean(error)
    # Remove nan to calculate relative error mean
    # relative_error = np.delete(relative_error, nan_index)
    relative_error_mean = utils.calculate_mean(relative_error)

    if x_error is not None:
        _, x_max_error = utils.calculate_max(x_error)
        x_mean_error = utils.calculate_mean(x_error)
        _, y_max_error = utils.calculate_max(y_error)
        y_mean_error = utils.calculate_mean(y_error)
        x_max_rel_error = round((x_max_error/dim[1] * 100), 3)
        x_mean_rel_error = round((x_mean_error/dim[1] * 100), 3)
        y_max_rel_error = round((y_max_error/dim[0] * 100), 3)
        y_mean_rel_error = round((y_mean_error/dim[0] * 100), 3)
    else:
        x_max_error = None
        y_max_error = None
        x_mean_error = None
        y_mean_error = None
        x_max_rel_error = None
        x_mean_rel_error = None
        y_max_rel_error = None
        y_mean_rel_error = None

    return [error_mean, [max_error_index, max_error]], \
           [x_mean_error, x_max_error], \
           [y_mean_error, y_max_error], \
           [relative_error_mean, [max_rel_error_index, max_rel_error]], \
           [x_mean_rel_error, x_max_rel_error], \
           [y_mean_rel_error, y_max_rel_error]


def error_histogram(error):
    hist = [sum((error >= 0) & (error < 1)),
            sum((error >= 1) & (error < 2)),
            sum((error >= 2) & (error < 5)),
            sum((error >= 5) & (error < 10)),
            sum((error >= 10) & (error < 20)),
            sum((error >= 20) & (error < 40)),
            sum((error >= 40) & (error < 80)),
            sum((error >= 80))]

    labels = ['[0, 1)', '[1, 2)', '[2, 5)', '[5, 10)', '[10, 20)', '[20, 40)', '[40, 80)', '>80']

    plt.figure()
    x = range(len(hist))
    y = hist

    plt.bar(x, y, color='red')

    plt.xticks(x, labels)
    plt.ylabel("Number errors")
    plt.xlabel('Error intervals')
    plt.title('Error histogram')


def relative_error_histogram(error):
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


def draw_max_error_samples(test_x, test_y, predict, gap, error_stats, rel_error_stats, data_type, dim):

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

    else:
        target = test_y[error_stats[1][0]]
        prediction = predict[error_stats[1][0]]
        rel_target = test_y[rel_error_stats[1][0]]
        rel_prediction = predict[rel_error_stats[1][0]]
        if "modeled" in data_type:
            target = utils.scale_position(target, dim[1], dim[0])
            prediction = utils.scale_position(prediction, dim[1], dim[0])
            rel_target = utils.scale_position(rel_target, dim[1], dim[0])
            rel_prediction = utils.scale_position(rel_prediction, dim[1], dim[0])

        f, (s1, s2) = plt.subplots(1, 2, sharey='all', sharex='all')
        frame_utils.draw_frame(s1, target, prediction, dim)
        frame_utils.draw_frame(s2, rel_target, rel_prediction, dim)

    s1.set_title(
        'Sample ' + str(error_stats[1][0]) + '\n' + 'Max. absolute error = ' + str(error_stats[1][1]) + '\n' +
        'Error mean = ' + "{0:.4f}".format(error_stats[0]))

    s2.set_title(
        'Sample ' + str(rel_error_stats[1][0]) + '\n' + 'Max. relative error = ' + str(rel_error_stats[1][1]) +
        '%' + '\n' + 'Relative error mean = ' + "{0:.4f}".format(rel_error_stats[0]) + '%')


def draw_error_breakdown(error_stats, x_error_stats, y_error_stats,
                         rel_error_stats, rel_x_error_stats, rel_y_error_stats):

    if x_error_stats[0] is not None:
        abs_error = [error_stats[0], error_stats[1][1]]
        rel_error = [rel_error_stats[0], rel_error_stats[1][1]]
        f, (s1, s2) = plt.subplots(1, 2, sharey='all', sharex='all')
        draw_bar_error(s1, abs_error, x_error_stats, y_error_stats, "Absolute error", [None, None, None])
        draw_bar_error(s2, rel_error, rel_x_error_stats, rel_y_error_stats, "Relative error", ["global", "x", "y"])
        f.legend()

    plt.show()


def draw_bar_error(fig, glob_val, x_val, y_val, title, labels):
    x = ['Mean', 'Max']
    x_pos = np.arange(len(x))
    width = 0.25

    fig.bar(x_pos - width, glob_val, width, color="darkslategray", label=labels[0])
    fig.bar(x_pos, x_val, width, color="darkcyan", label=labels[1])
    fig.bar(x_pos + width, y_val, width, color="darkturquoise", label=labels[2])

    fig.set_title(title)
    fig.set_xticks(x_pos)
    fig.set_xticklabels(x)

    return fig
