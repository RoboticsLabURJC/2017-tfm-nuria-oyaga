from Utils import func_utils, vect_utils, frame_utils, utils

from matplotlib import pyplot as plt
import numpy as np


def calculate_error(real, prediction, maximum):
    # Calculate error
    if len(real.shape) > 1:
        error = np.array([np.linalg.norm(np.array(real[i]) - np.array(prediction[i]))
                          for i in range(real.shape[0])])
        y_error = np.array([abs(real[i][0] - prediction[i][0]) for i in range(real.shape[0])])
        x_error = np.array([abs(real[i][1] - prediction[i][1]) for i in range(real.shape[0])])

    else:
        error = np.array([abs(real[i] - prediction[i]) for i in range(real.size)])
        x_error = None
        y_error = None

    # Calculate relative error
    relative_error = np.array([((error[i] / abs(maximum[i])) * 100) for i in range(error.size)])

    return error, x_error, y_error, relative_error


def get_error_stats(test_x, test_y, predict, gap, data_type, dim, error, x_error, y_error, relative_error, figures_dir):

    figures_dir += "error_stats/"
    # figures_dir = "error_stats/"
    utils.check_dirs(figures_dir, True)

    error_stats, x_error_stats, y_error_stats, \
        rel_error_stats, rel_x_error_stats, rel_y_error_stats = calculate_errors_statistics(error, x_error, y_error,
                                                                                            relative_error, dim)

    # Errors histograms
    error_histogram(error, figures_dir)
    relative_error_histogram(relative_error, figures_dir)

    # Draw max error
    draw_max_error_samples(test_x, test_y, predict, gap, error_stats, rel_error_stats, data_type, dim, figures_dir)
    figures = ["abs_hist", "rel_hist", "max_error"]

    # Draw error breakdowns
    if x_error_stats[0] is not None:
        draw_error_breakdown(error, x_error, y_error, relative_error, dim, figures_dir)
        figures.insert(2, "breakdown")

    # Draw real vs predictions
    draw_pred_vs_real(test_y, predict, dim, figures_dir)

    utils.combine_figures(figures_dir, figures)


def calculate_errors_statistics(error, x_error, y_error, relative_error, dim):
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


def error_histogram(error, fig_dir):

    hist = [sum((error >= 0) & (error < 1)),
            sum((error >= 1) & (error < 2)),
            sum((error >= 2) & (error < 5)),
            sum((error >= 5) & (error < 10)),
            sum((error >= 10) & (error < 20)),
            sum((error >= 20) & (error < 40)),
            sum((error >= 40) & (error < 80)),
            sum((error >= 80))]

    labels = ['[0, 1)', '[1, 2)', '[2, 5)', '[5, 10)', '[10, 20)', '[20, 40)', '[40, 80)', '>80']

    fig = plt.figure()
    x = range(len(hist))
    y = hist

    plt.bar(x, y, color='powderblue', edgecolor='black')

    plt.xticks(x, labels)
    plt.ylabel("Number errors")
    plt.xlabel('Error intervals')
    plt.title('Error histogram')

    fig.savefig(fig_dir + 'abs_hist.png')


def relative_error_histogram(error, fig_dir):

    hist = [sum((error >= 0) & (error < 10)),
            sum((error >= 10) & (error < 20)),
            sum((error >= 20) & (error < 30)),
            sum((error >= 30) & (error < 40)),
            sum((error >= 40) & (error < 60)),
            sum((error >= 60) & (error < 80)),
            sum((error >= 80) & (error < 100)),
            sum((error >= 100))]

    labels = ['<10', '[10,20)', '[20,30)', '[30,40)', '[40,60)', '[60,80)', '[80,100)', '>100']

    fig = plt.figure()
    x = range(len(hist))
    y = hist

    plt.bar(x, y, color='powderblue', edgecolor='black')

    plt.xticks(x, labels)
    plt.ylabel("Number errors")
    plt.xlabel('Relative error intervals')
    plt.title('Relative error histogram')

    fig.savefig(fig_dir + 'rel_hist.png')


def draw_max_error_samples(test_x, test_y, predict, gap, error_stats, rel_error_stats, data_type, dim, fig_dir):

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

        f, (s1, s2) = plt.subplots(1, 2, sharey='all', sharex='all')
        frame_utils.draw_frame(s1, target, prediction, dim)
        frame_utils.draw_frame(s2, rel_target, rel_prediction, dim)

    s1.set_title(
        'Sample ' + str(error_stats[1][0]) + '\n' + 'Max. absolute error = ' + str(error_stats[1][1]) + '\n' +
        'Error mean = ' + "{0:.4f}".format(error_stats[0]))

    s2.set_title(
        'Sample ' + str(rel_error_stats[1][0]) + '\n' + 'Max. relative error = ' + str(rel_error_stats[1][1]) +
        '%' + '\n' + 'Relative error mean = ' + "{0:.4f}".format(rel_error_stats[0]) + '%')

    f.savefig(fig_dir + 'max_error.png')


def draw_error_breakdown(error, x_error, y_error, relative_error, dim, fig_dir):

    f, (s1, s2) = plt.subplots(1, 2, sharex='all')
    draw_boxplot_error(s1, [error, x_error, y_error], "Absolute error")
    draw_boxplot_error(s2,
                       [relative_error, np.round((x_error/dim[1])*100, 3), np.round((y_error/dim[0])*100, 3)],
                       "Relative error (%)")

    f.savefig(fig_dir + 'breakdown.png')


def draw_boxplot_error(fig, data, title):
    median_props = dict(linestyle=':', linewidth=1, color='black')
    mean_line_props = dict(linestyle='-', linewidth=2, color='grey')

    b_plot = fig.boxplot(data, patch_artist=True, labels=["Global", "Dim. x", "Dim. y"], notch=True,
                         medianprops=median_props, meanprops=mean_line_props, meanline=True, showmeans=True)
    colors = ["lightcyan", "lavender", "thistle"]
    for patch, color in zip(b_plot['boxes'], colors):
        patch.set_facecolor(color)

    fig.set_title(title)
    fig.yaxis.grid(True)


def draw_pred_vs_real(real, predict, dim, fig_dir):
    x_real = [val[1] for val in real]
    x_predict = [val[1] for val in predict]
    y_real = [val[0] for val in real]
    y_predict = [val[0] for val in predict]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_real, x_predict, c='powderblue')
    ax1.plot([0, dim[1]], [0, dim[1]], 'k--', lw=2)
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Predicted')
    ax1.set_title("X")
    ax1.grid(True)

    ax2.scatter(y_real, y_predict, c='powderblue')
    ax2.plot([0, dim[0]], [0, dim[0]], 'k--', lw=2)
    ax2.set_xlabel('Real')
    ax2.set_title("Y")
    ax2.grid(True)

    fig.savefig(fig_dir + 'Real_VS_Pred.png')



