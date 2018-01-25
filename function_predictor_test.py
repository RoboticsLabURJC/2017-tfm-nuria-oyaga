"""
function_predictor_test.py: A script to test a model of predictor

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"

# Keras
from keras.models import load_model
# Others
import numpy as np
from matplotlib import pyplot as plt
import data_utils


def draw_function(fig, real_data, pred_data, gap):
    """
    Draw the data (function values) and the prediction result
    :param fig: Where draw the data
    :param real_data: Function to draw
    :param pred_data: Prediction result
    :param gap: Separation between known data and to predict
    """
    data_utils.draw_data(fig, real_data, gap)
    fig.scatter(len(real_data[0]) - 1 + gap, pred_data, 5, color='red')


def calculate_error(real, prediction):
    """
    Calculate error between real data and prediction
    :param real: Real data
    :param prediction: Prediction data
    :return: Error and relative error for each sample
    """
    # Calculate error
    error = np.array([abs(real[i] - prediction[i]) for i in range(real.size - 1)])

    # Calculate relative error
    relative_error = np.zeros(error.size)
    for i in range(relative_error.size - 1):
        if real[i] == 0:
            relative_error[i] = 'nan'
        else:
            relative_error[i] = np.array((error[i] / abs(real[i])) * 100)

    return error, relative_error


def get_errors_statistics(error, relative_error):
    """
    Get statistics from the error: mean and maximum value
    :param error: Error list
    :param relative_error: Relative error list
    :return: Mean and maximum (value and position) for each kind of error
    """
    # Locate nan in relative error
    index = np.argwhere(np.isnan(relative_error) == 1)

    # Max error
    max_error_index = np.argmax(error)
    max_error = round(float(error[max_error_index]), 3)

    # Replace Nan to 0 for calculate max
    relative_error_nanTo0 = relative_error
    for i in index:
        relative_error_nanTo0[i] = 0

    max_rel_error_index = np.argmax(relative_error_nanTo0)
    max_rel_error = round(float(relative_error_nanTo0[max_rel_error_index]), 3)

    # Calculate error mean
    error_mean = np.sum(error) / error.size

    # Remove nan to calculate relative error mean
    relative_error = np.delete(relative_error, index)

    # Calculate relative error mean
    relative_error_mean = np.sum(relative_error) / relative_error.size

    return [error_mean, [max_error_index, max_error]],\
           [relative_error_mean, [max_rel_error_index, max_rel_error]]


def interval2string(interval):
    """
    Transform an interval to a string to show labels in a figure
    :param interval: Interval to transform
    :return: String corresponds to the label
    """
    if len(interval) == 2:
        string = '[' + "{0:.3f}".format(interval[0]) + ',' + "{0:.3f}".format(interval[1]) + ')'
    else:
        string = '>' + "{0:.3f}".format(interval[0])

    return string


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

    x = range(len(hist))
    y = hist

    plt.bar(x, y, color='red')

    plt.xticks(x, labels)
    plt.ylabel("Number errors")
    plt.xlabel('Relative error intervals')
    plt.title('Relative error histogram')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()


if __name__ == '__main__':
    # Load data
    parameters, test_set = data_utils.read_data('functions_dataset/linear/linear_10_[None]_val.txt')

    gap = float(parameters[0][3])

    # Put the test data into the right shape
    print('Puting the test data into the right shape...')
    testX, testY = data_utils.reshape_data(test_set)

    # Load model
    model = load_model('Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                       '10_2/10_2_N_relu_mean_squared_error.h5')

    # Generate predictions
    prediction = model.predict(testX)

    # To make prediction of a sample
    # sample = np.array(testX[12]).reshape(1,20)
    # pred = model.predict(sample)
    # print(pred)

    # Calculate errors
    error, relative_error = calculate_error(testY, prediction)

    # Calculate stats
    error_stats, rel_error_stats = get_errors_statistics(error, relative_error)

    # Draw error percentage
    plt.figure()
    error_histogram(relative_error)

    # Draw the max errors points
    f, ((s1, s2)) = plt.subplots(1, 2)

    draw_function(s1,
                  [testX[error_stats[1][0]], testY[error_stats[1][0]]],
                  prediction[error_stats[1][0]],
                  gap)

    s1.set_title('Sample ' + str(error_stats[1][0]) + '\n' + 'Max. absolute error = ' + str(error_stats[1][1]))

    draw_function(s2,
                  [testX[rel_error_stats[1][0]],testY[rel_error_stats[1][0]]],
                  prediction[rel_error_stats[1][0]],
                  gap)

    s2.set_title('Sample ' + str(rel_error_stats[1][0]) + '\n' + 'Max. relative error = ' +
                 str(rel_error_stats[1][1]) + '%')
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 12,
            }
    plt.text(-30, -15, 'Error mean = ' + "{0:.4f}".format(error_stats[0]), fontdict=font)
    plt.text(7, -15, 'Relative error mean = ' + "{0:.4f}".format(rel_error_stats[0]) + '%', fontdict=font)
    plt.axis('equal')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show()