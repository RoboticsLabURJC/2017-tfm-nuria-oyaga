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


def draw_points(fig, real_data, pred_data, error, point, e_type):
    """
    Draw the real point and the predicted point
    :param fig: Where draw
    :param real_data: Real value of the data to predict
    :param pred_data: Prediction result
    :param error: Error commited in the prediction
    :param point: x position
    :param e_type: Error type
    """
    fig.scatter(point, real_data, color='green', label='Real')
    fig.scatter(point, pred_data, color='red', label='Prediction')

    if e_type == 'a':
        label = 'Abs. error = ' + str(error)
        limit = [real_data - (error * 2),
                 real_data + (error * 2)]
        fig.set_title('Absolute error between prediction and real')
    else:
        label = 'Rel. error = ' + str(error) + '%'
        limit = [real_data - ((error/100) * 2),
                 real_data + ((error/100) * 2)]
        fig.set_title('Relative error between prediction and real')

    fig.plot([point, point],
            [real_data, pred_data],
            '--',
            color='black',
            label=label)

    fig.set_ylim(limit[0], limit[1])
    fig.set_xlim(point-1, point+1)

    fig.legend()


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


def percentage_of_errors(error, max):
    """
    Calculate the percentage of errors commited in each interval (between 0 and maximum)
    :param error: Error array
    :param max: Maximum value
    :return: Percentage in each interval and label for each interval
    """
    # Calculate error percentage
    pct = lambda x: (x / len(error)) * 100

    interval = np.linspace(0,max,5)
    print(interval)

    pct_error = [pct(np.sum((error > interval[0]) & (error < interval[1]))),
                 pct(np.sum((error > interval[1]) & (error < interval[2]))),
                 pct(np.sum((error > interval[2]) & (error < interval[3]))),
                 pct(np.sum((error > interval[3])))]

    print(pct_error)

    labels = [interval2string([interval[0], interval[1]]),
              interval2string([interval[1], interval[2]]),
              interval2string([interval[2], interval[3]]),
              interval2string([interval[3]])
              ]

    return pct_error, labels


def draw_percentage(f, pct, labels):
    """
    Draw the percentage of errors in each interval
    :param f: Figure where draw
    :param pct: Percentages
    :param labels: Labels for each interval
    """
    x = range(len(pct))
    y = pct

    f.bar(x, y, color='red')

    f.set_ylim((0, 100))
    f.set_xticks(x)
    f.set_xticklabels(labels[0])
    f.set_xlabel(labels[1])
    f.set_ylabel("% over total errors")


if __name__ == '__main__':
    # Load data
    parameters, test_set = data_utils.read_data('functions_dataset/linear_10_[None]_test.txt')
    gap = float(parameters[0][3])

    # Put the test data into the right shape
    print('Puting the test data into the right shape...')
    testX, testY = data_utils.reshape_data(test_set)

    # Load model
    model = load_model('Models/linear_10_None_PredictorMultilayerPerceptron100.h5')

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

    print('Error mean = ', "{0:.4f}".format(error_stats[0]))
    print('Relative error mean = ', "{0:.4f}".format(rel_error_stats[0]), '%')

    # Draw error percentage
    '''
    f, (s1, s2) = plt.subplots(2, 1)

    pct, labels = percentage_of_errors(error, error_stats[1][1])
    draw_percentage(s1, pct, [labels, 'Interval of error committed'])

    rel_pct, rel_labels = percentage_of_errors(relative_error, rel_error_stats[1][1])
    draw_percentage(s2, rel_pct, [rel_labels, 'Interval of relative error committed (%)'])

    s1.set_title("Percentage of errors committed")

    plt.show()
    '''

    # Draw the max errors points
    f, ((s1, s2),(s3,s4)) = plt.subplots(2, 2)

    draw_function(s1,
                  [testX[error_stats[1][0]],testY[error_stats[1][0]]],
                  prediction[error_stats[1][0]],
                  gap)

    s1.set_title('Sample ' + str(error_stats[1][0]))

    draw_points(s2,
                testY[error_stats[1][0]],
                prediction[error_stats[1][0]],
                error_stats[1][1],
                len(testX[error_stats[1][0]]) - 1 + gap,
                'a')

    draw_function(s3,
                  [testX[rel_error_stats[1][0]],testY[rel_error_stats[1][0]]],
                  prediction[rel_error_stats[1][0]],
                  gap)

    s3.set_title('Sample ' + str(rel_error_stats[1][0]))

    draw_points(s4,
                testY[rel_error_stats[1][0]],
                prediction[rel_error_stats[1][0]],
                rel_error_stats[1][1],
                len(testX[rel_error_stats[1][0]]) - 1 + gap,
                'r')

    plt.show()