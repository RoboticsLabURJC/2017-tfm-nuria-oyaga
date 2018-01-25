"""

compare_models.py: A script to compare diferent models

"""
__author__ = "Nuria Oyaga"
__date__ = "2018/01/02"

import data_utils
import function_predictor_test
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np


def read_model_info(file):
    f = open(file)
    lines = f.readlines()
    info = []
    for line in lines:
        info.append(line.split(':')[1].split(' ')[1].split('\n')[0])

    return info


def draw_results(data, info):
    abs_ticks = []
    rel_ticks = []
    fig, ax = plt.subplots(1, 2)
    for i, element in enumerate(data):
        inf = info[i]
        '''label = 'Architecture: ' + inf[0] + '\n' + 'Activation: ' + inf[1] + '\n' + 'Loss: ' + inf[2] + '\n' + \
                'Dropout: ' + inf[3] + '\n' + 'Epochs: ' + inf[4] + '\n' + 'Batch-size: ' + inf[5]'''
        label = 'Epochs: ' + inf[4]
        ax[0].bar(i, element[0][0], 1)
        ax[1].bar(i, element[1][0], 1, label=label)
        abs_ticks.append('Max. sample:\n' + str(element[0][1][0]))
        rel_ticks.append('Max. sample:\n' + str(element[1][1][0]))
    print(abs_ticks)

    ax[0].set_xticks(range(len(data)))
    ax[0].xaxis.set_ticks_position('none')
    ax[0].set_xticklabels(abs_ticks, fontsize=10)
    ax[0].set_title('Absolute error')
    ax[1].set_xticks(range(len(data)))
    ax[1].xaxis.set_ticks_position('none')
    ax[1].set_xticklabels(rel_ticks, fontsize=10)
    ax[1].set_title('Relative error (%)')

    plt.legend()
    plt.suptitle('Architecture: ' + inf[0] + '; Activation: ' + inf[1] + '; Loss: ' + inf[2] + '; Dropout: ' + inf[3] +
                 '; Batch-size: ' + inf[5] + '; Train data: ' + inf[6])


if __name__ == '__main__':
    # Load data
    test_files = ['functions_dataset/linear/linear_10_[None]_val.txt',
                  'functions_dataset/linear/linear_10_[None]_val.txt',
                  'functions_dataset/linear/linear_10_[None]_val.txt']

    net_files = ['Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                 '100_2/100_2_N_relu_mean_squared_error.h5',
                 'Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                 '20_2/20_2_N_relu_mean_squared_error.h5',
                 'Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                 '10_2/10_2_N_relu_mean_squared_error.h5'
                 ]

    info_files = ['Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                  '100_2/100_2_N_relu_mean_squared_error_info.txt',
                  'Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                  '20_2/20_2_N_relu_mean_squared_error_info.txt',
                  'Models/linear_10_None_PredictorMultilayerPerceptron1_[10]_relu_mean_squared_error/'
                  '10_2/10_2_N_relu_mean_squared_error_info.txt'
                  ]

    stats = []
    infos = []
    for i in range(len(net_files)):
        parameters, test_set = data_utils.read_data(test_files[i])


        # Put the test data into the right shape
        print('Puting the test data into the right shape...')
        testX, testY = data_utils.reshape_data(test_set)

        # Load models
        model = load_model(net_files[i])

        # Load models info
        info = read_model_info(info_files[i])

        # Generate predictions
        prediction = model.predict(testX)

        # Calculate errors
        error, relative_error = function_predictor_test.calculate_error(testY, prediction)

        # Calculate stats
        error_stats, rel_error_stats = function_predictor_test.get_errors_statistics(error, relative_error)
        stats.append([error_stats, rel_error_stats])
        infos.append(info)

    draw_results(stats, infos)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()