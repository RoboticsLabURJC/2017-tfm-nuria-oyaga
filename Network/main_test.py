"""

TFM - main_test.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"

import sys
sys.path.insert(0, '/home/docker/2017-tfm-nuria-oyaga')

from Utils import utils, func_utils, vect_utils, frame_utils
from Network import Net

if __name__ == '__main__':
    conf = utils.get_config_file()

    data_type = conf['data_path'].split('/')[5]
    net_type = conf['model_path'].split('/')[6]

    print(data_type)
    # Load data
    if data_type == "Functions_dataset":
        parameters, test_set = func_utils.read_function_data(conf['data_path'])
        gap = float(parameters[0][3])

        print('Puting the test data into the right shape...')
        testX, testY = func_utils.reshape_function_data(test_set)

        to_test_net = Net.Mlp(model_file=conf['model_path'])

    elif data_type == "Vectors_dataset":
        parameters, test_set = vect_utils.read_vector_data(conf['data_path'])
        gap = parameters.iloc[0]['gap']

        print('Puting the test data into the right shape...')
        testX, testY = vect_utils.reshape_vector_data(test_set)
        if net_type == "NOREC":
            to_test_net = Net.Convolution1D(model_file=conf['model_path'])
        else:
            to_test_net = Net.Lstm(model_file=conf['model_path'])

    else:  # data_type == "Frames_dataset
        parameters, test_set = frame_utils.read_frame_data(conf['data_path'])
        gap = parameters.iloc[0]['gap']

        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            testX, testY = frame_utils.reshape_frame_data(test_set)
            to_test_net = Net.Convolution2D(model_file=conf['model_path'])
        else:
            print('Puting the test data into the right shape...')
            testX, testY = frame_utils.reshape_frame_data(test_set, True)
            to_test_net = Net.ConvolutionLstm(model_file=conf['model_path'])

    to_test_net.test(testX, testY, gap, data_type)
