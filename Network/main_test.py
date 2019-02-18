"""

TFM - main_test.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"

from Utils import utils
from Network import Net

if __name__ == '__main__':
    conf = utils.get_config_file()

    data_type = conf['data_path'].split('/')[5]

    print(data_type)
    # Load data
    if data_type == "Functions_dataset":
        parameters, test_set = utils.read_function_data(conf['data_path'])
        gap = float(parameters[0][3])

        print('Puting the test data into the right shape...')
        testX, testY = utils.reshape_function_data(test_set)

        to_test_net = Net.Mlp(model_file=conf['model_path'])

    else: # data_type == "Vectors_dataset":
        parameters, test_set = utils.read_vector_data(conf['data_path'])
        gap = parameters.iloc[0]['gap']

        print('Puting the test data into the right shape...')
        testX, testY = utils.reshape_vector_data(test_set)

        to_test_net = Net.Convolution1D(model_file=conf['model_path'])

    to_test_net.test(testX, testY, gap, data_type)
