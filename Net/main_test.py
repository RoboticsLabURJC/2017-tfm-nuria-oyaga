"""

TFM - main_test.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"


from utils import read_data, reshape_data, get_config_file
import net

if __name__ == '__main__':
    conf = get_config_file()

    # Load data
    parameters, test_set = read_data(conf['data_path'])

    print('Puting the test data into the right shape...')
    testX, testY = reshape_data(test_set)

    to_test_net = net.Mlp(model_file=conf['model_path'])

    to_test_net.test(testX, testY, float(parameters[0][3]))
