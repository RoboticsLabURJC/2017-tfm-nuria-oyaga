"""

TFM - main_test.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"


from utils import read_data, reshape_data
import net

if __name__ == '__main__':
    # Load data
    parameters, test_set = read_data('/home/nuria/Documents/MOVA/TFM/functions_dataset/linear/'
                                     'linear_10_[None]_test.txt')

    print('Puting the test data into the right shape...')
    testX, testY = reshape_data(test_set)

    to_test_net = net.Mlp(model_file='/home/nuria/Documents/MOVA/TFM/Models/MLP/linear/'
                                     'LINEAR_10_None_Predictor_15_False_relu_mean_squared_error.h5')

    to_test_net.test(testX, testY, float(parameters[0][3]))
